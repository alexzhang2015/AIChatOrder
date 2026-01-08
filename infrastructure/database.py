"""
SQLite 数据库持久化层

提供会话、订单、消息的 CRUD 操作。
"""

import json
import sqlite3
import threading
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

from exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    SessionNotFoundError,
    OrderNotFoundError
)

logger = logging.getLogger(__name__)

# 默认数据库路径 (项目根目录的 data 文件夹)
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "coffee_order.db"


@dataclass
class SessionModel:
    """会话数据模型"""
    session_id: str
    state: str = "greeting"
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    current_order_id: Optional[str] = None
    pending_item: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("pending_item"):
            data["pending_item"] = json.dumps(data["pending_item"], ensure_ascii=False)
        return data

    @classmethod
    def from_row(cls, row: Dict) -> "SessionModel":
        pending_item = row.get("pending_item")
        if pending_item and isinstance(pending_item, str):
            row["pending_item"] = json.loads(pending_item)
        return cls(**row)


@dataclass
class OrderModel:
    """订单数据模型"""
    order_id: str
    session_id: str
    status: str = "pending"
    total: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict) -> "OrderModel":
        return cls(**row)


@dataclass
class OrderItemModel:
    """订单项数据模型"""
    id: Optional[int] = None
    order_id: str = ""
    product_name: str = ""
    size: str = "中杯"
    temperature: str = "热"
    sweetness: str = "标准"
    milk_type: str = "全脂奶"
    extras: List[str] = field(default_factory=list)
    quantity: int = 1
    price: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["extras"] = json.dumps(data["extras"], ensure_ascii=False)
        return data

    @classmethod
    def from_row(cls, row: Dict) -> "OrderItemModel":
        extras = row.get("extras", "[]")
        if isinstance(extras, str):
            row["extras"] = json.loads(extras)
        return cls(**row)


@dataclass
class MessageModel:
    """消息数据模型"""
    id: Optional[int] = None
    session_id: str = ""
    role: str = ""  # user, assistant
    content: str = ""
    intent: Optional[str] = None
    confidence: Optional[float] = None
    slots: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("slots"):
            data["slots"] = json.dumps(data["slots"], ensure_ascii=False)
        return data

    @classmethod
    def from_row(cls, row: Dict) -> "MessageModel":
        slots = row.get("slots")
        if slots and isinstance(slots, str):
            row["slots"] = json.loads(slots)
        return cls(**row)


class Database:
    """SQLite 数据库管理器（单例模式，优化的连接管理）

    特性:
    - 线程本地连接池（每线程一个连接）
    - WAL 模式支持更好的并发
    - 连接健康检查
    - 统计跟踪
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return

        self.db_path = db_path or DEFAULT_DB_PATH
        self._local = threading.local()
        self._write_lock = threading.Lock()

        # 连接统计
        self._connection_count = 0
        self._stats_lock = threading.Lock()

        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库表
        self._init_tables()
        self._initialized = True

        logger.info(f"数据库初始化完成: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接（带健康检查）"""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            try:
                conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=30.0,
                    isolation_level=None  # 自动提交模式，配合 WAL
                )
                conn.row_factory = sqlite3.Row

                # 优化 PRAGMA 设置
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
                conn.execute("PRAGMA synchronous = NORMAL")  # 平衡性能和安全
                conn.execute("PRAGMA cache_size = -64000")  # 64MB 缓存
                conn.execute("PRAGMA temp_store = MEMORY")  # 临时表存内存

                self._local.connection = conn

                with self._stats_lock:
                    self._connection_count += 1

                logger.debug(f"创建新数据库连接 (总连接数: {self._connection_count})")

            except sqlite3.Error as e:
                raise DatabaseConnectionError(f"数据库连接失败: {e}")

        # 连接健康检查
        try:
            self._local.connection.execute("SELECT 1")
        except sqlite3.Error:
            logger.warning("数据库连接已断开，正在重连...")
            self._local.connection = None
            return self._get_connection()

        return self._local.connection

    def stats(self) -> Dict:
        """获取数据库统计"""
        with self._stats_lock:
            return {
                "db_path": str(self.db_path),
                "connection_count": self._connection_count,
                "initialized": self._initialized
            }

    @contextmanager
    def get_cursor(self, write: bool = False):
        """获取数据库游标的上下文管理器

        Args:
            write: 是否是写操作（需要加锁）
        """
        conn = self._get_connection()

        if write:
            self._write_lock.acquire()

        try:
            cursor = conn.cursor()
            yield cursor
            if write:
                conn.commit()
        except sqlite3.Error as e:
            if write:
                conn.rollback()
            raise DatabaseQueryError(f"数据库操作失败: {e}")
        finally:
            if write:
                self._write_lock.release()

    def _init_tables(self):
        """初始化数据库表"""
        with self.get_cursor(write=True) as cursor:
            # 会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'greeting',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    current_order_id TEXT,
                    pending_item TEXT
                )
            """)

            # 订单表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    total REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # 订单项表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    size TEXT NOT NULL DEFAULT '中杯',
                    temperature TEXT NOT NULL DEFAULT '热',
                    sweetness TEXT NOT NULL DEFAULT '标准',
                    milk_type TEXT NOT NULL DEFAULT '全脂奶',
                    extras TEXT NOT NULL DEFAULT '[]',
                    quantity INTEGER NOT NULL DEFAULT 1,
                    price REAL NOT NULL DEFAULT 0.0,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id)
                )
            """)

            # 消息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    intent TEXT,
                    confidence REAL,
                    slots TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_session
                ON orders(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_order_items_order
                ON order_items(order_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at)
            """)

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    @classmethod
    def reset_instance(cls):
        """重置单例实例（用于测试）"""
        with cls._lock:
            if cls._instance:
                cls._instance.close()
            cls._instance = None


class SessionRepository:
    """会话数据仓库"""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._lock = threading.Lock()

    def create(self, session_id: str) -> SessionModel:
        """创建新会话"""
        with self._lock:
            session = SessionModel(session_id=session_id)
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("""
                    INSERT INTO sessions (session_id, state, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (session.session_id, session.state, session.created_at, session.updated_at))
            logger.debug(f"创建会话: {session_id}")
            return session

    def get(self, session_id: str) -> Optional[SessionModel]:
        """获取会话"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if row:
                return SessionModel.from_row(dict(row))
            return None

    def get_or_create(self, session_id: str) -> SessionModel:
        """获取或创建会话"""
        session = self.get(session_id)
        if session:
            return session
        return self.create(session_id)

    def update(self, session: SessionModel) -> SessionModel:
        """更新会话"""
        with self._lock:
            session.updated_at = datetime.now().timestamp()
            pending_item_json = json.dumps(session.pending_item, ensure_ascii=False) if session.pending_item else None

            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("""
                    UPDATE sessions
                    SET state = ?, updated_at = ?, current_order_id = ?, pending_item = ?
                    WHERE session_id = ?
                """, (
                    session.state,
                    session.updated_at,
                    session.current_order_id,
                    pending_item_json,
                    session.session_id
                ))
            logger.debug(f"更新会话: {session.session_id}")
            return session

    def delete(self, session_id: str):
        """删除会话及其关联数据"""
        with self._lock:
            with self.db.get_cursor(write=True) as cursor:
                # 删除消息
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                # 获取关联订单
                cursor.execute("SELECT order_id FROM orders WHERE session_id = ?", (session_id,))
                order_ids = [row["order_id"] for row in cursor.fetchall()]
                # 删除订单项
                for order_id in order_ids:
                    cursor.execute("DELETE FROM order_items WHERE order_id = ?", (order_id,))
                # 删除订单
                cursor.execute("DELETE FROM orders WHERE session_id = ?", (session_id,))
                # 删除会话
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            logger.debug(f"删除会话: {session_id}")

    def cleanup_expired(self, timeout_seconds: int = 1800) -> int:
        """清理过期会话

        Args:
            timeout_seconds: 超时时间（秒），默认30分钟

        Returns:
            删除的会话数量
        """
        cutoff = datetime.now().timestamp() - timeout_seconds
        with self._lock:
            with self.db.get_cursor(write=True) as cursor:
                # 获取过期会话
                cursor.execute("""
                    SELECT session_id FROM sessions WHERE updated_at < ?
                """, (cutoff,))
                expired_ids = [row["session_id"] for row in cursor.fetchall()]

                for session_id in expired_ids:
                    # 删除消息
                    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                    # 获取关联订单
                    cursor.execute("SELECT order_id FROM orders WHERE session_id = ?", (session_id,))
                    order_ids = [row["order_id"] for row in cursor.fetchall()]
                    # 删除订单项
                    for order_id in order_ids:
                        cursor.execute("DELETE FROM order_items WHERE order_id = ?", (order_id,))
                    # 删除订单
                    cursor.execute("DELETE FROM orders WHERE session_id = ?", (session_id,))
                    # 删除会话
                    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

                logger.info(f"清理了 {len(expired_ids)} 个过期会话")
                return len(expired_ids)

    def list_all(self, limit: int = 100) -> List[SessionModel]:
        """列出所有会话"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))
            return [SessionModel.from_row(dict(row)) for row in cursor.fetchall()]


class OrderRepository:
    """订单数据仓库"""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._lock = threading.Lock()

    def create(self, order_id: str, session_id: str) -> OrderModel:
        """创建新订单"""
        with self._lock:
            order = OrderModel(order_id=order_id, session_id=session_id)
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("""
                    INSERT INTO orders (order_id, session_id, status, total, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (order.order_id, order.session_id, order.status, order.total, order.created_at))
            logger.debug(f"创建订单: {order_id}")
            return order

    def get(self, order_id: str) -> Optional[OrderModel]:
        """获取订单"""
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
            if row:
                return OrderModel.from_row(dict(row))
            return None

    def get_by_session(self, session_id: str) -> List[OrderModel]:
        """获取会话的所有订单"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM orders WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
            return [OrderModel.from_row(dict(row)) for row in cursor.fetchall()]

    def update(self, order: OrderModel) -> OrderModel:
        """更新订单"""
        with self._lock:
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("""
                    UPDATE orders
                    SET status = ?, total = ?
                    WHERE order_id = ?
                """, (order.status, order.total, order.order_id))
            logger.debug(f"更新订单: {order.order_id}")
            return order

    def delete(self, order_id: str):
        """删除订单及其订单项"""
        with self._lock:
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("DELETE FROM order_items WHERE order_id = ?", (order_id,))
                cursor.execute("DELETE FROM orders WHERE order_id = ?", (order_id,))
            logger.debug(f"删除订单: {order_id}")

    def add_item(self, item: OrderItemModel) -> OrderItemModel:
        """添加订单项"""
        with self._lock:
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("""
                    INSERT INTO order_items
                    (order_id, product_name, size, temperature, sweetness, milk_type, extras, quantity, price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.order_id,
                    item.product_name,
                    item.size,
                    item.temperature,
                    item.sweetness,
                    item.milk_type,
                    json.dumps(item.extras, ensure_ascii=False),
                    item.quantity,
                    item.price
                ))
                item.id = cursor.lastrowid
            logger.debug(f"添加订单项: {item.product_name} -> {item.order_id}")
            return item

    def get_items(self, order_id: str) -> List[OrderItemModel]:
        """获取订单的所有项"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM order_items WHERE order_id = ?
            """, (order_id,))
            return [OrderItemModel.from_row(dict(row)) for row in cursor.fetchall()]

    def update_item(self, item: OrderItemModel) -> OrderItemModel:
        """更新订单项"""
        with self._lock:
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("""
                    UPDATE order_items
                    SET product_name = ?, size = ?, temperature = ?, sweetness = ?,
                        milk_type = ?, extras = ?, quantity = ?, price = ?
                    WHERE id = ?
                """, (
                    item.product_name,
                    item.size,
                    item.temperature,
                    item.sweetness,
                    item.milk_type,
                    json.dumps(item.extras, ensure_ascii=False),
                    item.quantity,
                    item.price,
                    item.id
                ))
            return item

    def delete_item(self, item_id: int):
        """删除订单项"""
        with self._lock:
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("DELETE FROM order_items WHERE id = ?", (item_id,))


class MessageRepository:
    """消息数据仓库"""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self._lock = threading.Lock()

    def add(self, message: MessageModel) -> MessageModel:
        """添加消息"""
        with self._lock:
            slots_json = json.dumps(message.slots, ensure_ascii=False) if message.slots else None
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("""
                    INSERT INTO messages
                    (session_id, role, content, intent, confidence, slots, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.session_id,
                    message.role,
                    message.content,
                    message.intent,
                    message.confidence,
                    slots_json,
                    message.timestamp
                ))
                message.id = cursor.lastrowid
            return message

    def get_by_session(self, session_id: str, limit: int = 100) -> List[MessageModel]:
        """获取会话的消息历史"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
            """, (session_id, limit))
            return [MessageModel.from_row(dict(row)) for row in cursor.fetchall()]

    def get_recent(self, session_id: str, count: int = 10) -> List[MessageModel]:
        """获取最近的消息"""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
            """, (session_id, count))
            return [MessageModel.from_row(dict(row)) for row in cursor.fetchall()]

    def delete_by_session(self, session_id: str):
        """删除会话的所有消息"""
        with self._lock:
            with self.db.get_cursor(write=True) as cursor:
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


# 便捷函数
def get_database(db_path: Optional[Path] = None) -> Database:
    """获取数据库实例"""
    return Database(db_path)


def get_session_repo(db: Optional[Database] = None) -> SessionRepository:
    """获取会话仓库"""
    return SessionRepository(db)


def get_order_repo(db: Optional[Database] = None) -> OrderRepository:
    """获取订单仓库"""
    return OrderRepository(db)


def get_message_repo(db: Optional[Database] = None) -> MessageRepository:
    """获取消息仓库"""
    return MessageRepository(db)
