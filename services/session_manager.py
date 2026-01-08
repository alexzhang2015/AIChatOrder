"""会话管理器"""

import logging
import threading
import time
import uuid
from typing import Dict, Optional

from infrastructure.database import Database, SessionRepository, MessageRepository, MessageModel
from models.session import Session, ConversationState

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器（线程安全，支持数据库持久化）"""

    def __init__(self, use_db: bool = True):
        """初始化会话管理器

        Args:
            use_db: 是否使用数据库持久化，默认 True
        """
        self.sessions: Dict[str, Session] = {}  # 内存缓存
        self.session_timeout = 1800  # 30分钟超时
        self._lock = threading.Lock()  # 线程锁
        self.use_db = use_db

        if use_db:
            try:
                self._db = Database()
                self._session_repo = SessionRepository(self._db)
                self._message_repo = MessageRepository(self._db)
                logger.info("会话管理器已启用数据库持久化")
            except Exception as e:
                logger.warning(f"数据库初始化失败，使用纯内存模式: {e}")
                self.use_db = False

    def create_session(self) -> Session:
        """创建新会话"""
        with self._lock:
            session_id = str(uuid.uuid4())[:8]
            session = Session(session_id=session_id)
            self.sessions[session_id] = session

            # 持久化到数据库
            if self.use_db:
                try:
                    self._session_repo.create(session_id)
                except Exception as e:
                    logger.error(f"会话持久化失败: {e}")

            logger.debug(f"创建会话: {session_id}")
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        with self._lock:
            # 先从内存缓存获取
            session = self.sessions.get(session_id)

            if session:
                # 检查是否过期
                if (time.time() - session.created_at) > self.session_timeout:
                    self._delete_session_internal(session_id)
                    return None
                return session

            # 内存中没有，尝试从数据库恢复
            if self.use_db:
                try:
                    db_session = self._session_repo.get(session_id)
                    if db_session:
                        # 检查是否过期
                        if (time.time() - db_session.updated_at) > self.session_timeout:
                            self._session_repo.delete(session_id)
                            return None

                        # 恢复到内存
                        session = Session(
                            session_id=db_session.session_id,
                            state=ConversationState(db_session.state),
                            created_at=db_session.created_at
                        )

                        # 恢复消息历史
                        messages = self._message_repo.get_by_session(session_id)
                        for msg in messages:
                            session.history.append({
                                "role": msg.role,
                                "content": msg.content,
                                "intent_info": {
                                    "intent": msg.intent,
                                    "confidence": msg.confidence,
                                    "slots": msg.slots
                                } if msg.intent else None,
                                "timestamp": msg.timestamp
                            })

                        self.sessions[session_id] = session
                        return session
                except Exception as e:
                    logger.error(f"从数据库恢复会话失败: {e}")

            return None

    def update_session(self, session: Session):
        """更新会话（持久化）"""
        with self._lock:
            self.sessions[session.session_id] = session

            if self.use_db:
                try:
                    db_session = self._session_repo.get(session.session_id)
                    if db_session:
                        db_session.state = session.state.value
                        db_session.current_order_id = session.current_order.order_id if session.current_order else None
                        self._session_repo.update(db_session)
                except Exception as e:
                    logger.error(f"更新会话失败: {e}")

    def add_message(self, session_id: str, role: str, content: str,
                   intent: str = None, confidence: float = None, slots: Dict = None):
        """添加消息并持久化"""
        if self.use_db:
            try:
                message = MessageModel(
                    session_id=session_id,
                    role=role,
                    content=content,
                    intent=intent,
                    confidence=confidence,
                    slots=slots
                )
                self._message_repo.add(message)
            except Exception as e:
                logger.error(f"消息持久化失败: {e}")

    def delete_session(self, session_id: str):
        """删除会话"""
        with self._lock:
            self._delete_session_internal(session_id)

    def _delete_session_internal(self, session_id: str):
        """内部删除会话（不加锁）"""
        if session_id in self.sessions:
            del self.sessions[session_id]

        if self.use_db:
            try:
                self._session_repo.delete(session_id)
            except Exception as e:
                logger.error(f"删除会话失败: {e}")

        logger.debug(f"删除会话: {session_id}")

    def cleanup_expired(self) -> int:
        """清理过期会话"""
        with self._lock:
            # 清理内存中的过期会话
            expired = []
            for session_id, session in self.sessions.items():
                if (time.time() - session.created_at) > self.session_timeout:
                    expired.append(session_id)

            for session_id in expired:
                del self.sessions[session_id]

            # 清理数据库中的过期会话
            db_cleaned = 0
            if self.use_db:
                try:
                    db_cleaned = self._session_repo.cleanup_expired(self.session_timeout)
                except Exception as e:
                    logger.error(f"清理过期会话失败: {e}")

            total = len(expired) + db_cleaned
            if total > 0:
                logger.info(f"清理了 {total} 个过期会话")
            return total
