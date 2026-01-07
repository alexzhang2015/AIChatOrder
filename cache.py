"""
缓存模块

提供 API 响应缓存和会话 TTL 缓存功能。
"""

import time
import hashlib
import logging
import threading
import asyncio
from typing import Dict, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass, field

from cachetools import TTLCache, LRUCache

logger = logging.getLogger(__name__)


# ==================== API 响应缓存 ====================

class APICache:
    """API 响应缓存

    使用 TTL 缓存存储 OpenAI API 响应，减少重复调用。
    """

    def __init__(
        self,
        maxsize: int = 1000,
        ttl: int = 300  # 5分钟
    ):
        """初始化缓存

        Args:
            maxsize: 最大缓存条目数
            ttl: 缓存过期时间（秒）
        """
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _generate_key(text: str, method: str) -> str:
        """生成缓存键"""
        content = f"{method}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, method: str) -> Optional[Dict]:
        """获取缓存的响应"""
        key = self._generate_key(text, method)
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._hits += 1
                logger.debug(f"缓存命中: {key[:8]}...")
                return result
            self._misses += 1
            return None

    def set(self, text: str, method: str, response: Dict) -> None:
        """设置缓存"""
        key = self._generate_key(text, method)
        with self._lock:
            # 复制响应，添加缓存标记
            cached_response = response.copy()
            cached_response["_cached"] = True
            cached_response["_cached_at"] = time.time()
            self._cache[key] = cached_response
            logger.debug(f"缓存设置: {key[:8]}...")

    def invalidate(self, text: str, method: str) -> bool:
        """使特定缓存失效"""
        key = self._generate_key(text, method)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """清空所有缓存"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> Dict:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "size": len(self._cache),
                "maxsize": self._cache.maxsize,
                "ttl": self._cache.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4)
            }


# ==================== 会话 TTL 缓存 ====================

@dataclass
class CachedSession:
    """缓存的会话数据"""
    session_id: str
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        """更新最后访问时间"""
        self.last_accessed = time.time()

    def is_expired(self, ttl: int) -> bool:
        """检查是否过期"""
        return (time.time() - self.last_accessed) > ttl


class SessionCache:
    """会话缓存，支持 TTL 自动清理

    使用后台线程定期清理过期会话。
    """

    def __init__(
        self,
        maxsize: int = 10000,
        ttl: int = 1800,  # 30分钟
        cleanup_interval: int = 60  # 1分钟清理一次
    ):
        """初始化会话缓存

        Args:
            maxsize: 最大会话数
            ttl: 会话过期时间（秒）
            cleanup_interval: 清理间隔（秒）
        """
        self._sessions: Dict[str, CachedSession] = {}
        self._lock = threading.Lock()
        self.maxsize = maxsize
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval

        # 启动后台清理线程
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """启动后台清理线程"""
        def cleanup_loop():
            while not self._stop_cleanup.wait(self.cleanup_interval):
                self._cleanup_expired()

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="SessionCacheCleanup"
        )
        self._cleanup_thread.start()
        logger.info(f"会话缓存清理线程已启动 (间隔: {self.cleanup_interval}秒)")

    def _cleanup_expired(self) -> int:
        """清理过期会话"""
        with self._lock:
            expired_keys = [
                key for key, session in self._sessions.items()
                if session.is_expired(self.ttl)
            ]

            for key in expired_keys:
                del self._sessions[key]

            if expired_keys:
                logger.debug(f"清理了 {len(expired_keys)} 个过期会话")

            return len(expired_keys)

    def get(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if session.is_expired(self.ttl):
                    del self._sessions[session_id]
                    return None
                session.touch()
                return session.data
            return None

    def set(self, session_id: str, data: Dict) -> bool:
        """设置会话"""
        with self._lock:
            # 检查容量
            if len(self._sessions) >= self.maxsize and session_id not in self._sessions:
                # LRU 淘汰：删除最久未访问的
                oldest = min(
                    self._sessions.items(),
                    key=lambda x: x[1].last_accessed
                )
                del self._sessions[oldest[0]]
                logger.debug(f"LRU 淘汰会话: {oldest[0]}")

            self._sessions[session_id] = CachedSession(
                session_id=session_id,
                data=data
            )
            return True

    def update(self, session_id: str, data: Dict) -> bool:
        """更新会话"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].data = data
                self._sessions[session_id].touch()
                return True
            return False

    def delete(self, session_id: str) -> bool:
        """删除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if session.is_expired(self.ttl):
                    del self._sessions[session_id]
                    return False
                return True
            return False

    def stats(self) -> Dict:
        """获取缓存统计"""
        with self._lock:
            active = sum(
                1 for s in self._sessions.values()
                if not s.is_expired(self.ttl)
            )
            return {
                "total": len(self._sessions),
                "active": active,
                "maxsize": self.maxsize,
                "ttl": self.ttl
            }

    def shutdown(self):
        """关闭清理线程"""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2)
        logger.info("会话缓存清理线程已停止")


# ==================== 向量检索缓存 ====================

class VectorCache:
    """向量检索结果缓存

    缓存语义检索结果，减少重复向量查询开销。
    """

    def __init__(
        self,
        maxsize: int = 500,
        ttl: int = 600  # 10分钟
    ):
        """初始化向量缓存

        Args:
            maxsize: 最大缓存条目数
            ttl: 缓存过期时间（秒）
        """
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _generate_key(query: str, top_k: int) -> str:
        """生成缓存键"""
        content = f"vector:{query}:{top_k}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, top_k: int = 3) -> Optional[list]:
        """获取缓存的检索结果"""
        key = self._generate_key(query, top_k)
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._hits += 1
                logger.debug(f"向量缓存命中: {query[:20]}...")
                return result
            self._misses += 1
            return None

    def set(self, query: str, top_k: int, results: list) -> None:
        """设置缓存"""
        key = self._generate_key(query, top_k)
        with self._lock:
            self._cache[key] = results
            logger.debug(f"向量缓存设置: {query[:20]}...")

    def invalidate_all(self) -> int:
        """清空所有缓存（新增示例时使用）"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            return count

    def stats(self) -> Dict:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "size": len(self._cache),
                "maxsize": self._cache.maxsize,
                "ttl": self._cache.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4)
            }


# ==================== 缓存装饰器 ====================

def cached_classify(cache: APICache, method: str):
    """分类结果缓存装饰器

    Usage:
        @cached_classify(api_cache, "zero_shot")
        def classify_zero_shot(self, text: str) -> Dict:
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, text: str, *args, **kwargs) -> Dict:
            # 尝试从缓存获取
            cached = cache.get(text, method)
            if cached is not None:
                return cached

            # 调用原函数
            result = func(self, text, *args, **kwargs)

            # 缓存结果（只缓存成功的结果）
            if result.get("intent") != "UNKNOWN":
                cache.set(text, method, result)

            return result
        return wrapper
    return decorator


def async_cached_classify(cache: APICache, method: str):
    """异步分类结果缓存装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, text: str, *args, **kwargs) -> Dict:
            # 尝试从缓存获取
            cached = cache.get(text, method)
            if cached is not None:
                return cached

            # 调用原函数
            result = await func(self, text, *args, **kwargs)

            # 缓存结果
            if result.get("intent") != "UNKNOWN":
                cache.set(text, method, result)

            return result
        return wrapper
    return decorator


# ==================== 全局缓存实例 ====================

# API 响应缓存（5分钟过期，最多1000条）
api_cache = APICache(maxsize=1000, ttl=300)

# 会话缓存（30分钟过期，最多10000个会话）
session_cache = SessionCache(maxsize=10000, ttl=1800, cleanup_interval=60)

# 向量检索缓存（10分钟过期，最多500条）
vector_cache = VectorCache(maxsize=500, ttl=600)


def get_api_cache() -> APICache:
    """获取 API 缓存实例"""
    return api_cache


def get_session_cache() -> SessionCache:
    """获取会话缓存实例"""
    return session_cache


def get_vector_cache() -> VectorCache:
    """获取向量检索缓存实例"""
    return vector_cache
