"""
统一重试管理器

提供统一的重试逻辑，支持同步和异步函数，可配置重试策略。
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Any, Optional, Type, Tuple
from functools import wraps
from dataclasses import dataclass

from exceptions import (
    APIError, RetryableError, FatalError,
    RateLimitError, NetworkError, ServiceError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ==================== 重试策略 ====================

class RetryPolicy(ABC):
    """重试策略抽象基类"""

    @abstractmethod
    def get_wait_time(self, attempt: int, error: Exception) -> float:
        """获取等待时间

        Args:
            attempt: 当前尝试次数 (从1开始)
            error: 引发的异常

        Returns:
            等待时间（秒）
        """
        pass

    @abstractmethod
    def should_retry(self, error: Exception, attempt: int, max_attempts: int) -> bool:
        """判断是否应该重试

        Args:
            error: 引发的异常
            attempt: 当前尝试次数
            max_attempts: 最大尝试次数

        Returns:
            是否应该重试
        """
        pass


class ExponentialBackoffPolicy(RetryPolicy):
    """指数退避策略

    等待时间 = min(max_wait, min_wait * base^(attempt-1))
    """

    def __init__(
        self,
        min_wait: float = 1.0,
        max_wait: float = 30.0,
        base: float = 2.0,
        jitter: bool = True
    ):
        """初始化

        Args:
            min_wait: 最小等待时间（秒）
            max_wait: 最大等待时间（秒）
            base: 指数基数
            jitter: 是否添加随机抖动
        """
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.base = base
        self.jitter = jitter

    def get_wait_time(self, attempt: int, error: Exception) -> float:
        wait = self.min_wait * (self.base ** (attempt - 1))

        # 如果是限流错误且有 retry_after，使用它
        if isinstance(error, RateLimitError) and error.retry_after:
            wait = max(wait, error.retry_after)

        wait = min(wait, self.max_wait)

        # 添加随机抖动（0-25%）
        if self.jitter:
            import random
            wait *= (1 + random.random() * 0.25)

        return wait

    def should_retry(self, error: Exception, attempt: int, max_attempts: int) -> bool:
        # 致命错误不重试
        if isinstance(error, FatalError):
            return False
        # 可重试错误继续重试
        if isinstance(error, RetryableError):
            return attempt < max_attempts
        # 其他错误（如网络错误）也重试
        return attempt < max_attempts


class FixedIntervalPolicy(RetryPolicy):
    """固定间隔策略"""

    def __init__(self, interval: float = 1.0):
        self.interval = interval

    def get_wait_time(self, attempt: int, error: Exception) -> float:
        return self.interval

    def should_retry(self, error: Exception, attempt: int, max_attempts: int) -> bool:
        if isinstance(error, FatalError):
            return False
        return attempt < max_attempts


class LinearBackoffPolicy(RetryPolicy):
    """线性退避策略

    等待时间 = min(max_wait, min_wait + increment * (attempt-1))
    """

    def __init__(self, min_wait: float = 1.0, increment: float = 1.0, max_wait: float = 10.0):
        self.min_wait = min_wait
        self.increment = increment
        self.max_wait = max_wait

    def get_wait_time(self, attempt: int, error: Exception) -> float:
        wait = self.min_wait + self.increment * (attempt - 1)
        return min(wait, self.max_wait)

    def should_retry(self, error: Exception, attempt: int, max_attempts: int) -> bool:
        if isinstance(error, FatalError):
            return False
        return attempt < max_attempts


# ==================== 重试管理器 ====================

@dataclass
class RetryStats:
    """重试统计"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_retries: int = 0
    total_wait_time: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_retries(self) -> float:
        if self.successful_calls == 0:
            return 0.0
        return self.total_retries / self.successful_calls


class RetryManager:
    """统一重试管理器

    支持同步和异步函数的重试，提供统一的重试逻辑。

    Usage:
        manager = RetryManager(
            policy=ExponentialBackoffPolicy(),
            max_attempts=3
        )

        # 同步调用
        result = manager.execute(call_api, arg1, arg2)

        # 异步调用
        result = await manager.execute_async(call_api_async, arg1, arg2)

        # 作为装饰器
        @manager.retry
        def my_function():
            ...

        @manager.retry_async
        async def my_async_function():
            ...
    """

    def __init__(
        self,
        policy: Optional[RetryPolicy] = None,
        max_attempts: int = 3,
        retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """初始化

        Args:
            policy: 重试策略，默认使用指数退避
            max_attempts: 最大尝试次数
            retry_exceptions: 可重试的异常类型，默认使用 RetryableError
        """
        self.policy = policy or ExponentialBackoffPolicy()
        self.max_attempts = max_attempts
        self.retry_exceptions = retry_exceptions or (RetryableError, Exception)
        self.stats = RetryStats()

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """同步执行带重试

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            最后一次失败的异常
        """
        self.stats.total_calls += 1
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                self.stats.successful_calls += 1
                return result
            except Exception as e:
                last_error = e

                if not self.policy.should_retry(e, attempt, self.max_attempts):
                    logger.error(f"不可重试的错误: {e}")
                    self.stats.failed_calls += 1
                    raise

                if attempt < self.max_attempts:
                    wait_time = self.policy.get_wait_time(attempt, e)
                    self.stats.total_retries += 1
                    self.stats.total_wait_time += wait_time

                    logger.warning(
                        f"尝试 {attempt}/{self.max_attempts} 失败，"
                        f"{wait_time:.1f}秒后重试: {type(e).__name__}: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"所有 {self.max_attempts} 次尝试都失败: {e}")
                    self.stats.failed_calls += 1

        raise last_error or RetryableError("所有重试都失败了")

    async def execute_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """异步执行带重试

        Args:
            func: 要执行的异步函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            最后一次失败的异常
        """
        self.stats.total_calls += 1
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)
                self.stats.successful_calls += 1
                return result
            except Exception as e:
                last_error = e

                if not self.policy.should_retry(e, attempt, self.max_attempts):
                    logger.error(f"不可重试的错误: {e}")
                    self.stats.failed_calls += 1
                    raise

                if attempt < self.max_attempts:
                    wait_time = self.policy.get_wait_time(attempt, e)
                    self.stats.total_retries += 1
                    self.stats.total_wait_time += wait_time

                    logger.warning(
                        f"尝试 {attempt}/{self.max_attempts} 失败，"
                        f"{wait_time:.1f}秒后重试: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"所有 {self.max_attempts} 次尝试都失败: {e}")
                    self.stats.failed_calls += 1

        raise last_error or RetryableError("所有重试都失败了")

    def retry(self, func: Callable[..., T]) -> Callable[..., T]:
        """同步重试装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.execute(func, *args, **kwargs)
        return wrapper

    def retry_async(self, func: Callable[..., T]) -> Callable[..., T]:
        """异步重试装饰器"""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.execute_async(func, *args, **kwargs)
        return wrapper

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "total_retries": self.stats.total_retries,
            "success_rate": f"{self.stats.success_rate*100:.1f}%",
            "avg_retries": f"{self.stats.avg_retries:.2f}",
            "total_wait_time": f"{self.stats.total_wait_time:.1f}s"
        }

    def reset_stats(self):
        """重置统计"""
        self.stats = RetryStats()


# ==================== 便捷函数 ====================

# 默认重试管理器（指数退避，最多3次）
_default_manager: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """获取默认重试管理器"""
    global _default_manager
    if _default_manager is None:
        _default_manager = RetryManager(
            policy=ExponentialBackoffPolicy(min_wait=1.0, max_wait=30.0),
            max_attempts=3
        )
    return _default_manager


def with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """使用默认重试策略的装饰器"""
    return get_retry_manager().retry(func)


def with_retry_async(func: Callable[..., T]) -> Callable[..., T]:
    """使用默认重试策略的异步装饰器"""
    return get_retry_manager().retry_async(func)


# ==================== OpenAI 专用重试管理器 ====================

def create_openai_retry_manager(max_attempts: int = 3) -> RetryManager:
    """创建 OpenAI API 专用重试管理器

    针对 OpenAI API 的特点优化：
    - 限流错误使用更长的等待时间
    - 网络错误快速重试
    - 认证错误不重试
    """
    return RetryManager(
        policy=ExponentialBackoffPolicy(
            min_wait=1.0,
            max_wait=60.0,  # OpenAI 限流可能需要更长等待
            base=2.0,
            jitter=True
        ),
        max_attempts=max_attempts
    )
