"""
弹性机制模块

提供限流（Rate Limiting）和熔断器（Circuit Breaker）功能。
"""

import time
import asyncio
import logging
import threading
from enum import Enum
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== 令牌桶限流 ====================

class TokenBucket:
    """令牌桶限流器

    实现令牌桶算法，支持突发流量。

    Usage:
        limiter = TokenBucket(rate=10.0, capacity=20)

        if limiter.acquire():
            # 处理请求
        else:
            # 拒绝请求
    """

    def __init__(self, rate: float, capacity: int):
        """初始化

        Args:
            rate: 每秒生成的令牌数
            capacity: 桶容量（最大令牌数）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = threading.Lock()

        # 统计
        self.total_requests = 0
        self.accepted_requests = 0
        self.rejected_requests = 0

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """获取令牌

        Args:
            tokens: 需要的令牌数

        Returns:
            是否成功获取
        """
        with self._lock:
            self._refill()
            self.total_requests += 1

            if self.tokens >= tokens:
                self.tokens -= tokens
                self.accepted_requests += 1
                return True
            else:
                self.rejected_requests += 1
                return False

    def wait_time(self, tokens: int = 1) -> float:
        """计算等待时间

        Returns:
            需要等待的秒数，0 表示立即可用
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / self.rate

    async def acquire_async(self, tokens: int = 1, timeout: float = 0.0) -> bool:
        """异步获取令牌

        Args:
            tokens: 需要的令牌数
            timeout: 最大等待时间，0 表示不等待

        Returns:
            是否成功获取
        """
        wait = self.wait_time(tokens)
        if wait <= 0:
            return self.acquire(tokens)

        if timeout <= 0 or wait > timeout:
            self.rejected_requests += 1
            return False

        await asyncio.sleep(wait)
        return self.acquire(tokens)

    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            self._refill()
            return {
                "rate": self.rate,
                "capacity": self.capacity,
                "current_tokens": round(self.tokens, 2),
                "total_requests": self.total_requests,
                "accepted_requests": self.accepted_requests,
                "rejected_requests": self.rejected_requests,
                "acceptance_rate": (
                    self.accepted_requests / self.total_requests
                    if self.total_requests > 0 else 1.0
                )
            }

    def reset(self):
        """重置限流器"""
        with self._lock:
            self.tokens = self.capacity
            self.last_update = time.time()
            self.total_requests = 0
            self.accepted_requests = 0
            self.rejected_requests = 0


# ==================== 熔断器 ====================

class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常，允许请求
    OPEN = "open"          # 熔断，拒绝所有请求
    HALF_OPEN = "half_open"  # 半开，允许部分请求测试


@dataclass
class CircuitStats:
    """熔断器统计"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)


class CircuitBreaker:
    """熔断器

    实现熔断器模式，防止级联故障。

    状态转换:
    - CLOSED -> OPEN: 连续失败达到阈值
    - OPEN -> HALF_OPEN: 超时后自动转换
    - HALF_OPEN -> CLOSED: 连续成功达到阈值
    - HALF_OPEN -> OPEN: 任何失败

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, timeout=30.0)

        @breaker
        def call_external_service():
            ...

        # 或手动使用
        if breaker.allow_request():
            try:
                result = call_external_service()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 30.0,
        name: str = "default"
    ):
        """初始化

        Args:
            failure_threshold: 触发熔断的连续失败次数
            success_threshold: 恢复正常的连续成功次数
            timeout: 熔断状态持续时间（秒）
            name: 熔断器名称（用于日志）
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.name = name

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """当前状态"""
        with self._lock:
            self._check_state_timeout()
            return self._state

    def _check_state_timeout(self):
        """检查是否需要从 OPEN 转换到 HALF_OPEN"""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._stats.last_state_change
            if elapsed >= self.timeout:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState):
        """状态转换"""
        old_state = self._state
        self._state = new_state
        self._stats.last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._stats.consecutive_successes = 0

        logger.info(f"熔断器 [{self.name}] 状态变更: {old_state.value} -> {new_state.value}")

    def allow_request(self) -> bool:
        """检查是否允许请求"""
        with self._lock:
            self._check_state_timeout()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                return True

    def record_success(self):
        """记录成功"""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self):
        """记录失败"""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            self._check_state_timeout()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "timeout": self.timeout,
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "success_rate": (
                    self._stats.successful_calls / self._stats.total_calls
                    if self._stats.total_calls > 0 else 1.0
                )
            }

    def reset(self):
        """重置熔断器"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitStats()
            logger.info(f"熔断器 [{self.name}] 已重置")

    def __call__(self, func: Callable) -> Callable:
        """装饰器模式"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                raise CircuitOpenError(f"熔断器 [{self.name}] 处于开启状态")

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self.allow_request():
                raise CircuitOpenError(f"熔断器 [{self.name}] 处于开启状态")

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper


class CircuitOpenError(Exception):
    """熔断器开启异常"""
    pass


# ==================== 限流中间件 ====================

class RateLimiterManager:
    """限流管理器

    支持全局限流和按 IP 限流。
    """

    def __init__(
        self,
        rate: float = 10.0,
        capacity: int = 20,
        per_ip: bool = True
    ):
        """初始化

        Args:
            rate: 每秒请求数
            capacity: 突发容量
            per_ip: 是否按 IP 限流
        """
        self.rate = rate
        self.capacity = capacity
        self.per_ip = per_ip

        # 全局限流器
        self.global_limiter = TokenBucket(rate=rate * 10, capacity=capacity * 10)

        # 按 IP 限流器
        self._ip_limiters: Dict[str, TokenBucket] = {}
        self._ip_lock = threading.Lock()

        # 清理间隔（秒）
        self._cleanup_interval = 300
        self._last_cleanup = time.time()

    def _get_ip_limiter(self, ip: str) -> TokenBucket:
        """获取 IP 限流器"""
        with self._ip_lock:
            # 定期清理不活跃的限流器
            now = time.time()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_inactive()
                self._last_cleanup = now

            if ip not in self._ip_limiters:
                self._ip_limiters[ip] = TokenBucket(
                    rate=self.rate,
                    capacity=self.capacity
                )
            return self._ip_limiters[ip]

    def _cleanup_inactive(self):
        """清理不活跃的限流器"""
        now = time.time()
        inactive = [
            ip for ip, limiter in self._ip_limiters.items()
            if now - limiter.last_update > self._cleanup_interval
        ]
        for ip in inactive:
            del self._ip_limiters[ip]

        if inactive:
            logger.debug(f"清理了 {len(inactive)} 个不活跃的 IP 限流器")

    def allow(self, ip: Optional[str] = None) -> bool:
        """检查是否允许请求"""
        # 全局限流
        if not self.global_limiter.acquire():
            return False

        # 按 IP 限流
        if self.per_ip and ip:
            limiter = self._get_ip_limiter(ip)
            return limiter.acquire()

        return True

    async def allow_async(self, ip: Optional[str] = None, timeout: float = 1.0) -> bool:
        """异步检查是否允许请求"""
        # 全局限流
        if not await self.global_limiter.acquire_async(timeout=timeout):
            return False

        # 按 IP 限流
        if self.per_ip and ip:
            limiter = self._get_ip_limiter(ip)
            return await limiter.acquire_async(timeout=timeout)

        return True

    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._ip_lock:
            return {
                "global": self.global_limiter.stats(),
                "per_ip_count": len(self._ip_limiters),
                "per_ip_enabled": self.per_ip
            }


# ==================== FastAPI 中间件 ====================

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response, JSONResponse

    class RateLimitMiddleware(BaseHTTPMiddleware):
        """限流中间件"""

        def __init__(self, app, manager: RateLimiterManager):
            super().__init__(app)
            self.manager = manager

        async def dispatch(self, request: Request, call_next) -> Response:
            # 获取客户端 IP
            ip = request.client.host if request.client else None

            # 检查限流
            if not await self.manager.allow_async(ip):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "TooManyRequests",
                        "message": "请求过于频繁，请稍后重试"
                    },
                    headers={"Retry-After": "1"}
                )

            return await call_next(request)

except ImportError:
    pass


# ==================== 全局实例 ====================

_rate_limiter: Optional[RateLimiterManager] = None
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_rate_limiter(
    rate: float = 10.0,
    capacity: int = 20,
    per_ip: bool = True
) -> RateLimiterManager:
    """获取限流管理器"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiterManager(
            rate=rate,
            capacity=capacity,
            per_ip=per_ip
        )
    return _rate_limiter


def get_circuit_breaker(
    name: str = "default",
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout: float = 30.0
) -> CircuitBreaker:
    """获取熔断器"""
    global _circuit_breakers
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout
        )
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """获取所有熔断器状态"""
    return {
        name: breaker.stats()
        for name, breaker in _circuit_breakers.items()
    }
