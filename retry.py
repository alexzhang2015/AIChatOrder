"""
重试策略模块

提供基于 tenacity 的重试装饰器，支持指数退避和错误分类。
"""

import logging
import functools
from typing import Callable, Type, Tuple, Optional, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    RetryError
)

from exceptions import (
    RetryableError,
    FatalError,
    RateLimitError,
    NetworkError,
    ServiceError,
    TimeoutError,
    classify_openai_error
)

logger = logging.getLogger(__name__)


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,)
):
    """创建自定义重试装饰器

    Args:
        max_attempts: 最大重试次数
        min_wait: 最小等待时间（秒）
        max_wait: 最大等待时间（秒）
        exponential_base: 指数退避的基数
        retryable_exceptions: 需要重试的异常类型元组

    Returns:
        配置好的重试装饰器
    """
    return retry(
        retry=retry_if_exception_type(retryable_exceptions),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=min_wait,
            min=min_wait,
            max=max_wait,
            exp_base=exponential_base
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.DEBUG),
        reraise=True
    )


# 预配置的装饰器

# 标准重试：3次，1-30秒指数退避
retry_standard = create_retry_decorator(
    max_attempts=3,
    min_wait=1.0,
    max_wait=30.0
)

# 快速重试：2次，0.5-5秒
retry_fast = create_retry_decorator(
    max_attempts=2,
    min_wait=0.5,
    max_wait=5.0
)

# 持久重试：5次，2-60秒
retry_persistent = create_retry_decorator(
    max_attempts=5,
    min_wait=2.0,
    max_wait=60.0
)


def with_openai_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    fallback: Optional[Callable[..., Any]] = None
):
    """OpenAI API 专用重试装饰器

    自动将 OpenAI 异常转换为自定义异常，并根据异常类型决定是否重试。

    Args:
        max_attempts: 最大重试次数
        min_wait: 最小等待时间（秒）
        max_wait: 最大等待时间（秒）
        on_retry: 重试时的回调函数 (exception, attempt) -> None
        fallback: 所有重试失败后的降级函数

    Usage:
        @with_openai_retry(max_attempts=3)
        def call_openai_api():
            return client.chat.completions.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except FatalError:
                    # 不可重试错误，直接抛出
                    raise
                except RetryableError as e:
                    last_error = e
                    if attempt < max_attempts:
                        # 计算等待时间
                        wait_time = min(
                            min_wait * (2 ** (attempt - 1)),
                            max_wait
                        )

                        # 如果是速率限制错误，使用建议的等待时间
                        if isinstance(e, RateLimitError) and e.retry_after:
                            wait_time = min(e.retry_after, max_wait)

                        logger.warning(
                            f"API调用失败 (尝试 {attempt}/{max_attempts}): {e}. "
                            f"将在 {wait_time:.1f}秒后重试"
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        import time
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"API调用最终失败 ({max_attempts}次尝试后): {e}"
                        )
                except Exception as e:
                    # 未知异常，尝试分类
                    classified_error = classify_openai_error(e)

                    if isinstance(classified_error, FatalError):
                        raise classified_error

                    last_error = classified_error
                    if attempt < max_attempts:
                        wait_time = min(
                            min_wait * (2 ** (attempt - 1)),
                            max_wait
                        )
                        logger.warning(
                            f"API调用失败 (尝试 {attempt}/{max_attempts}): {e}. "
                            f"将在 {wait_time:.1f}秒后重试"
                        )

                        if on_retry:
                            on_retry(classified_error, attempt)

                        import time
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"API调用最终失败 ({max_attempts}次尝试后): {e}"
                        )

            # 所有重试都失败了
            if fallback:
                logger.info("使用降级方案")
                return fallback(*args, **kwargs)

            if last_error:
                raise last_error
            raise RetryableError("所有重试都失败了")

        return wrapper
    return decorator


def with_fallback(
    fallback_func: Callable[..., Any],
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_error: bool = True
):
    """降级装饰器

    当主函数失败时，自动调用降级函数。

    Args:
        fallback_func: 降级函数
        exceptions: 触发降级的异常类型
        log_error: 是否记录错误日志

    Usage:
        def fallback_classify(text):
            return {"intent": "UNKNOWN", "confidence": 0.5}

        @with_fallback(fallback_classify)
        def classify_with_llm(text):
            return client.chat.completions.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error:
                    logger.warning(
                        f"{func.__name__} 失败: {e}. 使用降级方案"
                    )
                return fallback_func(*args, **kwargs)
        return wrapper
    return decorator


class RetryContext:
    """重试上下文管理器

    提供更细粒度的重试控制。

    Usage:
        with RetryContext(max_attempts=3) as ctx:
            for attempt in ctx:
                try:
                    result = call_api()
                    break
                except RetryableError as e:
                    ctx.record_error(e)
                    if not ctx.should_retry():
                        raise
    """

    def __init__(
        self,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 30.0
    ):
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.current_attempt = 0
        self.errors: list = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        while self.current_attempt < self.max_attempts:
            self.current_attempt += 1
            yield self.current_attempt

    def record_error(self, error: Exception):
        """记录错误"""
        self.errors.append(error)

    def should_retry(self) -> bool:
        """判断是否应该重试"""
        if self.current_attempt >= self.max_attempts:
            return False

        if not self.errors:
            return True

        last_error = self.errors[-1]

        # 不可重试错误
        if isinstance(last_error, FatalError):
            return False

        return True

    def get_wait_time(self) -> float:
        """获取下次重试的等待时间"""
        wait_time = self.min_wait * (2 ** (self.current_attempt - 1))

        # 如果最后一个错误是速率限制，使用建议的等待时间
        if self.errors and isinstance(self.errors[-1], RateLimitError):
            suggested = self.errors[-1].retry_after
            if suggested:
                wait_time = max(wait_time, suggested)

        return min(wait_time, self.max_wait)

    def wait(self):
        """等待后再重试"""
        import time
        wait_time = self.get_wait_time()
        logger.debug(f"等待 {wait_time:.1f} 秒后重试")
        time.sleep(wait_time)
