"""
重试管理器测试
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, '/Users/sawzhang/code/AIChatOrder')

from retry_manager import (
    RetryManager,
    ExponentialBackoffPolicy,
    FixedIntervalPolicy,
    LinearBackoffPolicy,
    get_retry_manager
)
from exceptions import RetryableError, FatalError, RateLimitError


class TestExponentialBackoffPolicy:
    """指数退避策略测试"""

    def test_wait_time_increases(self):
        """测试等待时间指数增长"""
        policy = ExponentialBackoffPolicy(min_wait=1.0, base=2.0, jitter=False)

        wait1 = policy.get_wait_time(1, Exception())
        wait2 = policy.get_wait_time(2, Exception())
        wait3 = policy.get_wait_time(3, Exception())

        assert wait1 == 1.0
        assert wait2 == 2.0
        assert wait3 == 4.0

    def test_max_wait_cap(self):
        """测试最大等待时间限制"""
        policy = ExponentialBackoffPolicy(min_wait=1.0, max_wait=5.0, jitter=False)

        wait = policy.get_wait_time(10, Exception())

        assert wait == 5.0

    def test_rate_limit_retry_after(self):
        """测试限流错误的 retry_after"""
        policy = ExponentialBackoffPolicy(min_wait=1.0, jitter=False)
        error = RateLimitError("Rate limited", retry_after=10.0)

        wait = policy.get_wait_time(1, error)

        assert wait >= 10.0

    def test_should_retry_fatal_error(self):
        """测试致命错误不重试"""
        policy = ExponentialBackoffPolicy()

        should = policy.should_retry(FatalError("Fatal"), 1, 3)

        assert should is False

    def test_should_retry_retryable_error(self):
        """测试可重试错误"""
        policy = ExponentialBackoffPolicy()

        should = policy.should_retry(RetryableError("Retry"), 1, 3)

        assert should is True


class TestRetryManager:
    """重试管理器测试"""

    @pytest.fixture
    def manager(self):
        """创建测试用重试管理器"""
        return RetryManager(
            policy=ExponentialBackoffPolicy(min_wait=0.01, jitter=False),
            max_attempts=3
        )

    def test_execute_success(self, manager):
        """测试成功执行"""
        mock_func = Mock(return_value="success")

        result = manager.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_execute_retry_then_success(self, manager):
        """测试重试后成功"""
        mock_func = Mock(side_effect=[
            RetryableError("First fail"),
            "success"
        ])

        result = manager.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_all_retries_fail(self, manager):
        """测试所有重试都失败"""
        mock_func = Mock(side_effect=RetryableError("Always fail"))

        with pytest.raises(RetryableError):
            manager.execute(mock_func)

        assert mock_func.call_count == 3

    def test_execute_fatal_error_no_retry(self, manager):
        """测试致命错误不重试"""
        mock_func = Mock(side_effect=FatalError("Fatal"))

        with pytest.raises(FatalError):
            manager.execute(mock_func)

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_async_success(self, manager):
        """测试异步成功执行"""
        async def async_func():
            return "async success"

        result = await manager.execute_async(async_func)

        assert result == "async success"

    @pytest.mark.asyncio
    async def test_execute_async_retry(self, manager):
        """测试异步重试"""
        call_count = 0

        async def async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Fail first")
            return "success"

        result = await manager.execute_async(async_func)

        assert result == "success"
        assert call_count == 2

    def test_decorator(self, manager):
        """测试装饰器"""
        call_count = 0

        @manager.retry
        def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Fail first")
            return "decorated"

        result = my_func()

        assert result == "decorated"
        assert call_count == 2

    def test_stats(self, manager):
        """测试统计信息"""
        mock_func = Mock(return_value="ok")
        manager.execute(mock_func)
        manager.execute(mock_func)

        stats = manager.get_stats()

        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 2
        assert "100.0%" in stats["success_rate"]


class TestFixedIntervalPolicy:
    """固定间隔策略测试"""

    def test_constant_wait_time(self):
        """测试固定等待时间"""
        policy = FixedIntervalPolicy(interval=5.0)

        wait1 = policy.get_wait_time(1, Exception())
        wait2 = policy.get_wait_time(5, Exception())

        assert wait1 == 5.0
        assert wait2 == 5.0


class TestLinearBackoffPolicy:
    """线性退避策略测试"""

    def test_linear_increase(self):
        """测试线性增长"""
        policy = LinearBackoffPolicy(min_wait=1.0, increment=2.0, max_wait=10.0)

        wait1 = policy.get_wait_time(1, Exception())
        wait2 = policy.get_wait_time(2, Exception())
        wait3 = policy.get_wait_time(3, Exception())

        assert wait1 == 1.0
        assert wait2 == 3.0
        assert wait3 == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
