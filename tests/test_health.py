"""
健康检查模块测试
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, '/Users/sawzhang/code/AIChatOrder')

from health import (
    HealthChecker, HealthStatus, CheckResult, HealthReport,
    get_health_checker, reset_health_checker
)


class TestHealthChecker:
    """HealthChecker 测试"""

    @pytest.fixture
    def checker(self):
        """创建测试用健康检查器"""
        reset_health_checker()
        return HealthChecker()

    def test_register_check(self, checker):
        """测试注册健康检查"""
        def my_check():
            return {"status": "ok"}

        checker.register("test", my_check)
        assert "test" in checker._checks

    def test_unregister_check(self, checker):
        """测试取消注册"""
        checker.register("test", lambda: {})
        checker.unregister("test")
        assert "test" not in checker._checks

    @pytest.mark.asyncio
    async def test_check_all_success(self, checker):
        """测试所有检查成功"""
        checker.register("check1", lambda: {"value": 1})
        checker.register("check2", lambda: {"value": 2})

        report = await checker.check_all()

        assert report.status == HealthStatus.HEALTHY
        assert len(report.checks) == 2

    @pytest.mark.asyncio
    async def test_check_all_with_failure(self, checker):
        """测试包含失败的检查"""
        checker.register("ok_check", lambda: {"status": "ok"})
        checker.register("bad_check", lambda: (_ for _ in ()).throw(RuntimeError("Test error")))

        report = await checker.check_all()

        assert report.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_one(self, checker):
        """测试单项检查"""
        checker.register("test", lambda: {"value": 42})

        result = await checker.check_one("test")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY
        assert result.details["value"] == 42

    @pytest.mark.asyncio
    async def test_check_one_not_found(self, checker):
        """测试检查不存在的项"""
        result = await checker.check_one("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_async_check_function(self, checker):
        """测试异步检查函数"""
        async def async_check():
            await asyncio.sleep(0.01)
            return {"async": True}

        checker.register("async_test", async_check)

        result = await checker.check_one("async_test")

        assert result.status == HealthStatus.HEALTHY
        assert result.details["async"] is True


class TestHealthReport:
    """HealthReport 测试"""

    def test_to_dict(self):
        """测试转换为字典"""
        checks = [
            CheckResult(name="db", status=HealthStatus.HEALTHY, latency_ms=10.5),
            CheckResult(name="api", status=HealthStatus.UNHEALTHY, latency_ms=100.0, error="Timeout")
        ]
        report = HealthReport(status=HealthStatus.UNHEALTHY, checks=checks)

        result = report.to_dict()

        assert result["status"] == "unhealthy"
        assert "db" in result["checks"]
        assert "api" in result["checks"]
        assert result["checks"]["api"]["error"] == "Timeout"


class TestGetHealthChecker:
    """全局健康检查器测试"""

    def test_singleton(self):
        """测试单例模式"""
        reset_health_checker()
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        assert checker1 is checker2

    def test_has_default_checks(self):
        """测试默认检查项"""
        reset_health_checker()
        checker = get_health_checker()

        # 应该有内置检查
        assert len(checker._checks) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
