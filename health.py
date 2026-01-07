"""
健康检查模块

提供系统健康检查端点，检查数据库、OpenAI API、缓存等服务状态。
"""

import os
import time
import asyncio
import logging
from enum import Enum
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    status: HealthStatus
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class HealthReport:
    """健康检查报告"""
    status: HealthStatus
    checks: List[CheckResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "checks": {
                check.name: {
                    "status": check.status.value,
                    "latency_ms": round(check.latency_ms, 2),
                    "details": check.details,
                    **({"error": check.error} if check.error else {})
                }
                for check in self.checks
            }
        }


class HealthChecker:
    """健康检查器

    支持注册多个健康检查，异步执行并汇总结果。
    """

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._timeout = 5.0  # 单项检查超时时间

    def register(self, name: str, check_func: Callable):
        """注册健康检查

        Args:
            name: 检查项名称
            check_func: 检查函数，可以是同步或异步函数
        """
        self._checks[name] = check_func
        logger.debug(f"注册健康检查: {name}")

    def unregister(self, name: str):
        """取消注册"""
        self._checks.pop(name, None)

    async def _run_check(self, name: str, check_func: Callable) -> CheckResult:
        """执行单项检查"""
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(
                    check_func(),
                    timeout=self._timeout
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, check_func
                )

            latency = (time.time() - start) * 1000
            return CheckResult(
                name=name,
                status=HealthStatus.HEALTHY,
                latency_ms=latency,
                details=result if isinstance(result, dict) else {}
            )
        except asyncio.TimeoutError:
            latency = (time.time() - start) * 1000
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                error=f"检查超时 ({self._timeout}s)"
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"健康检查失败 [{name}]: {e}")
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                error=str(e)
            )

    async def check_all(self) -> HealthReport:
        """执行所有健康检查"""
        if not self._checks:
            return HealthReport(
                status=HealthStatus.HEALTHY,
                checks=[]
            )

        # 并发执行所有检查
        tasks = [
            self._run_check(name, func)
            for name, func in self._checks.items()
        ]
        results = await asyncio.gather(*tasks)

        # 计算总体状态
        statuses = [r.status for r in results]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return HealthReport(
            status=overall,
            checks=list(results)
        )

    async def check_one(self, name: str) -> Optional[CheckResult]:
        """执行单项健康检查"""
        if name not in self._checks:
            return None
        return await self._run_check(name, self._checks[name])


# ==================== 内置健康检查 ====================

async def check_database() -> Dict[str, Any]:
    """检查数据库连接"""
    from database import Database

    db = Database()
    start = time.time()

    with db.get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM orders")
        order_count = cursor.fetchone()[0]

    latency = (time.time() - start) * 1000

    return {
        "sessions": session_count,
        "orders": order_count,
        "query_latency_ms": round(latency, 2)
    }


async def check_openai() -> Dict[str, Any]:
    """检查 OpenAI API 连接"""
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置")

    base_url = os.getenv("OPENAI_BASE_URL")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=5.0
    )

    start = time.time()
    models = await client.models.list()
    latency = (time.time() - start) * 1000

    return {
        "available_models": len(models.data),
        "api_latency_ms": round(latency, 2),
        "base_url": base_url or "https://api.openai.com"
    }


def check_cache() -> Dict[str, Any]:
    """检查缓存状态"""
    from cache import get_api_cache, get_session_cache

    api_cache = get_api_cache()
    session_cache = get_session_cache()

    api_stats = api_cache.stats()
    session_stats = session_cache.stats()

    return {
        "api_cache": {
            "size": api_stats["size"],
            "hit_rate": f"{api_stats['hit_rate']*100:.1f}%"
        },
        "session_cache": {
            "active": session_stats["active"],
            "total": session_stats["total"]
        }
    }


def check_vector_store() -> Dict[str, Any]:
    """检查向量存储状态"""
    from vector_store import get_retriever

    retriever = get_retriever()

    return {
        "type": type(retriever).__name__,
        "example_count": len(retriever.examples) if hasattr(retriever, 'examples') else 0
    }


def check_intent_registry() -> Dict[str, Any]:
    """检查意图注册中心"""
    from intent_registry import get_intent_registry

    registry = get_intent_registry()
    stats = registry.stats()

    return {
        "version": stats["version"],
        "intent_count": stats["intent_count"],
        "rule_count": stats["rule_count"]
    }


# ==================== 全局实例 ====================

_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """获取健康检查器实例（单例）"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        # 注册内置检查
        _health_checker.register("database", check_database)
        _health_checker.register("openai", check_openai)
        _health_checker.register("cache", check_cache)
        _health_checker.register("vector_store", check_vector_store)
        _health_checker.register("intent_registry", check_intent_registry)
        logger.info("健康检查器已初始化，注册了 5 项检查")
    return _health_checker


def reset_health_checker():
    """重置健康检查器（用于测试）"""
    global _health_checker
    _health_checker = None
