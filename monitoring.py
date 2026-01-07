"""
监控和结构化日志模块

提供结构化日志、性能监控和请求追踪功能。
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
from collections import deque

# ==================== 结构化日志 ====================

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器

    输出 JSON 格式的日志，便于日志聚合和分析。
    """

    def __init__(self, service_name: str = "ai-order-system"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        # 添加位置信息
        if record.pathname:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName
            }

        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_data["data"] = record.extra_data

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }

        # 添加请求 ID（如果有）
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id

        return json.dumps(log_data, ensure_ascii=False)


class SensitiveDataFilter(logging.Filter):
    """敏感数据过滤器

    自动屏蔽日志中的敏感信息。
    """

    SENSITIVE_PATTERNS = [
        'api_key', 'apikey', 'api-key',
        'password', 'passwd', 'pwd',
        'token', 'secret', 'credential',
        'authorization', 'auth'
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        # 过滤消息中的敏感信息
        msg = record.getMessage().lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in msg:
                record.msg = self._mask_sensitive(record.msg)
                break
        return True

    def _mask_sensitive(self, text: str) -> str:
        """屏蔽敏感值"""
        import re
        # 匹配 key=value 或 key: value 格式
        patterns = [
            (r'(api[_-]?key\s*[=:]\s*)["\']?([^"\'\s,}]+)["\']?', r'\1****'),
            (r'(password\s*[=:]\s*)["\']?([^"\'\s,}]+)["\']?', r'\1****'),
            (r'(token\s*[=:]\s*)["\']?([^"\'\s,}]+)["\']?', r'\1****'),
            (r'(secret\s*[=:]\s*)["\']?([^"\'\s,}]+)["\']?', r'\1****'),
            (r'(sk-[a-zA-Z0-9]+)', '****'),
        ]
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result


class StructuredLogger:
    """结构化日志记录器

    提供便捷的结构化日志方法。
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._request_id: Optional[str] = None

    def set_request_id(self, request_id: str):
        """设置请求 ID"""
        self._request_id = request_id

    def _log(self, level: int, message: str, **kwargs):
        """内部日志方法"""
        extra = {'extra_data': kwargs}
        if self._request_id:
            extra['request_id'] = self._request_id
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)

    def log_request(self, method: str, path: str, user_id: Optional[str] = None):
        """记录请求开始"""
        self.info("request_started", method=method, path=path, user_id=user_id)

    def log_response(self, status_code: int, duration_ms: float):
        """记录请求结束"""
        self.info("request_completed", status_code=status_code, duration_ms=round(duration_ms, 2))

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """记录错误"""
        self.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )

    def log_classification(
        self,
        text: str,
        intent: str,
        confidence: float,
        method: str,
        duration_ms: float
    ):
        """记录分类结果"""
        self.info(
            "classification_completed",
            text_length=len(text),
            intent=intent,
            confidence=round(confidence, 4),
            method=method,
            duration_ms=round(duration_ms, 2)
        )


# ==================== 性能监控 ====================

@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """指标收集器

    收集和聚合应用程序指标。
    """

    def __init__(self, max_points: int = 10000):
        self._metrics: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._max_points = max_points

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录指标"""
        point = MetricPoint(name=name, value=value, labels=labels or {})

        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self._max_points)
            self._metrics[name].append(point)

    def record_duration(self, name: str, duration_seconds: float, labels: Optional[Dict[str, str]] = None):
        """记录持续时间（毫秒）"""
        self.record(name, duration_seconds * 1000, labels)

    def get_stats(self, name: str, window_seconds: int = 300) -> Dict[str, Any]:
        """获取指标统计"""
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            if name not in self._metrics:
                return {}

            points = [p for p in self._metrics[name] if p.timestamp >= cutoff]

        if not points:
            return {}

        values = [p.value for p in points]
        sorted_values = sorted(values)
        count = len(values)

        return {
            "name": name,
            "count": count,
            "sum": sum(values),
            "avg": sum(values) / count,
            "min": min(values),
            "max": max(values),
            "p50": sorted_values[int(count * 0.5)] if count > 0 else 0,
            "p95": sorted_values[int(count * 0.95)] if count > 0 else 0,
            "p99": sorted_values[int(count * 0.99)] if count > 0 else 0,
        }

    def get_all_stats(self, window_seconds: int = 300) -> Dict[str, Dict[str, Any]]:
        """获取所有指标统计"""
        with self._lock:
            names = list(self._metrics.keys())

        return {name: self.get_stats(name, window_seconds) for name in names}

    def clear(self, name: Optional[str] = None):
        """清除指标"""
        with self._lock:
            if name:
                self._metrics.pop(name, None)
            else:
                self._metrics.clear()


# ==================== 请求追踪 ====================

_request_context = threading.local()


def get_request_id() -> Optional[str]:
    """获取当前请求 ID"""
    return getattr(_request_context, 'request_id', None)


def set_request_id(request_id: str):
    """设置当前请求 ID"""
    _request_context.request_id = request_id


def generate_request_id() -> str:
    """生成请求 ID"""
    return str(uuid.uuid4())


@contextmanager
def request_context(request_id: Optional[str] = None):
    """请求上下文管理器"""
    rid = request_id or generate_request_id()
    old_id = get_request_id()
    set_request_id(rid)
    try:
        yield rid
    finally:
        if old_id:
            set_request_id(old_id)
        else:
            _request_context.request_id = None


# ==================== 性能装饰器 ====================

def monitor_performance(
    name: Optional[str] = None,
    collector: Optional[MetricsCollector] = None
):
    """性能监控装饰器

    Usage:
        @monitor_performance("api.classify")
        def classify(text):
            ...

        @monitor_performance()  # 使用函数名作为指标名
        async def async_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or f"{func.__module__}.{func.__name__}"
        metrics = collector or get_metrics_collector()

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    metrics.record(f"{metric_name}.success", 1)
                    return result
                except Exception as e:
                    metrics.record(f"{metric_name}.error", 1, {"error": type(e).__name__})
                    raise
                finally:
                    duration = time.time() - start
                    metrics.record_duration(f"{metric_name}.duration", duration)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    metrics.record(f"{metric_name}.success", 1)
                    return result
                except Exception as e:
                    metrics.record(f"{metric_name}.error", 1, {"error": type(e).__name__})
                    raise
                finally:
                    duration = time.time() - start
                    metrics.record_duration(f"{metric_name}.duration", duration)
            return sync_wrapper
    return decorator


# ==================== FastAPI 中间件 ====================

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
    import asyncio

    class MonitoringMiddleware(BaseHTTPMiddleware):
        """监控中间件

        自动添加请求追踪和性能监控。
        """

        def __init__(self, app, logger: Optional[StructuredLogger] = None):
            super().__init__(app)
            self.logger = logger or StructuredLogger("http")
            self.metrics = get_metrics_collector()

        async def dispatch(self, request: Request, call_next) -> Response:
            # 生成请求 ID
            request_id = request.headers.get("X-Request-ID") or generate_request_id()
            set_request_id(request_id)
            self.logger.set_request_id(request_id)

            # 记录请求开始
            self.logger.log_request(
                method=request.method,
                path=request.url.path
            )

            start = time.time()
            try:
                response = await call_next(request)

                # 记录请求结束
                duration = (time.time() - start) * 1000
                self.logger.log_response(response.status_code, duration)

                # 记录指标
                self.metrics.record_duration("http.request.duration", duration / 1000, {
                    "method": request.method,
                    "path": request.url.path,
                    "status": str(response.status_code)
                })

                # 添加响应头
                response.headers["X-Request-ID"] = request_id
                return response

            except Exception as e:
                duration = (time.time() - start) * 1000
                self.logger.log_error(e, {"path": request.url.path})
                self.metrics.record(f"http.request.error", 1, {"error": type(e).__name__})
                raise

except ImportError:
    # Starlette 未安装
    pass


# ==================== 全局实例 ====================

_metrics_collector: Optional[MetricsCollector] = None
_structured_logger: Optional[StructuredLogger] = None

import asyncio  # 确保导入


def get_metrics_collector() -> MetricsCollector:
    """获取指标收集器实例"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_structured_logger(name: str = "app") -> StructuredLogger:
    """获取结构化日志记录器"""
    return StructuredLogger(name)


def setup_logging(
    level: int = logging.INFO,
    structured: bool = True,
    service_name: str = "ai-order-system"
):
    """配置日志系统

    Args:
        level: 日志级别
        structured: 是否使用结构化日志格式
        service_name: 服务名称
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有处理器
    root_logger.handlers.clear()

    # 创建处理器
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if structured:
        handler.setFormatter(StructuredFormatter(service_name))
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    # 添加敏感数据过滤器
    handler.addFilter(SensitiveDataFilter())

    root_logger.addHandler(handler)

    logging.info(f"日志系统已配置: level={logging.getLevelName(level)}, structured={structured}")
