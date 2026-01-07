"""
统一异常定义模块

提供分层的异常体系，区分可重试和不可重试的错误类型。
"""

from typing import Optional, Dict, Any


class APIError(Exception):
    """API相关异常基类"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于API响应"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details
        }


class RetryableError(APIError):
    """可重试的API错误基类

    这类错误通常是临时性的，重试可能成功。
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None
    ):
        super().__init__(message, status_code, details)
        self.retry_after = retry_after  # 建议的重试等待时间（秒）


class FatalError(APIError):
    """不可重试的API错误基类

    这类错误是永久性的，重试不会改变结果。
    """
    pass


# ============ 可重试错误 ============

class RateLimitError(RetryableError):
    """速率限制错误 (HTTP 429)

    API调用频率超过限制时抛出。
    """

    def __init__(
        self,
        message: str = "API调用频率超过限制",
        retry_after: Optional[float] = None
    ):
        super().__init__(
            message=message,
            status_code=429,
            retry_after=retry_after or 60.0
        )


class NetworkError(RetryableError):
    """网络错误

    连接超时、DNS解析失败等网络问题。
    """

    def __init__(
        self,
        message: str = "网络连接失败",
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            details={"original_error": str(original_error)} if original_error else {}
        )
        self.original_error = original_error


class ServiceError(RetryableError):
    """服务端错误 (HTTP 5xx)

    API服务器临时不可用。
    """

    def __init__(
        self,
        message: str = "服务暂时不可用",
        status_code: int = 500
    ):
        super().__init__(
            message=message,
            status_code=status_code
        )


class TimeoutError(RetryableError):
    """请求超时错误

    API响应时间过长。
    """

    def __init__(
        self,
        message: str = "请求超时",
        timeout_seconds: Optional[float] = None
    ):
        super().__init__(
            message=message,
            details={"timeout_seconds": timeout_seconds} if timeout_seconds else {}
        )


# ============ 不可重试错误 ============

class AuthError(FatalError):
    """认证错误 (HTTP 401/403)

    API密钥无效或权限不足。
    """

    def __init__(
        self,
        message: str = "认证失败，请检查API密钥",
        status_code: int = 401
    ):
        super().__init__(
            message=message,
            status_code=status_code
        )


class BadRequestError(FatalError):
    """请求错误 (HTTP 400)

    请求参数无效。
    """

    def __init__(
        self,
        message: str = "请求参数无效",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=400,
            details=details
        )


class NotFoundError(FatalError):
    """资源不存在错误 (HTTP 404)

    请求的资源（如模型）不存在。
    """

    def __init__(
        self,
        message: str = "请求的资源不存在",
        resource: Optional[str] = None
    ):
        super().__init__(
            message=message,
            status_code=404,
            details={"resource": resource} if resource else {}
        )


class ModelNotFoundError(NotFoundError):
    """模型不存在错误

    指定的AI模型不存在或不可用。
    """

    def __init__(self, model_name: str):
        super().__init__(
            message=f"模型 '{model_name}' 不存在或不可用",
            resource=model_name
        )


class QuotaExceededError(FatalError):
    """配额耗尽错误

    API使用配额已用完。
    """

    def __init__(
        self,
        message: str = "API配额已用完"
    ):
        super().__init__(
            message=message,
            status_code=402
        )


class ContentFilterError(FatalError):
    """内容过滤错误

    请求或响应内容被安全过滤器阻止。
    """

    def __init__(
        self,
        message: str = "内容被安全过滤器阻止"
    ):
        super().__init__(
            message=message,
            status_code=400
        )


# ============ 业务错误 ============

class SessionError(APIError):
    """会话相关错误"""
    pass


class SessionNotFoundError(SessionError):
    """会话不存在"""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"会话 '{session_id}' 不存在或已过期",
            status_code=404,
            details={"session_id": session_id}
        )


class SessionExpiredError(SessionError):
    """会话已过期"""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"会话 '{session_id}' 已过期",
            status_code=410,
            details={"session_id": session_id}
        )


class OrderError(APIError):
    """订单相关错误"""
    pass


class OrderNotFoundError(OrderError):
    """订单不存在"""

    def __init__(self, order_id: str):
        super().__init__(
            message=f"订单 '{order_id}' 不存在",
            status_code=404,
            details={"order_id": order_id}
        )


class InvalidOrderStateError(OrderError):
    """订单状态无效"""

    def __init__(self, order_id: str, current_state: str, expected_states: list):
        super().__init__(
            message=f"订单 '{order_id}' 当前状态为 '{current_state}'，无法执行此操作",
            status_code=400,
            details={
                "order_id": order_id,
                "current_state": current_state,
                "expected_states": expected_states
            }
        )


# ============ 数据库错误 ============

class DatabaseError(APIError):
    """数据库相关错误"""
    pass


class DatabaseConnectionError(DatabaseError):
    """数据库连接错误"""

    def __init__(self, message: str = "数据库连接失败"):
        super().__init__(message=message, status_code=503)


class DatabaseQueryError(DatabaseError):
    """数据库查询错误"""

    def __init__(self, message: str = "数据库查询失败"):
        super().__init__(message=message, status_code=500)


# ============ 向量存储错误 ============

class VectorStoreError(APIError):
    """向量存储相关错误"""
    pass


class VectorStoreInitError(VectorStoreError):
    """向量存储初始化错误"""

    def __init__(self, message: str = "向量存储初始化失败"):
        super().__init__(message=message, status_code=503)


class EmbeddingError(VectorStoreError):
    """嵌入生成错误"""

    def __init__(self, message: str = "文本嵌入生成失败"):
        super().__init__(message=message, status_code=500)


def classify_openai_error(error: Exception) -> APIError:
    """将 OpenAI 库的异常转换为自定义异常

    Args:
        error: OpenAI 库抛出的异常

    Returns:
        对应的自定义异常
    """
    error_name = type(error).__name__
    error_message = str(error)

    # OpenAI 库的异常类型映射
    error_mapping = {
        "RateLimitError": RateLimitError,
        "APIConnectionError": NetworkError,
        "APITimeoutError": TimeoutError,
        "AuthenticationError": AuthError,
        "PermissionDeniedError": AuthError,
        "BadRequestError": BadRequestError,
        "NotFoundError": NotFoundError,
        "InternalServerError": ServiceError,
        "ServiceUnavailableError": ServiceError,
    }

    error_class = error_mapping.get(error_name)

    if error_class:
        if error_class == NetworkError:
            return NetworkError(message=error_message, original_error=error)
        elif error_class == AuthError:
            return AuthError(message=error_message)
        else:
            return error_class(message=error_message)

    # 根据 HTTP 状态码判断
    status_code = getattr(error, "status_code", None)
    if status_code:
        if status_code == 429:
            return RateLimitError(message=error_message)
        elif status_code == 401 or status_code == 403:
            return AuthError(message=error_message, status_code=status_code)
        elif status_code == 400:
            return BadRequestError(message=error_message)
        elif status_code == 404:
            return NotFoundError(message=error_message)
        elif 500 <= status_code < 600:
            return ServiceError(message=error_message, status_code=status_code)

    # 默认返回可重试错误
    return RetryableError(message=error_message)
