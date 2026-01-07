"""
输入验证模块

提供增强的输入验证功能，包括安全检查和业务规则验证。
"""

import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from .types import ClassifyMethod


# ==================== 验证常量 ====================

MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 2000
MAX_SESSION_ID_LENGTH = 50
MAX_ITEMS_PER_ORDER = 20

# 危险模式（可能的注入攻击）
DANGEROUS_PATTERNS = [
    r'<\s*script',
    r'javascript\s*:',
    r'on\w+\s*=',
    r'<\s*iframe',
    r'<\s*object',
    r'<\s*embed',
    r'\{\{.*\}\}',  # 模板注入
    r'\$\{.*\}',    # 模板注入
]

# SQL 注入关键字（用于警告，不阻止）
SQL_KEYWORDS = [
    'select', 'insert', 'update', 'delete', 'drop', 'union',
    'exec', 'execute', '--', ';--', '/*', '*/'
]


# ==================== 验证函数 ====================

def check_dangerous_patterns(text: str) -> List[str]:
    """检查危险模式

    Returns:
        匹配到的危险模式列表
    """
    matched = []
    text_lower = text.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(pattern)
    return matched


def check_sql_injection(text: str) -> bool:
    """检查可能的 SQL 注入

    Returns:
        是否包含 SQL 关键字
    """
    text_lower = text.lower()
    for keyword in SQL_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def sanitize_text(text: str) -> str:
    """清理文本

    移除或转义潜在的危险字符。
    """
    # 移除控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 标准化空白
    text = ' '.join(text.split())
    return text


def validate_session_id(session_id: str) -> bool:
    """验证会话 ID 格式

    支持 UUID 格式和简单的字母数字格式。
    """
    # UUID 格式
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    if re.match(uuid_pattern, session_id, re.IGNORECASE):
        return True

    # 简单字母数字格式
    simple_pattern = r'^[a-zA-Z0-9_-]{1,50}$'
    if re.match(simple_pattern, session_id):
        return True

    return False


# ==================== Pydantic 验证模型 ====================

class TextInputMixin:
    """文本输入验证混入"""

    @staticmethod
    def validate_text_content(v: str, field_name: str = "text") -> str:
        """验证文本内容"""
        # 清理空白
        v = v.strip()
        if not v:
            raise ValueError(f"{field_name}不能为空")

        # 清理危险字符
        v = sanitize_text(v)

        # 检查危险模式
        dangerous = check_dangerous_patterns(v)
        if dangerous:
            raise ValueError(f"输入包含不允许的内容")

        return v


class ClassifyRequestV2(BaseModel):
    """增强的分类请求模型"""
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="用户输入文本"
    )
    method: str = Field(
        default="function_calling",
        description="分类方法"
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("输入文本不能为空")

        v = sanitize_text(v)
        dangerous = check_dangerous_patterns(v)
        if dangerous:
            raise ValueError("输入包含不允许的内容")

        return v

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid_methods = ["zero_shot", "few_shot", "rag_enhanced", "function_calling"]
        if v not in valid_methods:
            raise ValueError(f"无效的分类方法: {v}，有效值: {valid_methods}")
        return v


class ChatRequestV2(BaseModel):
    """增强的对话请求模型"""
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SESSION_ID_LENGTH,
        description="会话ID"
    )
    message: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="用户消息"
    )

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not validate_session_id(v):
            raise ValueError("无效的会话ID格式")
        return v

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("消息不能为空")

        v = sanitize_text(v)
        dangerous = check_dangerous_patterns(v)
        if dangerous:
            raise ValueError("消息包含不允许的内容")

        return v


class OrderItemRequest(BaseModel):
    """订单项请求模型"""
    product_name: str = Field(..., min_length=1, max_length=100)
    size: Optional[str] = Field(default="中杯", max_length=20)
    temperature: Optional[str] = Field(default="热", max_length=20)
    sweetness: Optional[str] = Field(default="全糖", max_length=20)
    milk_type: Optional[str] = Field(default=None, max_length=50)
    extras: Optional[List[str]] = Field(default_factory=list, max_length=10)
    quantity: int = Field(default=1, ge=1, le=99)

    @field_validator('product_name')
    @classmethod
    def validate_product_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("产品名称不能为空")
        return sanitize_text(v)

    @field_validator('extras')
    @classmethod
    def validate_extras(cls, v: List[str]) -> List[str]:
        if v is None:
            return []
        return [sanitize_text(e.strip()) for e in v if e.strip()]


class OrderRequest(BaseModel):
    """订单请求模型"""
    session_id: str = Field(..., min_length=1, max_length=MAX_SESSION_ID_LENGTH)
    items: List[OrderItemRequest] = Field(..., min_length=1, max_length=MAX_ITEMS_PER_ORDER)

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not validate_session_id(v):
            raise ValueError("无效的会话ID格式")
        return v

    @model_validator(mode='after')
    def validate_order(self) -> 'OrderRequest':
        """验证整个订单"""
        total_quantity = sum(item.quantity for item in self.items)
        if total_quantity > 50:
            raise ValueError("单次订单总数量不能超过50杯")
        return self


# ==================== 导出 ====================

__all__ = [
    'MIN_TEXT_LENGTH',
    'MAX_TEXT_LENGTH',
    'MAX_SESSION_ID_LENGTH',
    'MAX_ITEMS_PER_ORDER',
    'check_dangerous_patterns',
    'check_sql_injection',
    'sanitize_text',
    'validate_session_id',
    'ClassifyRequestV2',
    'ChatRequestV2',
    'OrderItemRequest',
    'OrderRequest',
]
