"""API 请求/响应模型"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator

# 验证常量
VALID_METHODS = {"zero_shot", "few_shot", "rag_enhanced", "function_calling"}
MAX_TEXT_LENGTH = 500
MIN_TEXT_LENGTH = 1


class ClassifyRequest(BaseModel):
    """意图分类请求"""
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="要分类的用户输入文本"
    )
    method: str = Field(
        default="function_calling",
        description="分类方法: zero_shot, few_shot, rag_enhanced, function_calling"
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("输入文本不能为空")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"输入文本过长，最大 {MAX_TEXT_LENGTH} 字符")
        return v

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in VALID_METHODS:
            raise ValueError(f"无效的分类方法: {v}，可选: {', '.join(VALID_METHODS)}")
        return v


class CompareRequest(BaseModel):
    """方法对比请求"""
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="要对比的用户输入文本"
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("输入文本不能为空")
        return v


class ChatRequest(BaseModel):
    """多轮对话请求"""
    message: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="用户消息"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，如果为空则创建新会话"
    )
    method: str = Field(
        default="function_calling",
        description="分类方法"
    )
    use_langgraph: bool = Field(
        default=True,
        description="是否使用 LangGraph 工作流"
    )

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("消息不能为空")
        return v


class ResetRequest(BaseModel):
    """重置会话请求"""
    session_id: str = Field(
        ...,
        description="要重置的会话ID"
    )
    use_langgraph: bool = Field(
        default=True,
        description="是否使用 LangGraph 工作流"
    )
