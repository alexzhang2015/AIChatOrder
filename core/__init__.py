"""
核心模块

提供抽象接口和类型定义，解决循环依赖问题。
"""

from .interfaces import (
    IntentClassifier,
    SlotExtractorInterface,
    RetrieverInterface,
    ClassificationResult,
    SlotDict
)
from .types import (
    Intent,
    SessionState,
    OrderStatus
)

__all__ = [
    # 接口
    "IntentClassifier",
    "SlotExtractorInterface",
    "RetrieverInterface",
    # 类型
    "ClassificationResult",
    "SlotDict",
    "Intent",
    "SessionState",
    "OrderStatus",
]
