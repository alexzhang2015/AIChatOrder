"""意图定义模块"""

from enum import Enum
from typing import Dict

from intent_registry import get_intent_registry


class Intent(str, Enum):
    """点单系统意图分类"""
    ORDER_NEW = "ORDER_NEW"
    ORDER_MODIFY = "ORDER_MODIFY"
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_QUERY = "ORDER_QUERY"
    PRODUCT_INFO = "PRODUCT_INFO"
    RECOMMEND = "RECOMMEND"
    CUSTOMIZE = "CUSTOMIZE"
    PAYMENT = "PAYMENT"
    COMPLAINT = "COMPLAINT"
    CHITCHAT = "CHITCHAT"
    UNKNOWN = "UNKNOWN"


# 意图注册中心实例
_intent_registry = get_intent_registry()


def get_intent_descriptions() -> Dict:
    """获取意图描述（动态加载）"""
    return _intent_registry.get_intent_descriptions()


# 兼容旧代码的静态引用
INTENT_DESCRIPTIONS = _intent_registry.get_intent_descriptions()
