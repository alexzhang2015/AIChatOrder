"""
核心类型定义

提供系统中使用的枚举和类型常量。
"""

from enum import Enum


class Intent(str, Enum):
    """意图类型枚举"""
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

    @classmethod
    def from_string(cls, value: str) -> 'Intent':
        """从字符串转换为枚举"""
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN


class SessionState(str, Enum):
    """会话状态枚举"""
    IDLE = "idle"
    ORDERING = "ordering"
    CONFIRMING = "confirming"
    MODIFYING = "modifying"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class OrderStatus(str, Enum):
    """订单状态枚举"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ClassifyMethod(str, Enum):
    """分类方法枚举"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    RAG = "rag_enhanced"
    FUNCTION_CALLING = "function_calling"

    @classmethod
    def is_valid(cls, method: str) -> bool:
        """检查方法是否有效"""
        return method in [m.value for m in cls]


class Size(str, Enum):
    """杯型枚举"""
    SMALL = "小杯"
    MEDIUM = "中杯"
    LARGE = "大杯"
    EXTRA_LARGE = "超大杯"


class Temperature(str, Enum):
    """温度枚举"""
    HOT = "热"
    WARM = "温"
    COLD = "冰"
    EXTRA_ICE = "少冰"
    NO_ICE = "去冰"


class Sweetness(str, Enum):
    """甜度枚举"""
    FULL = "全糖"
    HALF = "半糖"
    LESS = "少糖"
    THREE_QUARTER = "七分糖"
    NONE = "无糖"


class MilkType(str, Enum):
    """奶类型枚举"""
    REGULAR = "普通牛奶"
    OAT = "燕麦奶"
    SOY = "豆奶"
    ALMOND = "杏仁奶"
    COCONUT = "椰奶"
    NONE = "不加奶"


# 默认值
DEFAULT_SIZE = Size.MEDIUM
DEFAULT_TEMPERATURE = Temperature.HOT
DEFAULT_SWEETNESS = Sweetness.FULL
DEFAULT_MILK_TYPE = MilkType.REGULAR
DEFAULT_QUANTITY = 1


# 槽位配置
SLOT_CONFIGS = {
    "product_name": {
        "required": True,
        "prompt": "请问您想要什么饮品？"
    },
    "size": {
        "required": False,
        "default": DEFAULT_SIZE.value,
        "prompt": "请问需要什么杯型？（小杯/中杯/大杯/超大杯）"
    },
    "temperature": {
        "required": False,
        "default": DEFAULT_TEMPERATURE.value,
        "prompt": "请问需要热饮还是冰饮？"
    },
    "sweetness": {
        "required": False,
        "default": DEFAULT_SWEETNESS.value,
        "prompt": "请问甜度需要怎样？（全糖/半糖/少糖/无糖）"
    },
    "milk_type": {
        "required": False,
        "default": DEFAULT_MILK_TYPE.value,
        "prompt": "请问需要什么类型的奶？"
    },
    "quantity": {
        "required": False,
        "default": DEFAULT_QUANTITY,
        "prompt": "请问需要几杯？"
    }
}
