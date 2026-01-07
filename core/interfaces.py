"""
抽象接口定义

定义核心组件的抽象接口，用于解耦模块依赖。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass


class SlotDict(TypedDict, total=False):
    """槽位字典类型定义"""
    product_name: str
    size: str
    temperature: str
    sweetness: str
    milk_type: str
    extras: List[str]
    quantity: int


@dataclass
class ClassificationResult:
    """分类结果"""
    intent: str
    confidence: float
    slots: SlotDict
    reasoning: str = ""
    method: str = ""
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "slots": dict(self.slots) if self.slots else {},
            "reasoning": self.reasoning,
            "method": self.method,
            "cached": self.cached
        }


class IntentClassifier(ABC):
    """意图分类器抽象接口

    定义意图分类器的标准接口，支持多种实现（OpenAI、本地模型等）。
    """

    @abstractmethod
    def classify(self, text: str, method: str = "zero_shot") -> Dict[str, Any]:
        """同步分类

        Args:
            text: 用户输入文本
            method: 分类方法

        Returns:
            分类结果字典
        """
        pass

    @abstractmethod
    async def classify_async(self, text: str, method: str = "zero_shot") -> Dict[str, Any]:
        """异步分类

        Args:
            text: 用户输入文本
            method: 分类方法

        Returns:
            分类结果字典
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查分类器是否可用"""
        pass


class SlotExtractorInterface(ABC):
    """槽位提取器抽象接口"""

    @abstractmethod
    def extract(self, text: str) -> SlotDict:
        """从文本中提取槽位

        Args:
            text: 用户输入文本

        Returns:
            提取的槽位字典
        """
        pass

    @abstractmethod
    def validate_slots(self, slots: SlotDict, required: List[str]) -> List[str]:
        """验证槽位完整性

        Args:
            slots: 槽位字典
            required: 必需的槽位列表

        Returns:
            缺失的槽位列表
        """
        pass


@dataclass
class RetrievalResult:
    """检索结果"""
    text: str
    intent: str
    slots: SlotDict
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "intent": self.intent,
            "slots": dict(self.slots) if self.slots else {},
            "similarity": self.similarity
        }


class RetrieverInterface(ABC):
    """检索器抽象接口"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """检索相似示例

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        pass

    @abstractmethod
    def add_example(self, text: str, intent: str, slots: Optional[SlotDict] = None):
        """添加示例

        Args:
            text: 示例文本
            intent: 意图标签
            slots: 槽位字典
        """
        pass

    @property
    @abstractmethod
    def example_count(self) -> int:
        """示例数量"""
        pass


class CacheInterface(ABC):
    """缓存抽象接口"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存"""
        pass

    @abstractmethod
    def clear(self) -> int:
        """清空缓存，返回清除的条目数"""
        pass


class SessionManagerInterface(ABC):
    """会话管理器抽象接口"""

    @abstractmethod
    def create_session(self, session_id: Optional[str] = None) -> str:
        """创建会话，返回会话ID"""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Any]:
        """获取会话"""
        pass

    @abstractmethod
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话"""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        pass
