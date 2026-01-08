"""
意图注册中心

从 YAML 配置文件动态加载意图定义，支持热更新。
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import yaml

logger = logging.getLogger(__name__)

# 默认配置路径
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "schema" / "intents.yaml"


@dataclass
class IntentDefinition:
    """意图定义"""
    name: str
    description: str
    color: str = "#9E9E9E"
    icon: str = "❓"
    keywords: List[str] = field(default_factory=list)
    priority: int = 100

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "desc": self.description,
            "color": self.color,
            "icon": self.icon
        }


@dataclass
class IntentRule:
    """意图匹配规则"""
    pattern: str
    intent: str
    confidence: float = 0.8
    _compiled: re.Pattern = field(default=None, repr=False)

    def __post_init__(self):
        self._compiled = re.compile(self.pattern)

    def match(self, text: str) -> Optional[float]:
        """尝试匹配文本，返回置信度或 None"""
        if self._compiled.search(text):
            return self.confidence
        return None


class IntentRegistry:
    """意图注册中心

    从 YAML 配置加载意图定义和规则。
    """

    def __init__(self, config_path: Optional[Path] = None):
        """初始化注册中心

        Args:
            config_path: 配置文件路径，默认使用 schema/intents.yaml
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.version = "1.0.0"
        self.intents: Dict[str, IntentDefinition] = {}
        self.rules: List[IntentRule] = []
        self._intent_enum = None

        self.load()

    def load(self) -> bool:
        """加载配置"""
        try:
            if not self.config_path.exists():
                logger.warning(f"意图配置文件不存在: {self.config_path}")
                self._load_defaults()
                return False

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            self.version = config.get("version", "1.0.0")

            # 加载意图定义
            for intent_id, intent_data in config.get("intents", {}).items():
                self.intents[intent_id] = IntentDefinition(
                    name=intent_data.get("name", intent_id),
                    description=intent_data.get("description", ""),
                    color=intent_data.get("color", "#9E9E9E"),
                    icon=intent_data.get("icon", "❓"),
                    keywords=intent_data.get("keywords", []),
                    priority=intent_data.get("priority", 100)
                )

            # 加载规则
            for rule_data in config.get("rules", []):
                self.rules.append(IntentRule(
                    pattern=rule_data.get("pattern", ""),
                    intent=rule_data.get("intent", "UNKNOWN"),
                    confidence=rule_data.get("confidence", 0.8)
                ))

            # 按优先级排序规则（优先级高的先匹配）
            self.rules.sort(
                key=lambda r: self.intents.get(r.intent, IntentDefinition("", "")).priority
            )

            logger.info(
                f"意图配置已加载: v{self.version}, "
                f"{len(self.intents)} 个意图, {len(self.rules)} 条规则"
            )
            return True

        except Exception as e:
            logger.error(f"加载意图配置失败: {e}")
            self._load_defaults()
            return False

    def _load_defaults(self):
        """加载默认配置"""
        default_intents = {
            "ORDER_NEW": IntentDefinition("新建订单", "用户想点新饮品", "#4CAF50", "🛒"),
            "ORDER_MODIFY": IntentDefinition("修改订单", "修改已点饮品的配置", "#2196F3", "✏️"),
            "ORDER_CANCEL": IntentDefinition("取消订单", "取消订单", "#f44336", "❌"),
            "ORDER_QUERY": IntentDefinition("查询订单", "查询订单状态", "#9C27B0", "🔍"),
            "PRODUCT_INFO": IntentDefinition("商品咨询", "价格、成分等信息", "#FF9800", "ℹ️"),
            "RECOMMEND": IntentDefinition("推荐请求", "请求推荐饮品", "#E91E63", "⭐"),
            "CUSTOMIZE": IntentDefinition("定制需求", "特殊定制需求", "#00BCD4", "🎨"),
            "PAYMENT": IntentDefinition("支付相关", "支付方式、优惠券等", "#8BC34A", "💳"),
            "COMPLAINT": IntentDefinition("投诉反馈", "投诉反馈", "#FF5722", "😤"),
            "CHITCHAT": IntentDefinition("闲聊", "问候、感谢等", "#607D8B", "💬"),
            "UNKNOWN": IntentDefinition("未知意图", "无法识别的意图", "#9E9E9E", "❓"),
        }
        self.intents = default_intents

        # 默认规则
        self.rules = [
            IntentRule(r'取消|不要了|算了|不点', 'ORDER_CANCEL', 0.95),
            IntentRule(r'换成?|改成?|加[一份]*|不要.*加', 'ORDER_MODIFY', 0.88),
            IntentRule(r'到哪|多久|状态|查.*订单', 'ORDER_QUERY', 0.92),
            IntentRule(r'多少钱|价格|卡路里|成分|有什么', 'PRODUCT_INFO', 0.85),
            IntentRule(r'推荐|好喝|建议|适合', 'RECOMMEND', 0.87),
            IntentRule(r'支付|付款|优惠|积分|买一送一', 'PAYMENT', 0.90),
            IntentRule(r'投诉|做错|太久|不满意', 'COMPLAINT', 0.88),
            IntentRule(r'你好|谢谢|天气|再见', 'CHITCHAT', 0.80),
            IntentRule(r'要|来|点|给我|帮我|想喝|来[份杯]', 'ORDER_NEW', 0.90),
        ]

        logger.info("已加载默认意图配置")

    def reload(self) -> bool:
        """重新加载配置"""
        self.intents.clear()
        self.rules.clear()
        self._intent_enum = None
        return self.load()

    def get_intent(self, intent_id: str) -> Optional[IntentDefinition]:
        """获取意图定义"""
        return self.intents.get(intent_id)

    def get_all_intents(self) -> Dict[str, IntentDefinition]:
        """获取所有意图"""
        return self.intents.copy()

    def get_intent_descriptions(self) -> Dict[str, Dict]:
        """获取所有意图描述（兼容旧格式）"""
        return {
            intent_id: intent.to_dict()
            for intent_id, intent in self.intents.items()
        }

    def get_intent_ids(self) -> List[str]:
        """获取所有意图 ID"""
        return list(self.intents.keys())

    def match_rules(self, text: str) -> Tuple[str, float]:
        """使用规则匹配意图

        Args:
            text: 用户输入文本

        Returns:
            (意图ID, 置信度) 元组
        """
        for rule in self.rules:
            confidence = rule.match(text)
            if confidence is not None:
                return rule.intent, confidence

        return "UNKNOWN", 0.5

    def create_enum(self) -> type:
        """动态创建 Intent 枚举类"""
        if self._intent_enum is None:
            enum_dict = {
                intent_id: intent_id
                for intent_id in self.intents.keys()
            }
            self._intent_enum = Enum("Intent", enum_dict, type=str)
        return self._intent_enum

    def validate_intent(self, intent_id: str) -> bool:
        """验证意图 ID 是否有效"""
        return intent_id in self.intents

    def stats(self) -> Dict:
        """获取统计信息"""
        return {
            "version": self.version,
            "intent_count": len(self.intents),
            "rule_count": len(self.rules),
            "config_path": str(self.config_path)
        }


# ==================== 全局实例 ====================

_registry: Optional[IntentRegistry] = None


def get_intent_registry() -> IntentRegistry:
    """获取意图注册中心实例（单例）"""
    global _registry
    if _registry is None:
        _registry = IntentRegistry()
    return _registry


def reload_intents() -> bool:
    """重新加载意图配置"""
    registry = get_intent_registry()
    return registry.reload()
