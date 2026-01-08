"""
槽位Schema配置管理模块

支持从YAML文件加载槽位定义，提供:
1. 槽位定义注册和管理
2. 值规范化和别名映射
3. 动态生成 Function Calling Schema
4. 意图定义和菜单数据管理
"""

import os
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum


# ==================== 数据类定义 ====================

@dataclass
class SlotValue:
    """槽位枚举值定义"""
    value: str
    aliases: List[str] = field(default_factory=list)
    price_delta: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, input_text: str) -> bool:
        """检查输入是否匹配该值"""
        input_lower = input_text.lower().strip()
        if self.value.lower() == input_lower:
            return True
        return any(alias.lower() == input_lower for alias in self.aliases)


@dataclass
class SlotDefinition:
    """槽位定义"""
    name: str
    type: str  # string, enum, array, number
    description: str
    required: bool = False
    default: Optional[Any] = None
    values: List[SlotValue] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    normalization: Dict[str, str] = field(default_factory=dict)
    number_mapping: Dict[str, int] = field(default_factory=dict)
    items_values: List[SlotValue] = field(default_factory=list)  # for array type

    def get_enum_values(self) -> List[str]:
        """获取枚举值列表"""
        return [v.value for v in self.values]

    def get_items_enum_values(self) -> List[str]:
        """获取数组元素的枚举值列表"""
        return [v.value for v in self.items_values]

    def get_all_aliases(self) -> Dict[str, str]:
        """获取所有别名到标准值的映射"""
        mapping = {}
        for v in self.values:
            mapping[v.value.lower()] = v.value
            for alias in v.aliases:
                mapping[alias.lower()] = v.value
        return mapping

    def get_items_aliases(self) -> Dict[str, str]:
        """获取数组元素的别名映射"""
        mapping = {}
        for v in self.items_values:
            mapping[v.value.lower()] = v.value
            for alias in v.aliases:
                mapping[alias.lower()] = v.value
        return mapping

    def normalize(self, input_value: str) -> Optional[str]:
        """将输入值规范化为标准值"""
        if not input_value:
            return None

        input_lower = input_value.lower().strip()

        # 1. 检查规范化映射
        if self.normalization:
            for key, std_value in self.normalization.items():
                if key.lower() == input_lower:
                    return std_value

        # 2. 检查枚举值和别名
        aliases = self.get_all_aliases()
        if input_lower in aliases:
            return aliases[input_lower]

        # 3. 模糊匹配 - 检查是否包含关键词
        for v in self.values:
            if v.value.lower() in input_lower or input_lower in v.value.lower():
                return v.value
            for alias in v.aliases:
                if alias.lower() in input_lower or input_lower in alias.lower():
                    return v.value

        return input_value  # 返回原值

    def normalize_number(self, input_value: Union[str, int]) -> int:
        """规范化数字值"""
        if isinstance(input_value, int):
            return input_value

        # 检查中文数字映射
        if self.number_mapping and input_value in self.number_mapping:
            return self.number_mapping[input_value]

        # 尝试直接转换
        try:
            return int(input_value)
        except ValueError:
            return self.default or 1

    def normalize_array(self, input_values: List[str]) -> List[str]:
        """规范化数组值"""
        if not input_values:
            return []

        aliases = self.get_items_aliases()
        result = []

        for val in input_values:
            val_lower = val.lower().strip()
            if val_lower in aliases:
                result.append(aliases[val_lower])
            else:
                # 模糊匹配
                matched = False
                for v in self.items_values:
                    if v.value.lower() in val_lower or val_lower in v.value.lower():
                        result.append(v.value)
                        matched = True
                        break
                    for alias in v.aliases:
                        if alias.lower() in val_lower or val_lower in alias.lower():
                            result.append(v.value)
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    result.append(val)

        return list(set(result))  # 去重

    def get_price_delta(self, value: str) -> float:
        """获取值的价格增量"""
        for v in self.values:
            if v.value == value:
                return v.price_delta
        for v in self.items_values:
            if v.value == value:
                return v.price_delta
        return 0

    def to_json_schema(self) -> Dict:
        """转换为 JSON Schema (用于 Function Calling)"""
        if self.type == "enum":
            return {
                "type": "string",
                "enum": self.get_enum_values(),
                "description": self.description
            }
        elif self.type == "array":
            if self.items_values:
                return {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": self.get_items_enum_values()
                    },
                    "description": self.description
                }
            else:
                return {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": self.description
                }
        elif self.type == "number":
            schema = {
                "type": "integer",
                "description": self.description
            }
            if "min" in self.validation:
                schema["minimum"] = self.validation["min"]
            if "max" in self.validation:
                schema["maximum"] = self.validation["max"]
            return schema
        else:
            return {
                "type": "string",
                "description": self.description
            }


@dataclass
class IntentDefinition:
    """意图定义"""
    name: str
    display_name: str
    description: str
    color: str = "#9E9E9E"
    icon: str = "question"
    examples: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.display_name,
            "desc": self.description,
            "color": self.color,
            "icon": self.icon
        }


@dataclass
class ProductDefinition:
    """产品定义"""
    name: str
    price: float
    calories: int
    description: str
    category: str = "咖啡"
    available: bool = True


# ==================== Schema注册中心 ====================

class SlotSchemaRegistry:
    """
    槽位Schema注册中心

    负责管理所有槽位、意图和菜单定义，提供:
    - 配置加载和热更新
    - 值规范化和验证
    - 动态Function Calling Schema生成
    """

    def __init__(self):
        self.slots: Dict[str, SlotDefinition] = {}
        self.intents: Dict[str, IntentDefinition] = {}
        self.products: Dict[str, ProductDefinition] = {}
        self.settings: Dict[str, Any] = {}
        self._watchers: List[Callable] = []
        self._version: str = "0.0.0"
        self._config_path: Optional[str] = None

    def load_from_yaml(self, path: str) -> 'SlotSchemaRegistry':
        """从YAML文件加载配置"""
        self._config_path = path

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._version = config.get("version", "0.0.0")
        self.settings = config.get("settings", {})

        # 加载槽位定义
        self._load_slots(config.get("slots", {}))

        # 加载意图定义
        self._load_intents(config.get("intents", {}))

        # 加载菜单
        self._load_menu(config.get("menu", {}))

        self._notify_watchers()
        return self

    def _load_slots(self, slots_config: Dict):
        """加载槽位配置"""
        self.slots.clear()

        for name, slot_config in slots_config.items():
            # 解析枚举值
            values = []
            for v in slot_config.get("values", []):
                if isinstance(v, dict):
                    values.append(SlotValue(
                        value=v["value"],
                        aliases=v.get("aliases", []),
                        price_delta=v.get("price_delta", 0),
                        metadata=v.get("metadata", {})
                    ))
                else:
                    values.append(SlotValue(value=str(v)))

            # 解析数组元素的枚举值
            items_values = []
            items_config = slot_config.get("items", {})
            for v in items_config.get("values", []):
                if isinstance(v, dict):
                    items_values.append(SlotValue(
                        value=v["value"],
                        aliases=v.get("aliases", []),
                        price_delta=v.get("price_delta", 0)
                    ))

            self.slots[name] = SlotDefinition(
                name=name,
                type=slot_config.get("type", "string"),
                description=slot_config.get("description", ""),
                required=slot_config.get("required", False),
                default=slot_config.get("default"),
                values=values,
                validation=slot_config.get("validation", {}),
                examples=slot_config.get("examples", []),
                normalization=slot_config.get("normalization", {}),
                number_mapping=slot_config.get("number_mapping", {}),
                items_values=items_values
            )

    def _load_intents(self, intents_config: Dict):
        """加载意图配置"""
        self.intents.clear()

        for name, intent_config in intents_config.items():
            self.intents[name] = IntentDefinition(
                name=name,
                display_name=intent_config.get("name", name),
                description=intent_config.get("description", ""),
                color=intent_config.get("color", "#9E9E9E"),
                icon=intent_config.get("icon", "question"),
                examples=intent_config.get("examples", []),
                keywords=intent_config.get("keywords", [])
            )

    def _load_menu(self, menu_config: Dict):
        """加载菜单配置"""
        self.products.clear()

        products_config = menu_config.get("products", {})
        for name, product_config in products_config.items():
            self.products[name] = ProductDefinition(
                name=name,
                price=product_config.get("price", 30),
                calories=product_config.get("calories", 0),
                description=product_config.get("description", ""),
                category=product_config.get("category", "咖啡"),
                available=product_config.get("available", True)
            )

    def reload(self) -> bool:
        """重新加载配置"""
        if self._config_path and os.path.exists(self._config_path):
            try:
                self.load_from_yaml(self._config_path)
                return True
            except Exception as e:
                print(f"配置重载失败: {e}")
                return False
        return False

    # ==================== 槽位操作 ====================

    def get_slot(self, name: str) -> Optional[SlotDefinition]:
        """获取槽位定义"""
        return self.slots.get(name)

    def register_slot(self, slot: SlotDefinition):
        """动态注册新槽位"""
        self.slots[slot.name] = slot
        self._notify_watchers()

    def unregister_slot(self, name: str):
        """移除槽位"""
        if name in self.slots:
            del self.slots[name]
            self._notify_watchers()

    def normalize_slots(self, raw_slots: Dict) -> Dict:
        """规范化所有槽位值"""
        normalized = {}

        for name, value in raw_slots.items():
            slot_def = self.get_slot(name)
            if not slot_def:
                normalized[name] = value
                continue

            if slot_def.type == "enum":
                norm_value = slot_def.normalize(value)
                if norm_value:
                    normalized[name] = norm_value
            elif slot_def.type == "number":
                normalized[name] = slot_def.normalize_number(value)
            elif slot_def.type == "array":
                if isinstance(value, list):
                    normalized[name] = slot_def.normalize_array(value)
                else:
                    normalized[name] = slot_def.normalize_array([value])
            else:
                # string type - 检查是否有规范化映射
                if slot_def.normalization:
                    norm_value = slot_def.normalize(value)
                    normalized[name] = norm_value if norm_value else value
                else:
                    normalized[name] = value

        return normalized

    def normalize_product_name(self, name: str) -> Optional[str]:
        """规范化产品名称"""
        if not name:
            return None

        # 使用 product_name 槽位的规范化逻辑
        slot_def = self.get_slot("product_name")
        if slot_def:
            normalized = slot_def.normalize(name)
            # 检查是否在菜单中
            if normalized in self.products:
                return normalized

        # 直接检查菜单
        if name in self.products:
            return name

        # 模糊匹配菜单
        for product_name in self.products:
            if product_name in name or name in product_name:
                return product_name

        return name

    def extract_extras_from_text(self, text: str) -> List[str]:
        """从文本中提取配料"""
        extras = []
        slot_def = self.get_slot("extras")

        if slot_def and slot_def.items_values:
            for item in slot_def.items_values:
                # 检查主值
                if item.value in text:
                    extras.append(item.value)
                    continue
                # 检查别名
                for alias in item.aliases:
                    if alias in text:
                        extras.append(item.value)
                        break

        return list(set(extras))

    # ==================== 意图操作 ====================

    def get_intent(self, name: str) -> Optional[IntentDefinition]:
        """获取意图定义"""
        return self.intents.get(name)

    def get_intent_descriptions(self) -> Dict[str, Dict]:
        """获取所有意图描述（兼容旧格式）"""
        return {name: intent.to_dict() for name, intent in self.intents.items()}

    # ==================== 产品操作 ====================

    def get_product(self, name: str) -> Optional[ProductDefinition]:
        """获取产品定义"""
        return self.products.get(name)

    def get_menu_dict(self) -> Dict[str, Dict]:
        """获取菜单字典（兼容旧格式）"""
        return {
            name: {
                "price": p.price,
                "calories": p.calories,
                "desc": p.description
            }
            for name, p in self.products.items()
            if p.available
        }

    def get_price_deltas(self) -> Dict[str, Dict[str, float]]:
        """获取所有价格增量"""
        result = {}

        # 杯型价格
        size_slot = self.get_slot("size")
        if size_slot:
            result["size"] = {v.value: v.price_delta for v in size_slot.values}

        # 奶类价格
        milk_slot = self.get_slot("milk_type")
        if milk_slot:
            result["milk_type"] = {v.value: v.price_delta for v in milk_slot.values}

        # 配料价格
        extras_slot = self.get_slot("extras")
        if extras_slot:
            result["extras"] = {v.value: v.price_delta for v in extras_slot.items_values}

        return result

    # ==================== Schema生成 ====================

    def generate_function_schema(self, include_intent: bool = True) -> Dict:
        """
        动态生成 Function Calling Schema

        Args:
            include_intent: 是否包含意图字段

        Returns:
            Function Calling 兼容的 Schema
        """
        properties = {}
        required = []

        # 添加意图字段
        if include_intent:
            properties["intent"] = {
                "type": "string",
                "enum": list(self.intents.keys()),
                "description": "识别的用户意图类型"
            }
            required.append("intent")

            properties["confidence"] = {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "意图识别置信度"
            }
            required.append("confidence")

        # 添加订单详情嵌套对象
        order_properties = {}
        for name, slot in self.slots.items():
            order_properties[name] = slot.to_json_schema()
            if slot.required:
                # 注意：槽位的required是针对订单的，不是针对整个请求的
                pass

        properties["order_details"] = {
            "type": "object",
            "properties": order_properties,
            "description": "订单详细信息"
        }

        # 添加推理说明
        properties["reasoning"] = {
            "type": "string",
            "description": "推理过程说明"
        }

        return {
            "name": "process_order_intent",
            "description": "处理咖啡点单相关的用户意图，识别意图类型并提取订单相关信息",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def generate_slot_extraction_schema(self) -> Dict:
        """生成纯槽位提取的 Schema（不含意图）"""
        properties = {}

        for name, slot in self.slots.items():
            properties[name] = slot.to_json_schema()

        return {
            "name": "extract_order_slots",
            "description": "从用户输入中提取点单相关的槽位信息",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": []
            }
        }

    # ==================== 监听器 ====================

    def on_change(self, callback: Callable[['SlotSchemaRegistry'], None]):
        """注册变更监听器"""
        self._watchers.append(callback)

    def _notify_watchers(self):
        """通知所有监听器"""
        for watcher in self._watchers:
            try:
                watcher(self)
            except Exception as e:
                print(f"监听器执行失败: {e}")

    # ==================== 调试和导出 ====================

    @property
    def version(self) -> str:
        """获取配置版本"""
        return self._version

    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            "version": self._version,
            "slots": {name: {
                "type": slot.type,
                "description": slot.description,
                "values": slot.get_enum_values() if slot.type == "enum" else None
            } for name, slot in self.slots.items()},
            "intents": list(self.intents.keys()),
            "products": list(self.products.keys())
        }

    def __repr__(self) -> str:
        return f"SlotSchemaRegistry(v{self._version}, slots={len(self.slots)}, intents={len(self.intents)}, products={len(self.products)})"


# ==================== 全局实例 ====================

_global_registry: Optional[SlotSchemaRegistry] = None


def get_schema_registry() -> SlotSchemaRegistry:
    """获取全局Schema注册中心"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SlotSchemaRegistry()
        # 尝试加载默认配置
        default_path = Path(__file__).parent.parent / "config" / "schema" / "slots.yaml"
        if default_path.exists():
            _global_registry.load_from_yaml(str(default_path))
            print(f"✅ Schema配置已加载: {_global_registry}")
    return _global_registry


def reload_schema() -> bool:
    """重新加载Schema配置"""
    registry = get_schema_registry()
    return registry.reload()


# ==================== 测试代码 ====================

def test_schema():
    """测试Schema功能"""
    print("=" * 60)
    print("槽位Schema配置测试")
    print("=" * 60)

    registry = get_schema_registry()
    print(f"\n{registry}")

    # 测试槽位规范化
    print("\n--- 槽位规范化测试 ---")
    test_cases = [
        {"product_name": "冰美式", "size": "大", "temperature": "冷的"},
        {"milk_type": "燕麦", "sweetness": "不加糖"},
        {"extras": ["浓缩", "焦糖"]},
        {"quantity": "两"},
    ]

    for raw in test_cases:
        normalized = registry.normalize_slots(raw)
        print(f"  原始: {raw}")
        print(f"  规范: {normalized}\n")

    # 测试产品名称规范化
    print("--- 产品名称规范化 ---")
    products = ["冰美式", "热拿铁", "澳白", "latte", "卡布"]
    for p in products:
        normalized = registry.normalize_product_name(p)
        print(f"  {p} -> {normalized}")

    # 测试 Function Schema 生成
    print("\n--- Function Calling Schema ---")
    schema = registry.generate_function_schema()
    print(f"  Schema名称: {schema['name']}")
    print(f"  参数数量: {len(schema['parameters']['properties'])}")

    # 测试意图描述
    print("\n--- 意图定义 ---")
    for name, intent in registry.intents.items():
        print(f"  {name}: {intent.display_name} - {intent.description[:20]}...")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_schema()
