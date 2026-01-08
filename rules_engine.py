"""
组合约束规则引擎 - Phase 1 基础能力增强

提供:
1. CustomizationRulesEngine - 饮品定制规则引擎
2. FuzzyExpressionMatcher - 模糊表达匹配器
3. EnhancedSlotNormalizer - 增强槽位标准化器
4. 产品约束验证和自动修正
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from difflib import SequenceMatcher


# ==================== 数据类定义 ====================

@dataclass
class NormalizationResult:
    """标准化结果"""
    value: Any
    confidence: float
    method: str  # exact, alias, fuzzy, semantic, pattern, colloquial, passthrough
    original: str = ""
    warning: Optional[str] = None


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    adjusted_slots: Dict[str, Any]
    warnings: List[str]
    auto_corrections: List[Dict[str, Any]]


@dataclass
class FuzzyMatchResult:
    """模糊匹配结果"""
    matched: bool
    slot_name: str
    value: Any
    confidence: float
    pattern: str
    action: Optional[str] = None  # 可选的动作，如 add_extra_shot
    extra_mappings: Dict[str, Any] = field(default_factory=dict)


# ==================== 模糊表达匹配器 ====================

class FuzzyExpressionMatcher:
    """
    模糊表达匹配器

    处理用户的模糊表达，如:
    - "不要那么甜" -> 半糖
    - "浓一点" -> 加浓缩shot
    - "续命水" -> 美式咖啡 + 双份浓缩
    """

    def __init__(self, fuzzy_config: Dict[str, List[Dict]]):
        """
        Args:
            fuzzy_config: 从 slots_v2.yaml 加载的 fuzzy_expressions 配置
        """
        self.patterns = {}
        self._compile_patterns(fuzzy_config)

    def _compile_patterns(self, config: Dict):
        """编译正则表达式模式"""
        for slot_name, patterns in config.items():
            self.patterns[slot_name] = []
            for p in patterns:
                try:
                    compiled = re.compile(p.get("pattern", ""), re.IGNORECASE)
                    self.patterns[slot_name].append({
                        "regex": compiled,
                        "pattern": p.get("pattern", ""),
                        "maps_to": p.get("maps_to"),
                        "maps_to_product": p.get("maps_to_product"),
                        "maps_to_espresso": p.get("maps_to_espresso"),
                        "maps_to_extras": p.get("maps_to_extras"),
                        "maps_to_milk": p.get("maps_to_milk"),
                        "action": p.get("action"),
                        "confidence": p.get("confidence", 0.7)
                    })
                except re.error:
                    print(f"警告: 无效的正则表达式 '{p.get('pattern')}'")

    def match(self, text: str) -> List[FuzzyMatchResult]:
        """
        匹配文本中的模糊表达

        Args:
            text: 用户输入文本

        Returns:
            匹配结果列表
        """
        results = []

        for slot_name, patterns in self.patterns.items():
            for p in patterns:
                if p["regex"].search(text):
                    result = FuzzyMatchResult(
                        matched=True,
                        slot_name=slot_name,
                        value=p.get("maps_to"),
                        confidence=p["confidence"],
                        pattern=p["pattern"],
                        action=p.get("action"),
                        extra_mappings={
                            k: v for k, v in {
                                "product_name": p.get("maps_to_product"),
                                "espresso_shots": p.get("maps_to_espresso"),
                                "extras": p.get("maps_to_extras"),
                                "milk_type": p.get("maps_to_milk")
                            }.items() if v is not None
                        }
                    )
                    results.append(result)

        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def match_slot(self, text: str, slot_name: str) -> Optional[FuzzyMatchResult]:
        """
        匹配特定槽位的模糊表达

        Args:
            text: 用户输入文本
            slot_name: 目标槽位名称

        Returns:
            匹配结果或 None
        """
        if slot_name not in self.patterns:
            return None

        for p in self.patterns[slot_name]:
            if p["regex"].search(text):
                return FuzzyMatchResult(
                    matched=True,
                    slot_name=slot_name,
                    value=p.get("maps_to"),
                    confidence=p["confidence"],
                    pattern=p["pattern"],
                    action=p.get("action")
                )

        return None


# ==================== 组合约束规则引擎 ====================

class CustomizationRulesEngine:
    """
    饮品定制规则引擎

    负责:
    - 验证槽位组合是否合法
    - 自动修正不合法的组合
    - 应用产品级别的约束
    - 返回警告和修正信息
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 包含 product_constraints, combination_rules, product_categories 的配置
        """
        self.product_constraints = config.get("product_constraints", {})
        self.combination_rules = config.get("combination_rules", {})
        self.product_categories = config.get("product_categories", {})
        self.menu = config.get("menu", {}).get("products", {})

    def _get_product_category(self, product_name: str) -> Optional[str]:
        """获取产品所属分类"""
        for cat_id, cat_config in self.product_categories.items():
            if product_name in cat_config.get("products", []):
                return cat_id
        return None

    def _get_category_constraints(self, category: str) -> Dict:
        """获取分类级别的约束"""
        cat_config = self.product_categories.get(category, {})
        constraints = {}

        if "temperature_constraint" in cat_config:
            constraints["temperature"] = cat_config["temperature_constraint"]
        if "milk_type_constraint" in cat_config:
            constraints["milk_type"] = cat_config["milk_type_constraint"]

        return constraints

    def validate_and_adjust(self, product_name: str, slots: Dict) -> ValidationResult:
        """
        验证并调整槽位组合

        Args:
            product_name: 产品名称
            slots: 原始槽位值

        Returns:
            ValidationResult 包含调整后的槽位和警告
        """
        warnings = []
        auto_corrections = []
        adjusted = dict(slots)

        # 1. 应用产品级别约束
        product_rules = self.product_constraints.get(product_name, {})
        for slot_name, constraint in product_rules.items():
            if slot_name not in adjusted:
                continue

            current_value = adjusted[slot_name]

            # 检查 forbidden
            if constraint.get("forbidden"):
                if current_value:
                    warnings.append(constraint.get("error_message", f"{product_name}不支持{slot_name}定制"))
                    adjusted.pop(slot_name, None)
                    auto_corrections.append({
                        "slot": slot_name,
                        "from": current_value,
                        "to": None,
                        "reason": constraint.get("error_message")
                    })
                continue

            # 检查 only 约束
            allowed_values = constraint.get("only", [])
            if allowed_values and current_value not in allowed_values:
                auto_correct = constraint.get("auto_correct")
                if auto_correct:
                    adjusted[slot_name] = auto_correct
                    auto_corrections.append({
                        "slot": slot_name,
                        "from": current_value,
                        "to": auto_correct,
                        "reason": constraint.get("error_message")
                    })
                    warnings.append(constraint.get("error_message", f"已将{slot_name}调整为{auto_correct}"))
                else:
                    warnings.append(constraint.get("warning_message") or constraint.get("error_message", f"{slot_name}值不支持"))

        # 2. 应用分类级别约束
        category = self._get_product_category(product_name)
        if category:
            cat_constraints = self._get_category_constraints(category)
            for slot_name, constraint in cat_constraints.items():
                if slot_name not in adjusted:
                    continue

                current_value = adjusted[slot_name]

                if constraint.get("forbidden"):
                    if current_value:
                        adjusted.pop(slot_name, None)
                        auto_corrections.append({
                            "slot": slot_name,
                            "from": current_value,
                            "to": None,
                            "reason": f"{category}类产品不支持{slot_name}定制"
                        })
                    continue

                allowed_values = constraint.get("only", [])
                if allowed_values and current_value not in allowed_values:
                    default_value = constraint.get("default", allowed_values[0] if allowed_values else None)
                    if default_value:
                        adjusted[slot_name] = default_value
                        auto_corrections.append({
                            "slot": slot_name,
                            "from": current_value,
                            "to": default_value,
                            "reason": f"{category}类产品{slot_name}只支持{allowed_values}"
                        })

        # 3. 应用组合规则
        forbidden_rules = self.combination_rules.get("forbidden_combinations", [])
        for rule in forbidden_rules:
            condition = rule.get("condition", {})
            if self._matches_condition(product_name, adjusted, condition):
                fix = rule.get("fix", {})
                for slot_name, fix_value in fix.items():
                    old_value = adjusted.get(slot_name)
                    if old_value != fix_value:
                        adjusted[slot_name] = fix_value
                        auto_corrections.append({
                            "slot": slot_name,
                            "from": old_value,
                            "to": fix_value,
                            "reason": rule.get("message")
                        })
                        warnings.append(rule.get("message", f"已自动调整{slot_name}"))

        # 4. 应用产品默认值
        product_config = self.menu.get(product_name, {})
        default_options = product_config.get("default_options", {})
        for slot_name, default_value in default_options.items():
            if slot_name not in adjusted:
                adjusted[slot_name] = default_value

        return ValidationResult(
            valid=len([w for w in warnings if "错误" in w or "不支持" in w]) == 0,
            adjusted_slots=adjusted,
            warnings=warnings,
            auto_corrections=auto_corrections
        )

    def _matches_condition(self, product_name: str, slots: Dict, condition: Dict) -> bool:
        """检查是否匹配条件"""
        # 检查产品名称
        if "product_name" in condition:
            if product_name != condition["product_name"]:
                return False

        # 检查产品分类
        if "product_category" in condition:
            category = self._get_product_category(product_name)
            if category != condition["product_category"]:
                return False

        # 检查槽位值
        for slot_name, expected in condition.items():
            if slot_name in ["product_name", "product_category"]:
                continue

            actual = slots.get(slot_name)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False

        return True

    def get_available_options(self, product_name: str, current_slots: Dict) -> Dict[str, List[str]]:
        """
        根据当前选择，返回每个槽位的可用选项

        用于前端动态显示可选项
        """
        category = self._get_product_category(product_name)
        cat_config = self.product_categories.get(category, {})
        product_rules = self.product_constraints.get(product_name, {})

        available = {}

        # 温度选项
        if "temperature" in product_rules:
            constraint = product_rules["temperature"]
            if constraint.get("forbidden"):
                available["temperature"] = []
            elif "only" in constraint:
                available["temperature"] = constraint["only"]
        elif category and "temperature_constraint" in cat_config:
            constraint = cat_config["temperature_constraint"]
            available["temperature"] = constraint.get("only", [])
        else:
            available["temperature"] = ["热", "冰", "温", "去冰", "少冰", "多冰"]

        # 奶类选项
        if "milk_type" in product_rules:
            constraint = product_rules["milk_type"]
            if constraint.get("forbidden"):
                available["milk_type"] = []
            elif "only" in constraint:
                available["milk_type"] = constraint["only"]
        else:
            available["milk_type"] = ["全脂奶", "脱脂奶", "燕麦奶", "椰奶", "豆奶", "杏仁奶"]

        return available


# ==================== 增强槽位标准化器 ====================

class EnhancedSlotNormalizer:
    """
    增强的槽位标准化器

    多策略标准化:
    1. 精确匹配
    2. 别名匹配
    3. 模糊匹配 (编辑距离)
    4. 正则模式匹配
    5. 俚语/口语化表达匹配
    """

    def __init__(self, slots_config: Dict, fuzzy_config: Dict = None, colloquial_config: Dict = None):
        """
        Args:
            slots_config: 槽位定义配置
            fuzzy_config: 模糊表达配置
            colloquial_config: 俚语映射配置
        """
        self.slots_config = slots_config
        self.fuzzy_matcher = FuzzyExpressionMatcher(fuzzy_config or {})
        self.colloquial_mappings = colloquial_config or {}
        self.alias_index = self._build_alias_index()
        self.fuzzy_threshold = 0.8

    def _build_alias_index(self) -> Dict[str, Dict[str, str]]:
        """构建别名索引"""
        index = {}

        for slot_name, slot_config in self.slots_config.items():
            index[slot_name] = {}

            # 处理 enum 类型的值
            for value_config in slot_config.get("values", []):
                if isinstance(value_config, dict):
                    value = value_config.get("value", "")
                    index[slot_name][value.lower()] = value
                    for alias in value_config.get("aliases", []):
                        index[slot_name][alias.lower()] = value
                else:
                    index[slot_name][str(value_config).lower()] = str(value_config)

            # 处理 normalization 映射
            for key, std_value in slot_config.get("normalization", {}).items():
                index[slot_name][key.lower()] = std_value

            # 处理 colloquial_mappings (俚语)
            for key, std_value in slot_config.get("colloquial_mappings", {}).items():
                index[slot_name][key.lower()] = std_value

            # 处理 array 类型的 items
            items_config = slot_config.get("items", {})
            for value_config in items_config.get("values", []):
                if isinstance(value_config, dict):
                    value = value_config.get("value", "")
                    index[slot_name][value.lower()] = value
                    for alias in value_config.get("aliases", []):
                        index[slot_name][alias.lower()] = value

        return index

    def normalize(self, slot_name: str, raw_value: str, context: Dict = None) -> NormalizationResult:
        """
        多策略标准化

        Args:
            slot_name: 槽位名称
            raw_value: 原始值
            context: 上下文信息（如用户消息全文）

        Returns:
            NormalizationResult
        """
        if not raw_value:
            return NormalizationResult(value=None, confidence=0, method="empty", original=raw_value)

        input_lower = raw_value.lower().strip()
        slot_aliases = self.alias_index.get(slot_name, {})

        # 1. 精确匹配
        if input_lower in slot_aliases:
            return NormalizationResult(
                value=slot_aliases[input_lower],
                confidence=1.0,
                method="exact",
                original=raw_value
            )

        # 2. 别名匹配 (包含关系)
        for alias, std_value in slot_aliases.items():
            if alias in input_lower or input_lower in alias:
                return NormalizationResult(
                    value=std_value,
                    confidence=0.9,
                    method="alias",
                    original=raw_value
                )

        # 3. 模糊匹配 (编辑距离)
        best_match = None
        best_score = 0
        for alias, std_value in slot_aliases.items():
            score = SequenceMatcher(None, input_lower, alias).ratio()
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = std_value

        if best_match:
            return NormalizationResult(
                value=best_match,
                confidence=best_score,
                method="fuzzy",
                original=raw_value
            )

        # 4. 模糊表达模式匹配
        if context:
            fuzzy_result = self.fuzzy_matcher.match_slot(context.get("user_message", ""), slot_name)
            if fuzzy_result and fuzzy_result.value:
                return NormalizationResult(
                    value=fuzzy_result.value,
                    confidence=fuzzy_result.confidence,
                    method="pattern",
                    original=raw_value
                )

        # 5. 返回原值 (passthrough)
        return NormalizationResult(
            value=raw_value,
            confidence=0.5,
            method="passthrough",
            original=raw_value
        )

    def normalize_with_fuzzy(self, user_message: str, slots: Dict) -> Tuple[Dict, List[FuzzyMatchResult]]:
        """
        结合模糊表达进行标准化

        Args:
            user_message: 用户原始消息
            slots: LLM 提取的槽位

        Returns:
            (标准化后的槽位, 模糊匹配结果列表)
        """
        normalized = {}
        fuzzy_matches = []

        # 1. 先处理模糊表达
        all_fuzzy = self.fuzzy_matcher.match(user_message)
        for fm in all_fuzzy:
            if fm.value:
                # 将模糊匹配的值加入或覆盖
                if fm.slot_name not in normalized or fm.confidence > 0.8:
                    if fm.slot_name in ["sweetness", "temperature", "milk"]:
                        actual_slot = "milk_type" if fm.slot_name == "milk" else fm.slot_name
                        normalized[actual_slot] = fm.value
                        fuzzy_matches.append(fm)

            # 处理额外映射
            for key, value in fm.extra_mappings.items():
                if key == "extras" and value:
                    if "extras" not in normalized:
                        normalized["extras"] = []
                    if isinstance(value, list):
                        normalized["extras"].extend(value)
                    else:
                        normalized["extras"].append(value)
                elif value:
                    normalized[key] = value

            fuzzy_matches.append(fm)

        # 2. 标准化 LLM 提取的槽位
        context = {"user_message": user_message}
        for slot_name, value in slots.items():
            if slot_name in normalized:
                # 模糊匹配已处理
                continue

            if isinstance(value, list):
                # 处理数组类型
                normalized_list = []
                for item in value:
                    result = self.normalize(slot_name, str(item), context)
                    if result.value:
                        normalized_list.append(result.value)
                normalized[slot_name] = list(set(normalized_list))
            else:
                result = self.normalize(slot_name, str(value), context)
                normalized[slot_name] = result.value

        return normalized, fuzzy_matches

    def extract_colloquial_intent(self, user_message: str) -> Optional[Dict]:
        """
        从俚语/口语化表达中提取意图

        例如: "来杯续命水" -> {"product_name": "美式咖啡", "espresso_shots": "双份"}
        """
        fuzzy_matches = self.fuzzy_matcher.match(user_message)

        for fm in fuzzy_matches:
            if fm.slot_name == "colloquial" and fm.extra_mappings:
                return fm.extra_mappings

        return None


# ==================== 全局加载函数 ====================

_rules_engine: Optional[CustomizationRulesEngine] = None
_fuzzy_matcher: Optional[FuzzyExpressionMatcher] = None
_enhanced_normalizer: Optional[EnhancedSlotNormalizer] = None
_config: Optional[Dict] = None


def load_v2_config() -> Dict:
    """加载 v2 配置"""
    global _config
    if _config is None:
        config_path = Path(__file__).parent / "config" / "schema" / "slots_v2.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                _config = yaml.safe_load(f)
            print(f"✅ Rules Engine 配置已加载: v{_config.get('version', '2.0.0')}")
        else:
            # 回退到 v1 配置
            config_path = Path(__file__).parent / "config" / "schema" / "slots.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                _config = yaml.safe_load(f)
            print(f"⚠️ 使用 v1 配置，部分功能不可用")
    return _config


def get_rules_engine() -> CustomizationRulesEngine:
    """获取规则引擎实例"""
    global _rules_engine
    if _rules_engine is None:
        config = load_v2_config()
        _rules_engine = CustomizationRulesEngine(config)
    return _rules_engine


def get_fuzzy_matcher() -> FuzzyExpressionMatcher:
    """获取模糊匹配器实例"""
    global _fuzzy_matcher
    if _fuzzy_matcher is None:
        config = load_v2_config()
        _fuzzy_matcher = FuzzyExpressionMatcher(config.get("fuzzy_expressions", {}))
    return _fuzzy_matcher


def get_enhanced_normalizer() -> EnhancedSlotNormalizer:
    """获取增强标准化器实例"""
    global _enhanced_normalizer
    if _enhanced_normalizer is None:
        config = load_v2_config()
        _enhanced_normalizer = EnhancedSlotNormalizer(
            slots_config=config.get("slots", {}),
            fuzzy_config=config.get("fuzzy_expressions", {}),
            colloquial_config=config.get("slots", {}).get("product_name", {}).get("colloquial_mappings", {})
        )
    return _enhanced_normalizer


# ==================== 测试代码 ====================

def test_rules_engine():
    """测试规则引擎"""
    print("=" * 60)
    print("Phase 1 规则引擎测试")
    print("=" * 60)

    # 加载配置
    engine = get_rules_engine()
    fuzzy = get_fuzzy_matcher()
    normalizer = get_enhanced_normalizer()

    # 测试1: 产品约束验证
    print("\n--- 产品约束测试 ---")
    test_cases = [
        ("星冰乐", {"temperature": "热", "size": "大杯"}),
        ("馥芮白", {"temperature": "冰", "milk_type": "燕麦奶"}),
        ("美式咖啡", {"temperature": "冰", "milk_type": "燕麦奶"}),
        ("拿铁", {"temperature": "冰", "sweetness": "无糖"}),
    ]

    for product, slots in test_cases:
        result = engine.validate_and_adjust(product, slots)
        print(f"\n{product} + {slots}")
        print(f"  调整后: {result.adjusted_slots}")
        if result.warnings:
            print(f"  警告: {result.warnings}")
        if result.auto_corrections:
            print(f"  自动修正: {result.auto_corrections}")

    # 测试2: 模糊表达匹配
    print("\n--- 模糊表达测试 ---")
    fuzzy_tests = [
        "我不要那么甜的",
        "来杯续命水",
        "咖啡味浓一点",
        "我想喝健康一点的",
        "给我来杯肥宅快乐水",
        "要一杯dirty",
        "不要太烫",
    ]

    for text in fuzzy_tests:
        matches = fuzzy.match(text)
        print(f"\n'{text}'")
        if matches:
            for m in matches:
                print(f"  -> {m.slot_name}: {m.value} (置信度: {m.confidence:.2f})")
                if m.extra_mappings:
                    print(f"     额外映射: {m.extra_mappings}")
        else:
            print("  -> 无匹配")

    # 测试3: 增强标准化
    print("\n--- 增强标准化测试 ---")
    normalize_tests = [
        ("product_name", "冰美式"),
        ("product_name", "澳白"),
        ("product_name", "latte"),
        ("size", "大"),
        ("size", "venti"),
        ("temperature", "凉的"),
        ("milk_type", "oat"),
        ("sweetness", "不加糖"),
    ]

    for slot, value in normalize_tests:
        result = normalizer.normalize(slot, value)
        print(f"  {slot}='{value}' -> '{result.value}' ({result.method}, 置信度: {result.confidence:.2f})")

    # 测试4: 结合模糊表达的标准化
    print("\n--- 综合标准化测试 ---")
    combined_tests = [
        ("来杯续命水，不要那么甜", {"product_name": "美式"}),
        ("大杯冰拿铁，浓一点", {"product_name": "拿铁", "size": "大", "temperature": "冰"}),
        ("我要一杯健康的燕麦拿铁", {"product_name": "拿铁", "milk_type": "燕麦"}),
    ]

    for user_msg, slots in combined_tests:
        normalized, fuzzy_matches = normalizer.normalize_with_fuzzy(user_msg, slots)
        print(f"\n'{user_msg}'")
        print(f"  原始槽位: {slots}")
        print(f"  标准化后: {normalized}")
        if fuzzy_matches:
            print(f"  模糊匹配: {[(m.slot_name, m.value) for m in fuzzy_matches if m.value]}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_rules_engine()
