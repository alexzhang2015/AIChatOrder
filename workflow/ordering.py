"""
LangGraph 工作流实现 - AI点单意图识别系统

使用 LangGraph 重构对话流程:
1. 意图识别节点
2. 业务处理节点 (新订单/修改/取消/查询/推荐等)
3. 响应生成节点
4. 状态持久化支持 (SQLite 数据库)
5. 配置化槽位支持 (YAML Schema)
6. 技能执行层 (Skills) 支持
"""

import os
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Annotated, TypedDict, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 导入重构后的模块
from services.classifier import OpenAIClassifier
from models.intent import INTENT_DESCRIPTIONS
from data.training import TRAINING_EXAMPLES
from nlp.extractor import SlotExtractor

# 导入数据库模块
from infrastructure.database import (
    Database, SessionRepository, OrderRepository, MessageRepository,
    SessionModel, OrderModel, OrderItemModel, MessageModel
)

logger = logging.getLogger(__name__)

# 导入配置化槽位模块
from models.slot_schema import get_schema_registry, SlotSchemaRegistry

# 导入技能执行层
from services.skills import get_skill_registry, SkillExecutor, SkillRegistry, SkillResult

# Phase 1: 导入规则引擎
from services.rules_engine import (
    get_rules_engine, get_fuzzy_matcher, get_enhanced_normalizer,
    CustomizationRulesEngine, FuzzyExpressionMatcher, EnhancedSlotNormalizer
)

# 导入优化模块
from infrastructure.monitoring import (
    get_structured_logger, get_metrics_collector,
    monitor_performance
)
from config import get_settings


# ==================== 状态定义 ====================

class OrderItemDict(TypedDict, total=False):
    """订单项字典"""
    product_name: str
    size: str
    temperature: str
    sweetness: str
    milk_type: str
    extras: List[str]
    quantity: int
    price: float
    description: str


class OrderDict(TypedDict, total=False):
    """订单字典"""
    order_id: str
    items: List[OrderItemDict]
    total: float
    status: str
    created_at: str


class MessageDict(TypedDict):
    """消息字典"""
    role: str
    content: str
    timestamp: str
    intent_info: Optional[Dict]


class OrderState(TypedDict, total=False):
    """
    LangGraph 工作流状态

    使用 Annotated 和 operator.add 实现消息列表的累加
    """
    # 会话信息
    session_id: str
    conversation_state: str  # greeting, taking_order, confirming, modifying, payment, completed

    # 当前轮次信息
    user_message: str
    intent: str
    confidence: float
    slots: Dict[str, Any]
    intent_result: Dict[str, Any]

    # 订单信息
    current_order: Optional[OrderDict]

    # 响应信息
    response: str
    suggestions: List[str]
    actions: List[str]

    # 历史消息 (使用累加方式)
    messages: Annotated[List[MessageDict], operator.add]

    # 技能执行结果
    skill_result: Optional[Dict[str, Any]]

    # 控制流
    next_node: str
    should_end: bool

    # 执行过程跟踪
    execution_trace: Annotated[List[Dict[str, Any]], operator.add]


# ==================== 订单项管理 ====================

class OrderItemManager:
    """订单项管理器 - 处理订单的创建和修改（支持配置化 + Phase 1 规则引擎）"""

    def __init__(self, schema_registry: Optional[SlotSchemaRegistry] = None):
        self.registry = schema_registry or get_schema_registry()
        # 从配置获取菜单和价格
        self._menu = self.registry.get_menu_dict()
        self._price_deltas = self.registry.get_price_deltas()
        # Phase 1: 初始化规则引擎
        self._rules_engine = get_rules_engine()
        self._enhanced_normalizer = get_enhanced_normalizer()

    def create_item(self, slots: Dict, user_message: str = "") -> Tuple[OrderItemDict, List[str]]:
        """
        创建订单项 (Phase 1 增强版)

        Args:
            slots: 原始槽位
            user_message: 用户原始消息（用于模糊表达匹配）

        Returns:
            (订单项, 警告列表)
        """
        warnings = []

        # Phase 1: 使用增强标准化器处理模糊表达
        if user_message:
            normalized_slots, fuzzy_matches = self._enhanced_normalizer.normalize_with_fuzzy(user_message, slots)
            # 记录模糊匹配信息
            for fm in fuzzy_matches:
                if fm.value and fm.confidence < 1.0:
                    warnings.append(f"理解「{fm.pattern}」为「{fm.value}」")
        else:
            # 回退到基础规范化
            normalized_slots = self.registry.normalize_slots(slots)

        product_name = normalized_slots.get("product_name", "")

        # Phase 1: 使用规则引擎验证和调整组合
        validation_result = self._rules_engine.validate_and_adjust(product_name, normalized_slots)
        adjusted_slots = validation_result.adjusted_slots
        warnings.extend(validation_result.warnings)

        # 从调整后的槽位获取值
        size = adjusted_slots.get("size", "中杯")
        temperature = adjusted_slots.get("temperature", "热")
        sweetness = adjusted_slots.get("sweetness", "标准")
        milk_type = adjusted_slots.get("milk_type", "全脂奶")
        extras = adjusted_slots.get("extras", [])
        # 确保 quantity 是整数
        quantity = adjusted_slots.get("quantity", 1)
        if isinstance(quantity, str):
            try:
                quantity = int(quantity)
            except ValueError:
                quantity = 1

        # 使用配置计算价格
        base = self._menu.get(product_name, {}).get("price", 30)
        size_add = self._price_deltas.get("size", {}).get(size, 0)
        milk_add = self._price_deltas.get("milk_type", {}).get(milk_type, 0)
        extras_prices = self._price_deltas.get("extras", {})
        extras_add = sum(extras_prices.get(e, 0) for e in extras)
        price = (base + size_add + milk_add + extras_add) * quantity

        # 生成描述
        parts = []
        if quantity > 1:
            parts.append(f"{quantity}杯")
        parts.append(size)
        parts.append(temperature)
        if sweetness != "标准":
            parts.append(sweetness)
        if milk_type != "全脂奶" and milk_type != "无奶":
            parts.append(milk_type)
        parts.append(product_name)
        if extras:
            parts.append(f"加{'/'.join(extras)}")
        description = "".join(parts)

        item = OrderItemDict(
            product_name=product_name,
            size=size,
            temperature=temperature,
            sweetness=sweetness,
            milk_type=milk_type,
            extras=extras,
            quantity=quantity,
            price=price,
            description=description
        )

        return item, warnings

    def update_item(self, item: OrderItemDict, slots: Dict) -> tuple[OrderItemDict, List[str]]:
        """更新订单项,返回更新后的项和修改说明列表"""
        # 使用Schema规范化槽位值
        normalized_slots = self.registry.normalize_slots(slots)
        modified = []
        new_item = dict(item)

        if "size" in normalized_slots:
            new_item["size"] = normalized_slots["size"]
            modified.append(f"杯型改为{normalized_slots['size']}")

        if "temperature" in normalized_slots:
            new_item["temperature"] = normalized_slots["temperature"]
            modified.append(f"温度改为{normalized_slots['temperature']}")

        if "sweetness" in normalized_slots:
            new_item["sweetness"] = normalized_slots["sweetness"]
            modified.append(f"甜度改为{normalized_slots['sweetness']}")

        if "milk_type" in normalized_slots:
            new_item["milk_type"] = normalized_slots["milk_type"]
            modified.append(f"奶类改为{normalized_slots['milk_type']}")

        if "extras" in normalized_slots:
            current_extras = list(new_item.get("extras", []))
            current_extras.extend(normalized_slots["extras"])
            new_item["extras"] = list(set(current_extras))  # 去重
            modified.append(f"添加{'/'.join(normalized_slots['extras'])}")

        # 处理产品名作为配料的情况 - 使用配置提取
        product_in_slot = slots.get("product_name", "")
        if product_in_slot and product_in_slot not in self._menu:
            extras_from_text = self.registry.extract_extras_from_text(product_in_slot)
            for extra in extras_from_text:
                current_extras = list(new_item.get("extras", []))
                if extra not in current_extras:
                    current_extras.append(extra)
                    new_item["extras"] = current_extras
                    modified.append(f"添加{extra}")

        # 重新计算价格和描述
        if modified:
            base = self._menu.get(new_item["product_name"], {}).get("price", 30)
            size_add = self._price_deltas.get("size", {}).get(new_item["size"], 0)
            milk_add = self._price_deltas.get("milk_type", {}).get(new_item["milk_type"], 0)
            extras_prices = self._price_deltas.get("extras", {})
            extras_add = sum(extras_prices.get(e, 0) for e in new_item.get("extras", []))
            new_item["price"] = (base + size_add + milk_add + extras_add) * new_item.get("quantity", 1)

            # 重新生成描述
            parts = []
            if new_item.get("quantity", 1) > 1:
                parts.append(f"{new_item['quantity']}杯")
            parts.append(new_item["size"])
            parts.append(new_item["temperature"])
            if new_item.get("sweetness", "标准") != "标准":
                parts.append(new_item["sweetness"])
            if new_item.get("milk_type", "全脂奶") != "全脂奶":
                parts.append(new_item["milk_type"])
            parts.append(new_item["product_name"])
            if new_item.get("extras"):
                parts.append(f"加{'/'.join(new_item['extras'])}")
            new_item["description"] = "".join(parts)

        return new_item, modified


# ==================== 工作流节点函数 ====================

class WorkflowNodes:
    """工作流节点集合 - 支持配置化和技能执行 (Phase 1 增强版)"""

    def __init__(self, classifier: OpenAIClassifier,
                 schema_registry: Optional[SlotSchemaRegistry] = None,
                 skill_registry: Optional[SkillRegistry] = None):
        self.classifier = classifier
        self.registry = schema_registry or get_schema_registry()
        self.item_manager = OrderItemManager(self.registry)

        # 从配置获取菜单
        self._menu = self.registry.get_menu_dict()

        # 初始化技能执行器
        self.skill_registry = skill_registry or get_skill_registry()
        self.skill_executor = SkillExecutor(self.skill_registry)

        # Phase 1: 初始化增强标准化器和规则引擎
        self._rules_engine = get_rules_engine()
        self._fuzzy_matcher = get_fuzzy_matcher()
        self._enhanced_normalizer = get_enhanced_normalizer()

    def _normalize_product_name(self, name: str) -> Optional[str]:
        """规范化产品名称 - 使用配置"""
        return self.registry.normalize_product_name(name)

    def _try_execute_skill(self, user_message: str, intent: str, slots: Dict) -> Optional[SkillResult]:
        """
        尝试匹配并执行技能

        优先级:
        1. 精确意图匹配
        2. 关键词匹配
        """
        # 尝试匹配技能
        matches = self.skill_registry.find_matching_skills(user_message, intent)

        if matches:
            matched_skill, score = matches[0]  # 取最佳匹配

            # 准备参数
            params = dict(slots) if slots else {}

            # 对于特定技能，补充必要参数
            if matched_skill.id == "nutrition_info" and "product_name" in params:
                pass  # 已有产品名
            elif matched_skill.id == "check_inventory" and "product_name" in params:
                pass  # 已有产品名
            elif matched_skill.id == "smart_recommend":
                # 检测天气相关词汇
                if "热" in user_message or "炎热" in user_message:
                    params["weather"] = "hot"
                elif "冷" in user_message or "冬天" in user_message:
                    params["weather"] = "cold"
                # 检测偏好
                if "减肥" in user_message or "低卡" in user_message:
                    params["preference"] = "适合减肥"

            # 执行技能
            return self.skill_executor.execute(matched_skill.id, params)

        return None

    # ==================== 意图识别节点 ====================

    def intent_recognition(self, state: OrderState) -> Dict:
        """
        意图识别节点

        输入: user_message
        输出: intent, confidence, slots, intent_result
        """
        user_message = state.get("user_message", "")
        start_time = time.time()

        # 使用 function_calling 方法进行意图识别
        intent_result = self.classifier.classify_function_calling(user_message)

        intent = intent_result.get("intent", "UNKNOWN")
        confidence = intent_result.get("confidence", 0)
        slots = intent_result.get("slots", {})

        # 添加意图描述信息
        intent_result["intent_info"] = INTENT_DESCRIPTIONS.get(intent, INTENT_DESCRIPTIONS["UNKNOWN"])

        # 规范化产品名称
        if "product_name" in slots:
            slots["product_name"] = self._normalize_product_name(slots["product_name"])

        elapsed = time.time() - start_time

        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
            "intent_result": intent_result,
            "execution_trace": [{
                "step": 1,
                "node": "intent_recognition",
                "name": "意图识别",
                "icon": "🧠",
                "status": "completed",
                "duration_ms": round(elapsed * 1000, 1),
                "details": {
                    "intent": intent,
                    "confidence": confidence,
                    "slots": slots,
                    "method": "function_calling"
                }
            }]
        }

    # ==================== 业务处理节点 ====================

    def _create_trace(self, node: str, name: str, icon: str, details: Dict = None) -> Dict:
        """创建执行跟踪记录"""
        return {
            "step": 2,  # 业务节点是第2步
            "node": node,
            "name": name,
            "icon": icon,
            "status": "completed",
            "details": details or {}
        }

    def handle_chitchat(self, state: OrderState) -> Dict:
        """处理闲聊"""
        session_id = state.get("session_id", "")
        greetings = [
            "您好！欢迎光临，请问想喝点什么？",
            "您好！今天想来杯什么咖啡呢？",
            "欢迎光临！我们有多款特色饮品，需要我推荐一下吗？"
        ]

        return {
            "response": greetings[hash(session_id) % len(greetings)],
            "suggestions": ["来杯拿铁", "有什么推荐", "看看菜单"],
            "actions": [],
            "conversation_state": "taking_order",
            "execution_trace": [self._create_trace("handle_chitchat", "闲聊处理", "💬")]
        }

    def handle_new_order(self, state: OrderState) -> Dict:
        """处理新订单 (Phase 1 增强版 - 支持模糊表达和约束验证)"""
        slots = state.get("slots", {})
        current_order = state.get("current_order")
        user_message = state.get("user_message", "")

        # Phase 1: 先检查是否有俚语/口语化表达
        colloquial_intent = self._enhanced_normalizer.extract_colloquial_intent(user_message)
        if colloquial_intent:
            # 将口语化意图合并到 slots（优先使用俚语识别的产品名）
            for key, value in colloquial_intent.items():
                if value:  # 只有有值的才合并
                    if key == "product_name":
                        # 俚语产品名优先级高
                        slots[key] = value
                    elif key not in slots or not slots[key]:
                        slots[key] = value

        # 规范化产品名称
        raw_product_name = slots.get("product_name", "")
        product_name = self._normalize_product_name(raw_product_name)

        # Phase 1: 如果没有从slots获取到产品名，尝试从俚语表达直接提取
        if not product_name or product_name not in self._menu:
            fuzzy_matches = self._fuzzy_matcher.match(user_message)
            for fm in fuzzy_matches:
                if fm.extra_mappings.get("product_name"):
                    product_name = fm.extra_mappings["product_name"]
                    # 同时合并其他映射
                    for k, v in fm.extra_mappings.items():
                        if k != "product_name" and v and k not in slots:
                            slots[k] = v
                    break

        if not product_name or product_name not in self._menu:
            return {
                "response": "请问您想点什么饮品呢？我们有拿铁、美式、卡布奇诺、摩卡等。",
                "suggestions": ["拿铁", "美式咖啡", "卡布奇诺", "有什么推荐"],
                "actions": [],
                "conversation_state": "taking_order",
                "execution_trace": [self._create_trace("handle_new_order", "创建订单", "🛒", {"status": "need_product"})]
            }

        # 更新slots中的产品名
        slots["product_name"] = product_name

        # Phase 1: 使用增强的配置化管理器创建订单项（支持模糊表达）
        item, rule_warnings = self.item_manager.create_item(slots, user_message)

        # 创建或更新订单
        if not current_order:
            current_order = OrderDict(
                order_id=f"ORD{int(time.time()) % 100000:05d}",
                items=[],
                total=0,
                status="pending",
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        # 添加订单项
        items = list(current_order.get("items", []))
        items.append(item)
        current_order["items"] = items
        current_order["total"] = sum(i.get("price", 0) for i in items)

        # 生成回复
        reply = f"好的，已添加 {item['description']}（¥{item['price']:.0f}）。"

        # Phase 1: 添加规则引擎的警告信息
        if rule_warnings:
            reply += f"\n💡 提示: {'; '.join(rule_warnings)}"

        if len(items) > 1:
            reply += f"\n当前订单共 {len(items)} 件商品，合计 ¥{current_order['total']:.0f}。"
        reply += "\n\n请问还需要别的吗？或者确认下单？"

        return {
            "response": reply,
            "suggestions": ["确认下单", "再来一杯", "换成大杯", "取消订单"],
            "actions": ["confirm_order", "add_item", "modify", "cancel"],
            "current_order": current_order,
            "conversation_state": "confirming",
            "execution_trace": [self._create_trace("handle_new_order", "创建订单", "🛒", {
                "action": "item_added",
                "item": item.get("description"),
                "price": item.get("price"),
                "rule_warnings": rule_warnings,
                "phase1_enhanced": True
            })]
        }

    def handle_modify_order(self, state: OrderState) -> Dict:
        """处理订单修改 (Phase 1 增强版 - 支持模糊表达和约束验证)"""
        slots = state.get("slots", {})
        current_order = state.get("current_order")
        user_message = state.get("user_message", "")

        if not current_order or not current_order.get("items"):
            return {
                "response": "您还没有点单哦，请先告诉我您想喝什么？",
                "suggestions": ["来杯拿铁", "看看菜单", "有什么推荐"],
                "actions": [],
                "conversation_state": "taking_order",
                "execution_trace": [self._create_trace("handle_modify_order", "修改订单", "✏️", {"status": "no_order"})]
            }

        # Phase 1: 使用模糊表达匹配器预处理用户消息
        fuzzy_matches = self._fuzzy_matcher.match(user_message)
        for fm in fuzzy_matches:
            if fm.value and fm.slot_name in ["sweetness", "temperature"]:
                actual_slot = fm.slot_name
                if actual_slot not in slots:
                    slots[actual_slot] = fm.value

        # 修改最后一个订单项
        items = list(current_order["items"])
        last_item = items[-1]

        # 使用配置化管理器更新订单项
        updated_item, modified = self.item_manager.update_item(last_item, slots)

        # Phase 1: 验证修改后的组合是否合法
        rule_warnings = []
        if modified:
            product_name = updated_item.get("product_name", "")
            validation = self._rules_engine.validate_and_adjust(product_name, updated_item)
            if validation.warnings:
                rule_warnings = validation.warnings
                # 应用约束调整
                for key, value in validation.adjusted_slots.items():
                    if key in updated_item:
                        updated_item[key] = value

            items[-1] = updated_item
            current_order["items"] = items
            current_order["total"] = sum(i.get("price", 0) for i in items)

            reply = f"好的，已为您{', '.join(modified)}。\n"
            if rule_warnings:
                reply += f"💡 提示: {'; '.join(rule_warnings)}\n"
            reply += f"现在是：{updated_item['description']}（¥{updated_item['price']:.0f}）\n"
            reply += "还需要其他调整吗？"
        else:
            reply = "请问您想修改什么呢？可以换杯型、温度、甜度或奶类。"

        return {
            "response": reply,
            "suggestions": ["确认下单", "换成冰的", "少糖", "加燕麦奶"],
            "actions": ["confirm_order", "modify"],
            "current_order": current_order,
            "conversation_state": "modifying",
            "execution_trace": [self._create_trace("handle_modify_order", "修改订单", "✏️", {
                "modified": modified,
                "new_description": updated_item.get("description") if modified else None,
                "rule_warnings": rule_warnings,
                "phase1_enhanced": True
            })]
        }

    def handle_cancel_order(self, state: OrderState) -> Dict:
        """处理取消订单"""
        current_order = state.get("current_order")

        if not current_order or not current_order.get("items"):
            return {
                "response": "您还没有点单哦，不需要取消。请问想喝点什么？",
                "suggestions": ["来杯拿铁", "有什么推荐"],
                "actions": [],
                "conversation_state": "taking_order",
                "execution_trace": [self._create_trace("handle_cancel_order", "取消订单", "❌", {"status": "no_order"})]
            }

        items_count = len(current_order.get("items", []))

        return {
            "response": f"好的，已为您取消订单（共{items_count}件商品）。\n有需要随时再点哦！",
            "suggestions": ["重新点单", "看看菜单"],
            "actions": [],
            "current_order": None,
            "conversation_state": "taking_order",
            "execution_trace": [self._create_trace("handle_cancel_order", "取消订单", "❌", {"cancelled_items": items_count})]
        }

    def handle_query_order(self, state: OrderState) -> Dict:
        """处理订单查询 - 集成预估时间技能"""
        current_order = state.get("current_order")
        user_message = state.get("user_message", "")

        if not current_order:
            return {
                "response": "您目前没有进行中的订单。请问需要点单吗？",
                "suggestions": ["我要点单", "看看菜单"],
                "actions": [],
                "conversation_state": "taking_order",
                "execution_trace": [self._create_trace("handle_query_order", "查询订单", "🔍", {"status": "no_order"})]
            }

        # 检测是否询问等待时间
        time_keywords = ["多久", "等多长时间", "什么时候好", "几分钟", "多长时间"]
        if any(kw in user_message for kw in time_keywords):
            # 使用预估时间技能
            order_items = [item.get("product_name", "") for item in current_order.get("items", [])]
            skill_result = self.skill_executor.execute("estimate_time", {
                "order_items": order_items,
                "store_id": "SH001"
            })
            if skill_result and skill_result.success:
                return {
                    "response": skill_result.message,
                    "suggestions": ["查看订单详情", "修改订单"],
                    "actions": [],
                    "skill_result": skill_result.to_dict(),
                    "execution_trace": [
                        self._create_trace("handle_query_order", "查询订单", "🔍"),
                        self._create_trace("skill_execution", "技能: 预估时间", "⏱️", skill_result.to_dict())
                    ]
                }

        status_text = {
            "pending": "待确认",
            "confirmed": "已确认，准备制作",
            "preparing": "制作中",
            "ready": "已完成，请取餐",
            "completed": "已完成",
            "cancelled": "已取消"
        }

        items_text = "\n".join([
            f"  • {item['description']} ¥{item['price']:.0f}"
            for item in current_order.get("items", [])
        ])

        reply = f"您的订单 {current_order['order_id']}：\n{items_text}\n\n"
        reply += f"合计：¥{current_order['total']:.0f}\n"
        reply += f"状态：{status_text.get(current_order.get('status', 'pending'), current_order.get('status'))}"

        return {
            "response": reply,
            "suggestions": ["确认下单", "修改订单", "取消订单"],
            "actions": ["confirm_order", "modify", "cancel"],
            "conversation_state": "confirming",
            "execution_trace": [self._create_trace("handle_query_order", "查询订单", "🔍", {
                "order_id": current_order.get("order_id"),
                "status": current_order.get("status")
            })]
        }

    def handle_product_info(self, state: OrderState) -> Dict:
        """处理商品信息查询 - 集成技能执行"""
        slots = state.get("slots", {})
        user_message = state.get("user_message", "")
        intent = state.get("intent", "")
        product_name = self._normalize_product_name(slots.get("product_name"))

        # 尝试匹配技能 (营养查询、库存查询等)
        skill_result = self._try_execute_skill(user_message, intent, slots)
        if skill_result and skill_result.success:
            return {
                "response": skill_result.message,
                "suggestions": [f"来杯{product_name}" if product_name else "来杯拿铁", "看看其他", "有什么推荐"],
                "actions": [],
                "skill_result": skill_result.to_dict(),
                "execution_trace": [
                    self._create_trace("handle_product_info", "商品咨询", "ℹ️"),
                    self._create_trace("skill_execution", f"技能: {skill_result.skill_id}", "🔧", skill_result.to_dict())
                ]
            }

        if product_name and product_name in self._menu:
            info = self._menu[product_name]
            reply = f"【{product_name}】\n"
            reply += f"价格：¥{info['price']}（中杯）\n"
            reply += f"热量：{info['calories']} 大卡\n"
            reply += f"介绍：{info['desc']}\n\n"
            reply += "需要来一杯吗？"
            suggestions = [f"来杯{product_name}", "看看其他", "有什么推荐"]
        else:
            # 从配置获取价格增量
            price_deltas = self.registry.get_price_deltas()
            size_prices = price_deltas.get("size", {})
            milk_prices = price_deltas.get("milk_type", {})

            reply = "我们的菜单：\n\n"
            for name, info in self._menu.items():
                reply += f"• {name}  ¥{info['price']}\n"

            reply += f"\n升杯：大杯+{size_prices.get('大杯', 4)}元，超大杯+{size_prices.get('超大杯', 7)}元\n"
            reply += f"换奶：燕麦奶/椰奶+{milk_prices.get('燕麦奶', 6)}元\n\n请问想喝什么？"
            suggestions = ["拿铁", "美式咖啡", "推荐一下"]

        return {
            "response": reply,
            "suggestions": suggestions,
            "actions": [],
            "execution_trace": [self._create_trace("handle_product_info", "商品咨询", "ℹ️", {"product": product_name})]
        }

    def handle_recommend(self, state: OrderState) -> Dict:
        """处理推荐请求 - 使用智能推荐技能"""
        user_message = state.get("user_message", "")
        intent = state.get("intent", "")
        slots = state.get("slots", {})

        # 使用智能推荐技能
        skill_result = self._try_execute_skill(user_message, intent, slots)
        if skill_result and skill_result.success:
            top_pick = skill_result.data.get("top_pick", "拿铁")
            recommendations = skill_result.data.get("recommendations", [])

            # 生成建议按钮
            suggestions = [f"来杯{r.get('product_name', '拿铁')}" for r in recommendations[:3]]

            return {
                "response": skill_result.message + "\n\n请问想试试哪一款？",
                "suggestions": suggestions,
                "actions": [],
                "skill_result": skill_result.to_dict(),
                "execution_trace": [
                    self._create_trace("handle_recommend", "智能推荐", "⭐"),
                    self._create_trace("skill_execution", "技能: 智能推荐", "🎯", {
                        "top_pick": top_pick,
                        "count": len(recommendations)
                    })
                ]
            }

        # 回退到默认推荐
        recommendations = [
            ("拿铁", "最受欢迎的经典选择，奶香与咖啡香完美平衡"),
            ("馥芮白", "澳洲风味，比拿铁更浓郁顺滑"),
            ("美式咖啡", "低卡之选，适合注重健康的您"),
        ]

        reply = "为您推荐：\n\n"
        for name, reason in recommendations:
            if name in self._menu:
                info = self._menu[name]
                reply += f"⭐ {name}（¥{info['price']}）\n   {reason}\n\n"
        reply += "请问想试试哪一款？"

        return {
            "response": reply,
            "suggestions": ["来杯拿铁", "来杯馥芮白", "来杯美式"],
            "actions": [],
            "execution_trace": [self._create_trace("handle_recommend", "智能推荐", "⭐", {"fallback": True})]
        }

    def handle_payment(self, state: OrderState) -> Dict:
        """处理支付/确认订单"""
        current_order = state.get("current_order")

        if not current_order or not current_order.get("items"):
            return {
                "response": "您还没有点单哦，请先告诉我您想喝什么？",
                "suggestions": ["来杯拿铁", "看看菜单"],
                "actions": [],
                "conversation_state": "taking_order",
                "execution_trace": [self._create_trace("handle_payment", "确认订单", "💳", {"status": "no_order"})]
            }

        # 确认订单
        current_order["status"] = "confirmed"
        total = current_order["total"]

        items_text = "\n".join([
            f"  • {item['description']} ¥{item['price']:.0f}"
            for item in current_order.get("items", [])
        ])

        reply = f"订单已确认！\n\n"
        reply += f"订单号：{current_order['order_id']}\n"
        reply += f"{items_text}\n"
        reply += f"──────────\n"
        reply += f"合计：¥{total:.0f}\n\n"
        reply += "支持微信、支付宝、Apple Pay\n"
        reply += "请稍候，您的饮品马上就好！"

        # 更新状态为制作中
        current_order["status"] = "preparing"

        return {
            "response": reply,
            "suggestions": ["再点一单", "查看订单"],
            "actions": ["new_order", "query_order"],
            "current_order": current_order,
            "conversation_state": "completed",
            "should_end": True,
            "execution_trace": [self._create_trace("handle_payment", "确认订单", "💳", {
                "order_id": current_order.get("order_id"),
                "total": total,
                "items_count": len(current_order.get("items", []))
            })]
        }

    def handle_complaint(self, state: OrderState) -> Dict:
        """处理投诉"""
        return {
            "response": "非常抱歉给您带来不好的体验！我已经记录下您的反馈，会尽快为您处理。\n\n请问具体是什么问题呢？我们会尽力改进。",
            "suggestions": ["重新制作", "取消订单", "联系经理"],
            "actions": [],
            "execution_trace": [self._create_trace("handle_complaint", "投诉处理", "😤")]
        }

    def handle_unknown(self, state: OrderState) -> Dict:
        """处理未知意图"""
        return {
            "response": "抱歉，我没有理解您的意思。您可以说'我要点一杯拿铁'来点单，或者问我'有什么推荐的'。",
            "suggestions": ["我要点杯拿铁", "有什么推荐的", "菜单有什么"],
            "actions": [],
            "execution_trace": [self._create_trace("handle_unknown", "未知意图", "❓")]
        }

    # ==================== 消息记录节点 ====================

    def record_message(self, state: OrderState) -> Dict:
        """记录对话消息"""
        user_message = state.get("user_message", "")
        response = state.get("response", "")
        intent = state.get("intent", "")
        confidence = state.get("confidence", 0)
        slots = state.get("slots", {})

        timestamp = datetime.now().strftime("%H:%M:%S")

        messages = [
            MessageDict(
                role="user",
                content=user_message,
                timestamp=timestamp,
                intent_info=None
            ),
            MessageDict(
                role="assistant",
                content=response,
                timestamp=timestamp,
                intent_info={
                    "intent": intent,
                    "confidence": confidence,
                    "slots": slots
                }
            )
        ]

        return {"messages": messages}


# ==================== 路由函数 ====================

def route_by_intent(state: OrderState) -> str:
    """根据意图路由到对应的处理节点"""
    intent = state.get("intent", "UNKNOWN")
    current_order = state.get("current_order")
    slots = state.get("slots", {})
    user_message = state.get("user_message", "").lower()

    # 获取配置
    registry = get_schema_registry()
    menu = registry.get_menu_dict()

    # 智能路由：基于上下文修正意图

    # 情况1: 用户说"确认下单"/"下单"/"结账"等，但被误识别为ORDER_NEW
    # 使用配置中的PAYMENT意图关键词
    payment_intent = registry.get_intent("PAYMENT")
    confirm_keywords = payment_intent.keywords if payment_intent else ["确认", "下单", "结账", "买单"]
    if intent == "ORDER_NEW" and any(kw in user_message for kw in confirm_keywords):
        if current_order and current_order.get("items"):
            if not slots.get("product_name"):
                return "handle_payment"

    # 情况2: 用户说"加XXX"，但被误识别为ORDER_NEW
    add_keywords = ["加一", "加份", "加个", "多加", "再加"]
    if intent == "ORDER_NEW" and any(kw in user_message for kw in add_keywords):
        if current_order and current_order.get("items"):
            # 检查是否是加配料而非加商品
            product_name = slots.get("product_name", "")
            if product_name and product_name not in menu:
                return "handle_modify_order"
            # 使用配置检测配料关键词
            extras_from_text = registry.extract_extras_from_text(user_message)
            if extras_from_text:
                return "handle_modify_order"

    # 情况3: 没有产品名但有修改属性的，应该是修改订单
    if intent == "ORDER_NEW":
        if current_order and current_order.get("items"):
            if not slots.get("product_name"):
                # 检查是否有其他修改属性
                if any(key in slots for key in ["size", "temperature", "sweetness", "milk_type", "extras"]):
                    return "handle_modify_order"

    routing = {
        "CHITCHAT": "handle_chitchat",
        "ORDER_NEW": "handle_new_order",
        "ORDER_MODIFY": "handle_modify_order",
        "CUSTOMIZE": "handle_modify_order",
        "ORDER_CANCEL": "handle_cancel_order",
        "ORDER_QUERY": "handle_query_order",
        "PRODUCT_INFO": "handle_product_info",
        "RECOMMEND": "handle_recommend",
        "PAYMENT": "handle_payment",
        "COMPLAINT": "handle_complaint",
    }

    return routing.get(intent, "handle_unknown")


# ==================== 工作流构建 ====================

class OrderingWorkflow:
    """
    AI点单对话工作流 - 支持配置化和数据库持久化

    使用 LangGraph 实现的状态机:

    [用户输入] → [意图识别] → [路由] → [业务处理] → [记录消息] → [响应]
                                  ↓
                     ┌────────────┼────────────┐
                     ↓            ↓            ↓
                [新订单]    [修改订单]    [其他处理...]

    特性:
    - 支持YAML配置化槽位定义
    - 自动槽位值规范化
    - 从配置读取菜单和价格
    - SQLite 数据库持久化会话和订单
    """

    def __init__(self, classifier: Optional[OpenAIClassifier] = None,
                 schema_registry: Optional[SlotSchemaRegistry] = None,
                 use_db: bool = True):
        if classifier is None:
            classifier = OpenAIClassifier()

        self.classifier = classifier
        self.registry = schema_registry or get_schema_registry()
        self.nodes = WorkflowNodes(classifier, self.registry)
        self.graph = self._build_graph()

        # 使用内存检查点保存会话状态
        self.checkpointer = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

        # 监听配置变更
        self.registry.on_change(self._on_schema_change)

        # 结构化日志和监控
        self._logger = get_structured_logger("workflow")
        self._metrics = get_metrics_collector()

        # 数据库持久化
        self.use_db = use_db
        if use_db:
            try:
                self._db = Database()
                self._session_repo = SessionRepository(self._db)
                self._order_repo = OrderRepository(self._db)
                self._message_repo = MessageRepository(self._db)
                self._logger.info("workflow_init", details="工作流已启用数据库持久化")
            except Exception as e:
                self._logger.warning("workflow_init", details=f"数据库初始化失败，仅使用内存模式: {e}")
                self.use_db = False

    def _on_schema_change(self, registry: SlotSchemaRegistry):
        """配置变更回调"""
        print(f"📝 Schema配置已更新: v{registry.version}")
        # 重新初始化节点
        self.nodes = WorkflowNodes(self.classifier, registry)

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流图"""
        workflow = StateGraph(OrderState)

        # 添加节点
        workflow.add_node("intent_recognition", self.nodes.intent_recognition)
        workflow.add_node("handle_chitchat", self.nodes.handle_chitchat)
        workflow.add_node("handle_new_order", self.nodes.handle_new_order)
        workflow.add_node("handle_modify_order", self.nodes.handle_modify_order)
        workflow.add_node("handle_cancel_order", self.nodes.handle_cancel_order)
        workflow.add_node("handle_query_order", self.nodes.handle_query_order)
        workflow.add_node("handle_product_info", self.nodes.handle_product_info)
        workflow.add_node("handle_recommend", self.nodes.handle_recommend)
        workflow.add_node("handle_payment", self.nodes.handle_payment)
        workflow.add_node("handle_complaint", self.nodes.handle_complaint)
        workflow.add_node("handle_unknown", self.nodes.handle_unknown)
        workflow.add_node("record_message", self.nodes.record_message)

        # 设置入口点
        workflow.set_entry_point("intent_recognition")

        # 添加条件边 - 根据意图路由
        workflow.add_conditional_edges(
            "intent_recognition",
            route_by_intent,
            {
                "handle_chitchat": "handle_chitchat",
                "handle_new_order": "handle_new_order",
                "handle_modify_order": "handle_modify_order",
                "handle_cancel_order": "handle_cancel_order",
                "handle_query_order": "handle_query_order",
                "handle_product_info": "handle_product_info",
                "handle_recommend": "handle_recommend",
                "handle_payment": "handle_payment",
                "handle_complaint": "handle_complaint",
                "handle_unknown": "handle_unknown",
            }
        )

        # 所有处理节点都流向消息记录节点
        for node in [
            "handle_chitchat", "handle_new_order", "handle_modify_order",
            "handle_cancel_order", "handle_query_order", "handle_product_info",
            "handle_recommend", "handle_payment", "handle_complaint", "handle_unknown"
        ]:
            workflow.add_edge(node, "record_message")

        # 消息记录后结束
        workflow.add_edge("record_message", END)

        return workflow

    @monitor_performance("workflow.process_message")
    def process_message(self, session_id: Optional[str], user_message: str) -> Dict:
        """
        处理用户消息

        Args:
            session_id: 会话ID，如果为空则创建新会话
            user_message: 用户消息

        Returns:
            包含响应、订单状态等信息的字典
        """
        start_time = time.time()

        # 生成或使用会话ID
        is_new_session = not session_id
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        self._logger.info("process_message_start",
                          session_id=session_id,
                          is_new_session=is_new_session,
                          message_length=len(user_message))

        # 数据库：确保会话存在
        if self.use_db and is_new_session:
            try:
                self._session_repo.create(session_id)
            except Exception as e:
                self._logger.error("session_create_failed", session_id=session_id, error=str(e))

        # 配置线程ID用于状态持久化
        config = {"configurable": {"thread_id": session_id}}

        # 获取当前状态
        try:
            current_state = self.app.get_state(config)
            existing_order = current_state.values.get("current_order") if current_state.values else None
            existing_messages = current_state.values.get("messages", []) if current_state.values else []
            conversation_state = current_state.values.get("conversation_state", "greeting") if current_state.values else "greeting"
        except Exception:
            existing_order = None
            existing_messages = []
            conversation_state = "greeting"

        # 构建输入状态
        input_state = OrderState(
            session_id=session_id,
            user_message=user_message,
            current_order=existing_order,
            messages=[],  # 新消息将被添加
            conversation_state=conversation_state,
            execution_trace=[]  # 初始化执行跟踪
        )

        # 执行工作流
        result = self.app.invoke(input_state, config)

        # 数据库：持久化消息
        if self.use_db:
            try:
                # 保存用户消息
                intent_result = result.get("intent_result", {})
                self._message_repo.add(MessageModel(
                    session_id=session_id,
                    role="user",
                    content=user_message,
                    intent=intent_result.get("intent"),
                    confidence=intent_result.get("confidence"),
                    slots=intent_result.get("slots")
                ))

                # 保存助手回复
                self._message_repo.add(MessageModel(
                    session_id=session_id,
                    role="assistant",
                    content=result.get("response", "")
                ))

                # 更新会话状态
                db_session = self._session_repo.get(session_id)
                if db_session:
                    db_session.state = result.get("conversation_state", "taking_order")
                    order = result.get("current_order")
                    if order:
                        db_session.current_order_id = order.get("order_id")
                    self._session_repo.update(db_session)

                # 持久化订单
                order = result.get("current_order")
                if order and order.get("order_id"):
                    self._persist_order(session_id, order)

            except Exception as e:
                self._logger.error("persist_failed", session_id=session_id, error=str(e))

        # 记录处理完成
        elapsed_time = time.time() - start_time
        intent_result = result.get("intent_result", {})
        self._logger.info("process_message_complete",
                          session_id=session_id,
                          intent=intent_result.get("intent"),
                          confidence=intent_result.get("confidence"),
                          elapsed_ms=round(elapsed_time * 1000, 2))

        # 记录指标
        self._metrics.record_request(
            endpoint="workflow.process_message",
            method="POST",
            status_code=200,
            duration=elapsed_time
        )

        # 构建返回结果
        return {
            "session_id": session_id,
            "state": result.get("conversation_state", "taking_order"),
            "reply": result.get("response", ""),
            "intent_result": result.get("intent_result", {}),
            "order": result.get("current_order"),
            "history": existing_messages + result.get("messages", []),
            "suggestions": result.get("suggestions", []),
            "actions": result.get("actions", []),
            "skill_result": result.get("skill_result"),
            "execution_trace": result.get("execution_trace", [])
        }

    def _persist_order(self, session_id: str, order: Dict):
        """持久化订单到数据库"""
        if not self.use_db:
            return

        try:
            order_id = order.get("order_id")

            # 检查订单是否已存在
            existing = self._order_repo.get(order_id)
            if not existing:
                # 创建新订单
                order_model = self._order_repo.create(order_id, session_id)
            else:
                order_model = existing

            # 更新订单状态和总价
            order_model.status = order.get("status", "pending")
            order_model.total = order.get("total", 0.0)
            self._order_repo.update(order_model)

            # 同步订单项（简化处理：先删除再添加）
            existing_items = self._order_repo.get_items(order_id)
            for item in existing_items:
                self._order_repo.delete_item(item.id)

            for item_dict in order.get("items", []):
                item_model = OrderItemModel(
                    order_id=order_id,
                    product_name=item_dict.get("product_name", ""),
                    size=item_dict.get("size", "中杯"),
                    temperature=item_dict.get("temperature", "热"),
                    sweetness=item_dict.get("sweetness", "标准"),
                    milk_type=item_dict.get("milk_type", "全脂奶"),
                    extras=item_dict.get("extras", []),
                    quantity=item_dict.get("quantity", 1),
                    price=item_dict.get("price", 0.0)
                )
                self._order_repo.add_item(item_model)

        except Exception as e:
            logger.error(f"订单持久化失败: {e}")

    def create_session(self) -> Dict:
        """创建新会话"""
        session_id = str(uuid.uuid4())[:8]

        # 数据库：持久化会话
        if self.use_db:
            try:
                self._session_repo.create(session_id)
                logger.debug(f"会话已持久化: {session_id}")
            except Exception as e:
                logger.error(f"会话持久化失败: {e}")

        welcome_message = MessageDict(
            role="assistant",
            content="您好！欢迎光临，请问想喝点什么？",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            intent_info=None
        )

        return {
            "session_id": session_id,
            "state": "greeting",
            "history": [welcome_message],
            "suggestions": ["来杯拿铁", "有什么推荐", "看看菜单"]
        }

    def reset_session(self, session_id: str) -> Dict:
        """重置会话"""
        # 创建新的会话ID
        new_session_id = str(uuid.uuid4())[:8]

        welcome_message = MessageDict(
            role="assistant",
            content="您好！欢迎光临，请问想喝点什么？",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            intent_info=None
        )

        return {
            "session_id": new_session_id,
            "state": "greeting",
            "history": [welcome_message],
            "suggestions": ["来杯拿铁", "有什么推荐", "看看菜单"]
        }

    def get_graph_visualization(self) -> str:
        """获取工作流图的Mermaid表示"""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return """
graph TD
    A[用户输入] --> B[意图识别]
    B --> C{路由}
    C --> D[新订单]
    C --> E[修改订单]
    C --> F[取消订单]
    C --> G[查询订单]
    C --> H[商品信息]
    C --> I[推荐]
    C --> J[支付]
    C --> K[投诉]
    C --> L[闲聊]
    C --> M[未知]
    D --> N[记录消息]
    E --> N
    F --> N
    G --> N
    H --> N
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
    N --> O[响应]
"""


# ==================== 测试代码 ====================

def test_workflow():
    """测试工作流"""
    print("=" * 60)
    print("LangGraph 工作流测试")
    print("=" * 60)

    # 创建工作流
    workflow = OrderingWorkflow()

    # 测试用例
    test_cases = [
        "你好",
        "来杯大杯冰拿铁",
        "换成燕麦奶",
        "加一份浓缩",
        "确认下单",
    ]

    session_id = None

    for i, message in enumerate(test_cases):
        print(f"\n[{i+1}] 用户: {message}")
        result = workflow.process_message(session_id, message)
        session_id = result["session_id"]

        print(f"    意图: {result['intent_result'].get('intent', 'N/A')} "
              f"(置信度: {result['intent_result'].get('confidence', 0):.2f})")
        print(f"    回复: {result['reply'][:80]}...")

        if result.get("order"):
            print(f"    订单: {result['order'].get('order_id')} - "
                  f"¥{result['order'].get('total', 0):.0f}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

    # 打印工作流图
    print("\n工作流图 (Mermaid):")
    print(workflow.get_graph_visualization())


if __name__ == "__main__":
    test_workflow()
