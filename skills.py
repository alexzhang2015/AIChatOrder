"""
Skills 执行层模块

提供类似 Claude Skill 的可配置执行能力:
1. SkillDefinition - 技能定义
2. SkillRegistry - 技能注册中心
3. SkillExecutor - 技能执行器
4. 内置技能处理器实现
5. 测试框架
"""

import os
import re
import yaml
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod


# ==================== 数据类定义 ====================

@dataclass
class ParameterDefinition:
    """参数定义"""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)


@dataclass
class ReturnDefinition:
    """返回值定义"""
    type: str
    properties: Dict[str, Dict] = field(default_factory=dict)
    description: str = ""


@dataclass
class TestCase:
    """测试用例"""
    name: str
    input: Dict[str, Any]
    expected: Dict[str, Any]
    validate: Optional[str] = None  # 验证表达式


@dataclass
class SkillTrigger:
    """技能触发条件"""
    keywords: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)


@dataclass
class SkillDefinition:
    """技能定义"""
    id: str
    name: str
    description: str
    category: str
    enabled: bool = True
    triggers: SkillTrigger = field(default_factory=SkillTrigger)
    parameters: List[ParameterDefinition] = field(default_factory=list)
    returns: Optional[ReturnDefinition] = None
    handler: str = ""
    examples: List[Dict] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)

    def matches_intent(self, intent: str) -> bool:
        """检查是否匹配意图"""
        return intent in self.triggers.intents

    def matches_keywords(self, text: str) -> bool:
        """检查是否匹配关键词"""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.triggers.keywords)

    def matches(self, text: str, intent: Optional[str] = None) -> Tuple[bool, float]:
        """
        检查是否匹配，返回 (是否匹配, 匹配分数)
        """
        score = 0.0

        # 检查意图匹配
        if intent and self.matches_intent(intent):
            score += 0.5

        # 检查关键词匹配
        if self.matches_keywords(text):
            # 计算匹配的关键词数量
            text_lower = text.lower()
            matched_keywords = [kw for kw in self.triggers.keywords if kw.lower() in text_lower]
            score += min(0.5, len(matched_keywords) * 0.2)

        return score > 0, score

    def validate_params(self, params: Dict) -> Tuple[bool, Optional[str]]:
        """验证参数"""
        for param_def in self.parameters:
            if param_def.required and param_def.name not in params:
                return False, f"缺少必需参数: {param_def.name}"

            if param_def.name in params:
                value = params[param_def.name]

                # 枚举验证
                if param_def.enum and value not in param_def.enum:
                    return False, f"参数 {param_def.name} 值必须是: {param_def.enum}"

                # 正则验证
                if param_def.validation.get("pattern"):
                    if not re.match(param_def.validation["pattern"], str(value)):
                        return False, param_def.validation.get("message", f"参数 {param_def.name} 格式不正确")

        return True, None

    def to_function_schema(self) -> Dict:
        """转换为 Function Calling Schema"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.id,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


@dataclass
class SkillResult:
    """技能执行结果"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None
    execution_time: float = 0.0
    skill_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error": self.error,
            "execution_time": self.execution_time,
            "skill_id": self.skill_id
        }


@dataclass
class CategoryDefinition:
    """分类定义"""
    id: str
    name: str
    description: str
    icon: str = ""


# ==================== 技能处理器基类 ====================

class SkillHandler(ABC):
    """技能处理器基类"""

    @abstractmethod
    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        """执行技能"""
        pass

    def get_name(self) -> str:
        """获取处理器名称"""
        return self.__class__.__name__


# ==================== 内置技能处理器 ====================

class InventoryCheckHandler(SkillHandler):
    """库存查询处理器"""

    # 模拟库存数据
    INVENTORY = {
        "拿铁": {"available": True, "quantity": 50},
        "美式咖啡": {"available": True, "quantity": 80},
        "卡布奇诺": {"available": True, "quantity": 30},
        "摩卡": {"available": True, "quantity": 25},
        "星冰乐": {"available": True, "quantity": 20},
        "馥芮白": {"available": True, "quantity": 15},
        "抹茶拿铁": {"available": True, "quantity": 40},
        "焦糖玛奇朵": {"available": True, "quantity": 35},
        "季节限定款": {"available": False, "quantity": 0, "restock_time": "明天上午"},
    }

    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        product_name = params.get("product_name", "")

        if not product_name:
            return SkillResult(
                success=False,
                error="请指定要查询的商品名称",
                skill_id="check_inventory"
            )

        # 查找商品（支持模糊匹配）
        inventory_item = None
        for name, data in self.INVENTORY.items():
            if name in product_name or product_name in name:
                inventory_item = (name, data)
                break

        if inventory_item:
            name, data = inventory_item
            if data["available"]:
                message = f"{name}库存充足（剩余{data['quantity']}杯），随时可以为您制作！"
            else:
                restock = data.get("restock_time", "待定")
                message = f"抱歉，{name}暂时缺货，预计{restock}补货"

            return SkillResult(
                success=True,
                data={
                    "product_name": name,
                    "available": data["available"],
                    "quantity": data["quantity"],
                    "estimated_restock": data.get("restock_time")
                },
                message=message,
                skill_id="check_inventory"
            )
        else:
            return SkillResult(
                success=True,
                data={"available": False, "quantity": 0},
                message=f"抱歉，我们目前没有 {product_name} 这款商品",
                skill_id="check_inventory"
            )


class ApplyCouponHandler(SkillHandler):
    """优惠券应用处理器"""

    # 模拟优惠券数据
    COUPONS = {
        "WELCOME10": {"type": "percentage", "value": 10, "min_order": 0, "description": "新人9折"},
        "COFFEE20": {"type": "percentage", "value": 20, "min_order": 30, "description": "满30享8折"},
        "SAVE5": {"type": "fixed", "value": 5, "min_order": 25, "description": "满25减5"},
        "FREEUP": {"type": "free_upgrade", "value": 1, "min_order": 0, "description": "免费升杯"},
        "BOGO": {"type": "bogo", "value": 1, "min_order": 0, "description": "买一送一"},
    }

    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        coupon_code = params.get("coupon_code", "").upper()
        order_total = params.get("order_total", 0)

        if not coupon_code:
            return SkillResult(
                success=False,
                error="请提供优惠券码",
                skill_id="apply_coupon"
            )

        coupon = self.COUPONS.get(coupon_code)

        if not coupon:
            return SkillResult(
                success=True,
                data={"valid": False},
                message=f"优惠券码 {coupon_code} 无效或已过期",
                skill_id="apply_coupon"
            )

        # 检查最低消费
        if order_total < coupon["min_order"]:
            return SkillResult(
                success=True,
                data={"valid": False, "min_order": coupon["min_order"]},
                message=f"该优惠券需要满 ¥{coupon['min_order']} 才能使用，当前订单 ¥{order_total}",
                skill_id="apply_coupon"
            )

        # 计算折扣
        discount_type = coupon["type"]
        discount_value = coupon["value"]

        if discount_type == "percentage":
            discount_amount = order_total * discount_value / 100
            final_price = order_total - discount_amount
            message = f"优惠券已应用！{coupon['description']}，节省 ¥{discount_amount:.0f}"
        elif discount_type == "fixed":
            discount_amount = discount_value
            final_price = max(0, order_total - discount_amount)
            message = f"优惠券已应用！{coupon['description']}，节省 ¥{discount_amount}"
        elif discount_type == "free_upgrade":
            discount_amount = 4  # 升杯差价
            final_price = order_total
            message = f"优惠券已应用！{coupon['description']}，免费升杯"
        else:  # bogo
            discount_amount = order_total / 2
            final_price = order_total / 2
            message = f"优惠券已应用！{coupon['description']}，第二杯免费"

        return SkillResult(
            success=True,
            data={
                "valid": True,
                "discount_type": discount_type,
                "discount_value": discount_value,
                "discount_amount": discount_amount,
                "final_price": final_price,
                "coupon_description": coupon["description"]
            },
            message=message,
            skill_id="apply_coupon"
        )


class CheckPointsHandler(SkillHandler):
    """积分查询处理器"""

    # 模拟会员数据
    MEMBERS = {
        "M001": {"points": 350, "level": "金星会员", "name": "张三"},
        "M002": {"points": 1200, "level": "钻石会员", "name": "李四"},
        "DEFAULT": {"points": 50, "level": "绿星会员", "name": "顾客"},
    }

    REWARDS = [
        {"points": 100, "reward": "中杯饮品兑换券"},
        {"points": 200, "reward": "大杯饮品兑换券"},
        {"points": 300, "reward": "任意饮品兑换券"},
        {"points": 500, "reward": "买一送一券"},
    ]

    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        member_id = params.get("member_id", "DEFAULT")

        member = self.MEMBERS.get(member_id, self.MEMBERS["DEFAULT"])
        points = member["points"]
        level = member["level"]

        # 计算可兑换奖励
        available_rewards = [r for r in self.REWARDS if r["points"] <= points]

        # 计算下一等级需要的积分
        level_thresholds = {"绿星会员": 200, "金星会员": 500, "钻石会员": 1500}
        next_level_points = level_thresholds.get(level, 200)

        message = f"您当前是{level}，拥有{points}积分。"
        if available_rewards:
            message += f"可兑换: {available_rewards[-1]['reward']}"

        return SkillResult(
            success=True,
            data={
                "points": points,
                "level": level,
                "available_rewards": available_rewards,
                "next_level_points": next_level_points,
                "member_name": member["name"]
            },
            message=message,
            skill_id="check_points"
        )


class SmartRecommendHandler(SkillHandler):
    """智能推荐处理器"""

    RECOMMENDATIONS = {
        "hot": [
            {"product_name": "冰美式", "reason": "清爽提神，适合炎热天气", "match_score": 0.95},
            {"product_name": "星冰乐", "reason": "冰爽甜蜜，消暑首选", "match_score": 0.90},
            {"product_name": "冰拿铁", "reason": "经典冰咖啡，醇厚顺滑", "match_score": 0.85},
        ],
        "cold": [
            {"product_name": "热拿铁", "reason": "温暖身心，冬日必备", "match_score": 0.95},
            {"product_name": "热摩卡", "reason": "巧克力与咖啡的温暖组合", "match_score": 0.90},
            {"product_name": "热卡布奇诺", "reason": "绵密奶泡，暖心之选", "match_score": 0.85},
        ],
        "diet": [
            {"product_name": "美式咖啡", "reason": "仅15大卡，无负担提神", "match_score": 0.98},
            {"product_name": "冷萃咖啡", "reason": "低卡路里，口感顺滑", "match_score": 0.90},
        ],
        "default": [
            {"product_name": "拿铁", "reason": "经典畅销款，奶香与咖啡完美平衡", "match_score": 0.95},
            {"product_name": "馥芮白", "reason": "澳洲风味，浓郁丝滑", "match_score": 0.88},
            {"product_name": "卡布奇诺", "reason": "传统意式，奶泡绵密", "match_score": 0.85},
        ]
    }

    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        weather = params.get("weather", "normal")
        preference = params.get("preference", "")
        time_of_day = params.get("time_of_day", "afternoon")

        # 根据条件选择推荐
        if weather == "hot":
            recommendations = self.RECOMMENDATIONS["hot"]
            reason = "天气炎热"
        elif weather == "cold":
            recommendations = self.RECOMMENDATIONS["cold"]
            reason = "天气寒冷"
        elif preference and ("减肥" in preference or "低卡" in preference or "健康" in preference):
            recommendations = self.RECOMMENDATIONS["diet"]
            reason = "注重健康"
        else:
            recommendations = self.RECOMMENDATIONS["default"]
            reason = "经典推荐"

        top_pick = recommendations[0]["product_name"]

        message = f"根据{reason}，为您推荐：\n"
        for rec in recommendations[:3]:
            message += f"• {rec['product_name']} - {rec['reason']}\n"

        return SkillResult(
            success=True,
            data={
                "recommendations": recommendations,
                "top_pick": top_pick,
                "reason": reason
            },
            message=message,
            skill_id="smart_recommend"
        )


class EstimateTimeHandler(SkillHandler):
    """预估时间处理器"""

    # 每种饮品的基础制作时间（分钟）
    PREP_TIME = {
        "拿铁": 3,
        "美式咖啡": 2,
        "卡布奇诺": 3,
        "摩卡": 3,
        "星冰乐": 3,
        "馥芮白": 2,
        "抹茶拿铁": 3,
        "焦糖玛奇朵": 3,
    }

    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        order_items = params.get("order_items", [])
        test_mode = params.get("_test_mode", False)

        if not order_items:
            return SkillResult(
                success=False,
                error="请提供订单商品列表",
                skill_id="estimate_time"
            )

        # 计算总时间（多品有并行制作优化）
        total_time = 0
        for item in order_items:
            item_name = item if isinstance(item, str) else item.get("product_name", "")
            prep = self.PREP_TIME.get(item_name, 2)
            total_time += prep

        # 多品并行优化：减少20%时间
        if len(order_items) > 1:
            total_time = round(total_time * 0.8)

        # 排队情况（测试模式使用固定值）
        if test_mode:
            queue_position = 0
            queue_time = 0
        else:
            queue_position = random.randint(2, 5)
            queue_time = queue_position * 1.5

        estimated_minutes = round(total_time + queue_time)

        if queue_position == 0:
            message = f"马上开始制作，预计{estimated_minutes}分钟后可取"
        else:
            message = f"预计{estimated_minutes}分钟后可取，您前面还有{queue_position}位顾客"

        return SkillResult(
            success=True,
            data={
                "estimated_minutes": estimated_minutes,
                "prep_time": round(total_time),
                "queue_position": queue_position,
                "queue_time": round(queue_time)
            },
            message=message,
            skill_id="estimate_time"
        )


class NutritionInfoHandler(SkillHandler):
    """营养信息处理器"""

    NUTRITION = {
        "拿铁": {"calories": 190, "caffeine": 150, "sugar": 18, "protein": 10, "fat": 7},
        "美式咖啡": {"calories": 15, "caffeine": 225, "sugar": 0, "protein": 1, "fat": 0},
        "卡布奇诺": {"calories": 120, "caffeine": 150, "sugar": 10, "protein": 8, "fat": 5},
        "摩卡": {"calories": 290, "caffeine": 175, "sugar": 35, "protein": 10, "fat": 12},
        "星冰乐": {"calories": 350, "caffeine": 95, "sugar": 50, "protein": 5, "fat": 15},
        "馥芮白": {"calories": 140, "caffeine": 195, "sugar": 12, "protein": 9, "fat": 6},
        "抹茶拿铁": {"calories": 240, "caffeine": 80, "sugar": 28, "protein": 12, "fat": 7},
        "焦糖玛奇朵": {"calories": 250, "caffeine": 150, "sugar": 33, "protein": 10, "fat": 8},
    }

    SIZE_MULTIPLIER = {"中杯": 1.0, "大杯": 1.3, "超大杯": 1.6}
    MILK_CALORIES = {"全脂奶": 0, "脱脂奶": -30, "燕麦奶": -20, "椰奶": -10, "豆奶": -25}

    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        product_name = params.get("product_name", "")
        size = params.get("size", "中杯")
        milk_type = params.get("milk_type", "全脂奶")

        # 查找商品
        nutrition = None
        found_name = None
        for name in self.NUTRITION:
            if name in product_name or product_name in name:
                nutrition = self.NUTRITION[name].copy()
                found_name = name
                break

        if not nutrition:
            return SkillResult(
                success=False,
                error=f"未找到 {product_name} 的营养信息",
                skill_id="nutrition_info"
            )

        # 调整杯型
        multiplier = self.SIZE_MULTIPLIER.get(size, 1.0)
        for key in ["calories", "sugar", "protein", "fat"]:
            nutrition[key] = round(nutrition[key] * multiplier)

        # 调整奶类
        nutrition["calories"] += self.MILK_CALORIES.get(milk_type, 0)

        message = f"{size}{found_name}（{milk_type}）营养信息：\n"
        message += f"热量: {nutrition['calories']}大卡 | 咖啡因: {nutrition['caffeine']}mg\n"
        message += f"糖: {nutrition['sugar']}g | 蛋白质: {nutrition['protein']}g | 脂肪: {nutrition['fat']}g"

        return SkillResult(
            success=True,
            data={
                "product_name": found_name,
                "size": size,
                "milk_type": milk_type,
                **nutrition
            },
            message=message,
            skill_id="nutrition_info"
        )


class StoreInfoHandler(SkillHandler):
    """门店信息处理器"""

    STORES = {
        "SH001": {
            "name": "示例咖啡店（旗舰店）",
            "address": "上海市黄浦区南京东路100号",
            "hours": "07:00-22:00",
            "phone": "021-12345678"
        },
        "SH002": {
            "name": "示例咖啡店（陆家嘴店）",
            "address": "上海市浦东新区陆家嘴环路168号",
            "hours": "07:00-22:00",
            "phone": "021-87654321"
        },
        "DEFAULT": {
            "name": "示例咖啡店（默认门店）",
            "address": "上海市中心",
            "hours": "07:00-22:00",
            "phone": "400-123-4567"
        }
    }

    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        store_id = params.get("store_id", "DEFAULT")

        store = self.STORES.get(store_id, self.STORES["DEFAULT"])

        message = f"🏪 {store['name']}\n"
        message += f"📍 {store['address']}\n"
        message += f"🕐 营业时间: {store['hours']}\n"
        message += f"📞 {store['phone']}"

        return SkillResult(
            success=True,
            data={
                "store_name": store["name"],
                "address": store["address"],
                "hours": store["hours"],
                "phone": store["phone"]
            },
            message=message,
            skill_id="store_info"
        )


class ModifyOrderHandler(SkillHandler):
    """订单修改处理器"""

    PRICE_DIFF = {
        "size": {"中杯": 0, "大杯": 4, "超大杯": 7},
        "milk_type": {"全脂奶": 0, "燕麦奶": 6, "椰奶": 6, "豆奶": 4}
    }

    def execute(self, params: Dict[str, Any], context: Dict[str, Any] = None) -> SkillResult:
        order_id = params.get("order_id")
        modifications = params.get("modifications", {})

        if not order_id:
            return SkillResult(
                success=False,
                error="请提供订单号",
                skill_id="modify_order"
            )

        if not modifications:
            return SkillResult(
                success=False,
                error="请指定要修改的内容",
                skill_id="modify_order"
            )

        # 计算差价
        price_diff = 0
        changes = []

        if "size" in modifications:
            new_size = modifications["size"]
            price_diff += self.PRICE_DIFF["size"].get(new_size, 0)
            changes.append(f"杯型改为{new_size}")

        if "milk_type" in modifications:
            new_milk = modifications["milk_type"]
            price_diff += self.PRICE_DIFF["milk_type"].get(new_milk, 0)
            changes.append(f"奶类改为{new_milk}")

        if "temperature" in modifications:
            changes.append(f"温度改为{modifications['temperature']}")

        if "sweetness" in modifications:
            changes.append(f"甜度改为{modifications['sweetness']}")

        if "extras" in modifications:
            changes.append(f"添加{'/'.join(modifications['extras'])}")
            price_diff += len(modifications["extras"]) * 5  # 假设配料每个5元

        if changes:
            message = f"订单 {order_id} 已修改：{', '.join(changes)}"
            if price_diff > 0:
                message += f"，需补差价 ¥{price_diff}"
            elif price_diff < 0:
                message += f"，可退还 ¥{-price_diff}"
        else:
            message = "订单无变更"

        return SkillResult(
            success=True,
            data={
                "success": True,
                "order_id": order_id,
                "modifications": modifications,
                "changes": changes,
                "price_diff": price_diff
            },
            message=message,
            skill_id="modify_order"
        )


# ==================== 技能注册中心 ====================

class SkillRegistry:
    """
    技能注册中心

    负责:
    - 加载和管理技能定义
    - 技能匹配和路由
    - 处理器注册
    """

    def __init__(self):
        self.skills: Dict[str, SkillDefinition] = {}
        self.categories: Dict[str, CategoryDefinition] = {}
        self.handlers: Dict[str, SkillHandler] = {}
        self.settings: Dict[str, Any] = {}
        self._version: str = "0.0.0"
        self._config_path: Optional[str] = None

        # 注册内置处理器
        self._register_builtin_handlers()

    def _register_builtin_handlers(self):
        """注册内置处理器"""
        self.register_handler("inventory.check_stock", InventoryCheckHandler())
        self.register_handler("payment.apply_coupon", ApplyCouponHandler())
        self.register_handler("customer.check_points", CheckPointsHandler())
        self.register_handler("recommendation.smart_recommend", SmartRecommendHandler())
        self.register_handler("order.estimate_time", EstimateTimeHandler())
        self.register_handler("inventory.nutrition_info", NutritionInfoHandler())
        self.register_handler("customer.store_info", StoreInfoHandler())
        self.register_handler("order.modify_order", ModifyOrderHandler())

    def load_from_yaml(self, path: str) -> 'SkillRegistry':
        """从YAML文件加载配置"""
        self._config_path = path

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._version = config.get("version", "0.0.0")
        self.settings = config.get("settings", {})

        # 加载分类
        self._load_categories(config.get("categories", {}))

        # 加载技能
        self._load_skills(config.get("skills", {}))

        return self

    def _load_categories(self, categories_config: Dict):
        """加载分类配置"""
        self.categories.clear()
        for cat_id, cat_config in categories_config.items():
            self.categories[cat_id] = CategoryDefinition(
                id=cat_id,
                name=cat_config.get("name", cat_id),
                description=cat_config.get("description", ""),
                icon=cat_config.get("icon", "")
            )

    def _load_skills(self, skills_config: Dict):
        """加载技能配置"""
        self.skills.clear()

        for skill_id, skill_config in skills_config.items():
            # 解析触发条件
            triggers_config = skill_config.get("triggers", {})
            triggers = SkillTrigger(
                keywords=triggers_config.get("keywords", []),
                intents=triggers_config.get("intents", [])
            )

            # 解析参数
            parameters = []
            for param_name, param_config in skill_config.get("parameters", {}).items():
                parameters.append(ParameterDefinition(
                    name=param_name,
                    type=param_config.get("type", "string"),
                    description=param_config.get("description", ""),
                    required=param_config.get("required", False),
                    default=param_config.get("default"),
                    enum=param_config.get("enum", []),
                    validation=param_config.get("validation", {}),
                    examples=param_config.get("examples", [])
                ))

            # 解析返回值
            returns_config = skill_config.get("returns", {})
            returns = ReturnDefinition(
                type=returns_config.get("type", "object"),
                properties=returns_config.get("properties", {}),
                description=returns_config.get("description", "")
            ) if returns_config else None

            # 解析测试用例
            test_cases = []
            for tc in skill_config.get("test_cases", []):
                test_cases.append(TestCase(
                    name=tc.get("name", ""),
                    input=tc.get("input", {}),
                    expected=tc.get("expected", {}),
                    validate=tc.get("validate")
                ))

            self.skills[skill_id] = SkillDefinition(
                id=skill_id,
                name=skill_config.get("name", skill_id),
                description=skill_config.get("description", ""),
                category=skill_config.get("category", "default"),
                enabled=skill_config.get("enabled", True),
                triggers=triggers,
                parameters=parameters,
                returns=returns,
                handler=skill_config.get("handler", ""),
                examples=skill_config.get("examples", []),
                test_cases=test_cases
            )

    def register_handler(self, handler_id: str, handler: SkillHandler):
        """注册处理器"""
        self.handlers[handler_id] = handler

    def get_skill(self, skill_id: str) -> Optional[SkillDefinition]:
        """获取技能定义"""
        return self.skills.get(skill_id)

    def get_handler(self, handler_id: str) -> Optional[SkillHandler]:
        """获取处理器"""
        return self.handlers.get(handler_id)

    def find_matching_skills(self, text: str, intent: Optional[str] = None) -> List[Tuple[SkillDefinition, float]]:
        """
        查找匹配的技能

        Returns:
            List of (skill, score) tuples, sorted by score descending
        """
        matches = []

        for skill in self.skills.values():
            if not skill.enabled:
                continue

            matched, score = skill.matches(text, intent)
            if matched:
                matches.append((skill, score))

        # 按分数排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_enabled_skills(self) -> List[SkillDefinition]:
        """获取所有启用的技能"""
        return [s for s in self.skills.values() if s.enabled]

    def generate_skills_schema(self) -> List[Dict]:
        """生成所有技能的 Function Calling Schema"""
        return [skill.to_function_schema() for skill in self.get_enabled_skills()]

    @property
    def version(self) -> str:
        return self._version

    def __repr__(self) -> str:
        enabled = len(self.get_enabled_skills())
        return f"SkillRegistry(v{self._version}, skills={len(self.skills)}, enabled={enabled}, handlers={len(self.handlers)})"


# ==================== 技能执行器 ====================

class SkillExecutor:
    """
    技能执行器

    负责:
    - 技能调用
    - 参数验证
    - 结果处理
    - 错误处理和重试
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def execute(self, skill_id: str, params: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """
        执行技能

        Args:
            skill_id: 技能ID
            params: 参数
            context: 上下文信息

        Returns:
            SkillResult
        """
        start_time = time.time()

        # 获取技能定义
        skill = self.registry.get_skill(skill_id)
        if not skill:
            return SkillResult(
                success=False,
                error=f"技能 '{skill_id}' 不存在",
                skill_id=skill_id
            )

        if not skill.enabled:
            return SkillResult(
                success=False,
                error=f"技能 '{skill_id}' 已禁用",
                skill_id=skill_id
            )

        # 验证参数
        valid, error_msg = skill.validate_params(params)
        if not valid:
            return SkillResult(
                success=False,
                error=error_msg,
                skill_id=skill_id
            )

        # 获取处理器
        handler = self.registry.get_handler(skill.handler)
        if not handler:
            return SkillResult(
                success=False,
                error=f"处理器 '{skill.handler}' 未注册",
                skill_id=skill_id
            )

        # 执行
        try:
            result = handler.execute(params, context)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"执行错误: {str(e)}",
                skill_id=skill_id,
                execution_time=time.time() - start_time
            )

    def execute_by_match(self, text: str, intent: Optional[str] = None,
                         params: Optional[Dict] = None,
                         context: Optional[Dict] = None) -> Optional[SkillResult]:
        """
        根据匹配执行技能

        自动选择最匹配的技能执行
        """
        matches = self.registry.find_matching_skills(text, intent)

        if not matches:
            return None

        # 选择得分最高的技能
        best_skill, score = matches[0]

        # 合并参数
        exec_params = params or {}

        return self.execute(best_skill.id, exec_params, context)


# ==================== 测试框架 ====================

class SkillTester:
    """技能测试器"""

    def __init__(self, registry: SkillRegistry, executor: SkillExecutor):
        self.registry = registry
        self.executor = executor

    def run_test_case(self, skill_id: str, test_case: TestCase) -> Tuple[bool, str]:
        """
        运行单个测试用例

        Returns:
            (passed, message)
        """
        # 添加测试模式标记
        test_input = {**test_case.input, "_test_mode": True}
        result = self.executor.execute(skill_id, test_input)

        if not result.success:
            return False, f"执行失败: {result.error}"

        # 验证预期结果
        for key, expected_value in test_case.expected.items():
            actual_value = result.data.get(key)
            if actual_value != expected_value:
                return False, f"字段 '{key}' 期望 {expected_value}，实际 {actual_value}"

        # 执行自定义验证表达式
        if test_case.validate:
            try:
                # 安全执行验证表达式
                local_vars = {"result": result.data}
                if not eval(test_case.validate, {"__builtins__": {}}, local_vars):
                    return False, f"验证表达式失败: {test_case.validate}"
            except Exception as e:
                return False, f"验证表达式错误: {e}"

        return True, "通过"

    def run_skill_tests(self, skill_id: str) -> Dict[str, Any]:
        """运行技能的所有测试"""
        skill = self.registry.get_skill(skill_id)
        if not skill:
            return {"error": f"技能 '{skill_id}' 不存在"}

        results = {
            "skill_id": skill_id,
            "skill_name": skill.name,
            "total": len(skill.test_cases),
            "passed": 0,
            "failed": 0,
            "cases": []
        }

        for tc in skill.test_cases:
            passed, message = self.run_test_case(skill_id, tc)
            results["cases"].append({
                "name": tc.name,
                "passed": passed,
                "message": message
            })
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有技能的测试"""
        all_results = {
            "total_skills": 0,
            "total_cases": 0,
            "passed": 0,
            "failed": 0,
            "skills": []
        }

        for skill_id in self.registry.skills:
            skill_results = self.run_skill_tests(skill_id)
            if "error" not in skill_results:
                all_results["total_skills"] += 1
                all_results["total_cases"] += skill_results["total"]
                all_results["passed"] += skill_results["passed"]
                all_results["failed"] += skill_results["failed"]
                all_results["skills"].append(skill_results)

        return all_results


# ==================== 全局实例 ====================

_global_registry: Optional[SkillRegistry] = None
_global_executor: Optional[SkillExecutor] = None


def get_skill_registry() -> SkillRegistry:
    """获取全局技能注册中心"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
        # 尝试加载默认配置
        default_path = Path(__file__).parent / "schema" / "skills.yaml"
        if default_path.exists():
            _global_registry.load_from_yaml(str(default_path))
            print(f"✅ Skills配置已加载: {_global_registry}")
    return _global_registry


def get_skill_executor() -> SkillExecutor:
    """获取全局技能执行器"""
    global _global_executor
    if _global_executor is None:
        _global_executor = SkillExecutor(get_skill_registry())
    return _global_executor


# ==================== 测试代码 ====================

def test_skills():
    """测试技能系统"""
    print("=" * 60)
    print("Skills 执行层测试")
    print("=" * 60)

    registry = get_skill_registry()
    executor = get_skill_executor()
    tester = SkillTester(registry, executor)

    print(f"\n{registry}")
    print(f"已注册处理器: {list(registry.handlers.keys())}")

    # 测试单个技能执行
    print("\n" + "-" * 40)
    print("技能执行测试")
    print("-" * 40)

    test_executions = [
        ("check_inventory", {"product_name": "拿铁"}),
        ("apply_coupon", {"coupon_code": "COFFEE20", "order_total": 50}),
        ("check_points", {"member_id": "M001"}),
        ("smart_recommend", {"weather": "hot"}),
        ("estimate_time", {"order_items": ["拿铁", "美式咖啡"]}),
        ("nutrition_info", {"product_name": "拿铁", "size": "大杯"}),
        ("store_info", {"store_id": "SH001"}),
    ]

    for skill_id, params in test_executions:
        result = executor.execute(skill_id, params)
        status = "✅" if result.success else "❌"
        print(f"\n{status} {skill_id}")
        print(f"   参数: {params}")
        print(f"   消息: {result.message[:60]}..." if len(result.message) > 60 else f"   消息: {result.message}")
        print(f"   耗时: {result.execution_time*1000:.1f}ms")

    # 测试技能匹配
    print("\n" + "-" * 40)
    print("技能匹配测试")
    print("-" * 40)

    test_texts = [
        ("拿铁还有吗", None),
        ("我有优惠券", "PAYMENT"),
        ("我有多少积分", None),
        ("天气很热喝什么好", "RECOMMEND"),
        ("要等多久", "ORDER_QUERY"),
        ("拿铁多少卡路里", "PRODUCT_INFO"),
    ]

    for text, intent in test_texts:
        matches = registry.find_matching_skills(text, intent)
        if matches:
            best_skill, score = matches[0]
            print(f"\n'{text}' (intent={intent})")
            print(f"   匹配: {best_skill.name} (score={score:.2f})")
        else:
            print(f"\n'{text}' -> 无匹配")

    # 运行所有测试用例
    print("\n" + "-" * 40)
    print("测试用例执行")
    print("-" * 40)

    all_results = tester.run_all_tests()
    print(f"\n总计: {all_results['total_skills']} 个技能, {all_results['total_cases']} 个用例")
    print(f"通过: {all_results['passed']}, 失败: {all_results['failed']}")

    for skill_result in all_results["skills"]:
        status = "✅" if skill_result["failed"] == 0 else "⚠️"
        print(f"\n{status} {skill_result['skill_name']} ({skill_result['passed']}/{skill_result['total']})")
        for case in skill_result["cases"]:
            case_status = "✓" if case["passed"] else "✗"
            print(f"   {case_status} {case['name']}: {case['message']}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_skills()
