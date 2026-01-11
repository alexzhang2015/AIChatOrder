"""
评估环境管理

提供隔离的测试环境，支持:
- 会话状态重置
- 订单状态管理
- 模拟服务注入
"""

import uuid
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OrderItem:
    """订单项"""
    product_name: str
    size: str = "中杯"
    temperature: str = "热"
    sweetness: str = "标准"
    milk_type: str = "牛奶"
    extras: List[str] = field(default_factory=list)
    quantity: int = 1
    price: float = 0.0


@dataclass
class Order:
    """订单"""
    order_id: str
    items: List[OrderItem] = field(default_factory=list)
    total: float = 0.0
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionState:
    """会话状态"""
    session_id: str
    order: Optional[Order] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class EvalEnvironment:
    """
    评估环境管理器

    提供隔离的测试环境，每次评估前重置状态
    """

    def __init__(self):
        self._session: Optional[SessionState] = None
        self._classifier = None
        self._workflow = None
        self._rules_engine = None
        self._fuzzy_matcher = None
        self._initialized = False

    def initialize(self):
        """初始化环境，加载必要的组件"""
        if self._initialized:
            return

        try:
            # 延迟导入以避免循环依赖
            from services.classifier import OpenAIClassifier
            from services.rules_engine import get_rules_engine, get_fuzzy_matcher

            self._classifier = OpenAIClassifier()
            self._rules_engine = get_rules_engine()
            self._fuzzy_matcher = get_fuzzy_matcher()
            self._initialized = True
            logger.info("评估环境初始化完成")
        except Exception as e:
            logger.error(f"评估环境初始化失败: {e}")
            raise

    def reset(self, session_id: Optional[str] = None):
        """
        重置环境状态

        Args:
            session_id: 可选的会话 ID，不提供则自动生成
        """
        self._session = SessionState(
            session_id=session_id or str(uuid.uuid4())
        )
        logger.debug(f"环境已重置，新会话: {self._session.session_id}")

    def get_state(self) -> Dict[str, Any]:
        """获取当前环境状态"""
        if not self._session:
            return {}

        return {
            "session_id": self._session.session_id,
            "order": self._order_to_dict(self._session.order) if self._session.order else None,
            "messages": self._session.messages,
            "context": self._session.context
        }

    def set_state(self, state: Dict[str, Any]):
        """设置环境状态"""
        if not self._session:
            self.reset()

        if "order" in state and state["order"]:
            order_data = state["order"]
            items = [
                OrderItem(**item) if isinstance(item, dict) else item
                for item in order_data.get("items", [])
            ]
            self._session.order = Order(
                order_id=order_data.get("order_id", str(uuid.uuid4())),
                items=items,
                total=order_data.get("total", 0),
                status=order_data.get("status", "pending")
            )

        if "messages" in state:
            self._session.messages = state["messages"]

        if "context" in state:
            self._session.context = state["context"]

    def add_message(self, role: str, content: str, **kwargs):
        """添加消息到会话"""
        if not self._session:
            self.reset()

        self._session.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        })

    def classify(self, text: str, method: str = "few_shot") -> Dict[str, Any]:
        """
        调用分类器进行意图识别

        Args:
            text: 用户输入
            method: 分类方法

        Returns:
            分类结果
        """
        if not self._initialized:
            self.initialize()

        try:
            result = self._classifier.classify(text, method=method)
            return result
        except Exception as e:
            logger.error(f"分类失败: {e}")
            return {"intent": "UNKNOWN", "confidence": 0.0, "slots": {}, "error": str(e)}

    def validate_constraints(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证产品约束

        Args:
            slots: 槽位数据

        Returns:
            验证结果
        """
        if not self._initialized:
            self.initialize()

        try:
            result = self._rules_engine.validate_and_adjust(
                product_name=slots.get("product_name", ""),
                slots=slots
            )
            return {
                "valid": result.valid,
                "adjusted_slots": result.adjusted_slots,
                "warnings": result.warnings,
                "auto_corrections": [
                    {"slot": c.get("slot"), "from": c.get("from"), "to": c.get("to"), "message": c.get("message")}
                    for c in result.auto_corrections
                ] if hasattr(result, "auto_corrections") else []
            }
        except Exception as e:
            logger.error(f"约束验证失败: {e}")
            return {"valid": True, "adjusted_slots": slots, "warnings": [], "auto_corrections": [], "error": str(e)}

    def match_fuzzy_expression(self, text: str) -> Dict[str, Any]:
        """
        匹配模糊表达

        Args:
            text: 用户输入

        Returns:
            匹配结果
        """
        if not self._initialized:
            self.initialize()

        try:
            result = self._fuzzy_matcher.match(text)
            if result:
                return {
                    "matched": result.matched,
                    "slot_name": result.slot_name,
                    "value": result.value,
                    "confidence": result.confidence,
                    "pattern": result.pattern,
                    "action": result.action,
                    "extra_mappings": result.extra_mappings
                }
            return {"matched": False}
        except Exception as e:
            logger.error(f"模糊匹配失败: {e}")
            return {"matched": False, "error": str(e)}

    def create_order(self, items: List[Dict[str, Any]]) -> Order:
        """创建订单"""
        if not self._session:
            self.reset()

        order_items = [OrderItem(**item) for item in items]
        total = sum(item.price * item.quantity for item in order_items)

        self._session.order = Order(
            order_id=str(uuid.uuid4())[:8].upper(),
            items=order_items,
            total=total,
            status="pending"
        )

        return self._session.order

    def _order_to_dict(self, order: Order) -> Dict[str, Any]:
        """将订单对象转换为字典"""
        return {
            "order_id": order.order_id,
            "items": [
                {
                    "product_name": item.product_name,
                    "size": item.size,
                    "temperature": item.temperature,
                    "sweetness": item.sweetness,
                    "milk_type": item.milk_type,
                    "extras": item.extras,
                    "quantity": item.quantity,
                    "price": item.price
                }
                for item in order.items
            ],
            "total": order.total,
            "status": order.status,
            "created_at": order.created_at
        }

    @property
    def session_id(self) -> Optional[str]:
        """获取当前会话 ID"""
        return self._session.session_id if self._session else None

    @property
    def classifier(self):
        """获取分类器"""
        if not self._initialized:
            self.initialize()
        return self._classifier

    @property
    def rules_engine(self):
        """获取规则引擎"""
        if not self._initialized:
            self.initialize()
        return self._rules_engine

    @property
    def fuzzy_matcher(self):
        """获取模糊匹配器"""
        if not self._initialized:
            self.initialize()
        return self._fuzzy_matcher

    def agent_respond(self, user_input: str) -> Dict[str, Any]:
        """
        Agent 响应函数，用于对话模拟

        Args:
            user_input: 用户输入

        Returns:
            包含响应文本和订单状态的字典
        """
        if not self._initialized:
            self.initialize()

        if not self._session:
            self.reset()

        # 记录用户消息
        self.add_message("user", user_input)

        try:
            # 1. 意图识别和槽位提取
            result = self.classify(user_input)
            intent = result.get("intent", "UNKNOWN")
            slots = result.get("slots", {})
            confidence = result.get("confidence", 0)

            # 2. 应用模糊匹配
            fuzzy_result = self.match_fuzzy_expression(user_input)
            if fuzzy_result.get("matched"):
                slot_name = fuzzy_result.get("slot_name")
                if slot_name:
                    slots[slot_name] = fuzzy_result.get("value")

            # 3. 生成响应
            response_text, order_complete = self._generate_response(intent, slots)

            # 4. 更新订单状态
            if intent == "ORDER_NEW" and slots.get("product_name"):
                self._update_order(slots)

            # 记录 Agent 消息
            self.add_message("agent", response_text, intent=intent, slots=slots)

            return {
                "response": response_text,
                "intent": intent,
                "slots": slots,
                "confidence": confidence,
                "order_complete": order_complete,
                "order": self._order_to_dict(self._session.order) if self._session.order else None
            }

        except Exception as e:
            logger.error(f"Agent 响应失败: {e}")
            return {
                "response": "抱歉，系统出现了问题，请稍后再试。",
                "error": str(e),
                "order_complete": False
            }

    def _generate_response(self, intent: str, slots: Dict[str, Any]) -> tuple:
        """
        根据意图生成响应

        Returns:
            (响应文本, 是否完成订单)
        """
        order_complete = False

        if intent == "ORDER_NEW":
            product = slots.get("product_name", "")
            if product:
                # 检查必要槽位
                missing = []
                if not slots.get("size"):
                    missing.append("杯型")
                if not slots.get("temperature"):
                    missing.append("温度")

                if missing:
                    response = f"好的，{product}。请问要{missing[0]}呢？"
                else:
                    size = slots.get("size", "中杯")
                    temp = slots.get("temperature", "热")
                    sweetness = slots.get("sweetness", "标准糖")
                    response = f"好的，{size}{temp}{product}，{sweetness}。确认订单吗？"
                    # 如果所有必要信息都有了，标记为等待确认
                    if self._session and self._session.order:
                        order_complete = True
            else:
                response = "请问您想喝点什么？"

        elif intent == "ORDER_MODIFY":
            if self._session and self._session.order:
                response = "好的，已经帮您修改了订单。还有其他需要吗？"
            else:
                response = "您还没有下单哦，请先告诉我想喝什么？"

        elif intent == "ORDER_CANCEL":
            if self._session and self._session.order:
                self._session.order = None
                response = "好的，已取消订单。下次再来哦！"
            else:
                response = "您还没有下单哦。"

        elif intent == "ORDER_QUERY":
            if self._session and self._session.order:
                items = self._session.order.items
                if items:
                    desc = "、".join([f"{i.size}{i.temperature}{i.product_name}" for i in items])
                    response = f"您的订单：{desc}，共 {self._session.order.total} 元。"
                else:
                    response = "您的购物车是空的。"
            else:
                response = "您还没有下单哦。"

        elif intent == "RECOMMEND":
            response = "我们的招牌是拿铁，香浓顺滑。或者您可以试试美式，提神醒脑。您想要哪个？"

        elif intent == "PRODUCT_INFO":
            product = slots.get("product_name", "")
            if product:
                response = f"{product}是我们的人气产品，口感丰富，很受欢迎。"
            else:
                response = "请问您想了解哪款产品？"

        elif intent == "CHITCHAT":
            response = "哈哈，请问您想喝点什么咖啡呢？"

        else:
            response = "不好意思没听清，您是要点咖啡吗？"

        return response, order_complete

    def _update_order(self, slots: Dict[str, Any]):
        """更新订单"""
        if not self._session:
            self.reset()

        product_name = slots.get("product_name", "")
        if not product_name:
            return

        # 验证约束
        validated = self.validate_constraints(slots)
        adjusted_slots = validated.get("adjusted_slots", slots)

        item = OrderItem(
            product_name=product_name,
            size=adjusted_slots.get("size", "中杯"),
            temperature=adjusted_slots.get("temperature", "热"),
            sweetness=adjusted_slots.get("sweetness", "标准"),
            milk_type=adjusted_slots.get("milk_type", "牛奶"),
            quantity=adjusted_slots.get("quantity", 1),
            price=self._get_price(product_name, adjusted_slots.get("size", "中杯"))
        )

        if self._session.order:
            self._session.order.items.append(item)
            self._session.order.total += item.price * item.quantity
        else:
            self._session.order = Order(
                order_id=str(uuid.uuid4())[:8].upper(),
                items=[item],
                total=item.price * item.quantity,
                status="pending"
            )

    def _get_price(self, product_name: str, size: str = "中杯") -> float:
        """获取产品价格（简化实现）"""
        base_prices = {
            "美式咖啡": 22.0,
            "美式": 22.0,
            "拿铁": 28.0,
            "卡布奇诺": 28.0,
            "摩卡": 32.0,
            "焦糖玛奇朵": 32.0,
            "星冰乐": 35.0,
        }

        size_factor = {"小杯": 0.8, "中杯": 1.0, "大杯": 1.2}

        base = base_prices.get(product_name, 25.0)
        factor = size_factor.get(size, 1.0)

        return round(base * factor, 1)
