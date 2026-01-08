"""点单对话助手"""

import time
from typing import Dict, Optional, TYPE_CHECKING

from models.session import Session, ConversationState
from models.order import Order, OrderItem
from data.menu import MENU
from .session_manager import SessionManager

if TYPE_CHECKING:
    from .classifier import OpenAIClassifier


class OrderingAssistant:
    """点单对话助手"""

    def __init__(self, classifier: 'OpenAIClassifier'):
        self.classifier = classifier
        self.session_manager = SessionManager()

    def process_message(self, session_id: Optional[str], user_message: str, method: str = "function_calling") -> Dict:
        """处理用户消息"""
        # 获取或创建会话
        if session_id:
            session = self.session_manager.get_session(session_id)
        else:
            session = None

        if not session:
            session = self.session_manager.create_session()

        # 添加用户消息
        session.add_message("user", user_message)

        # 意图识别
        if method == "zero_shot":
            intent_result = self.classifier.classify_zero_shot(user_message)
        elif method == "few_shot":
            intent_result = self.classifier.classify_few_shot(user_message)
        elif method == "rag_enhanced":
            intent_result = self.classifier.classify_rag(user_message)
        else:
            intent_result = self.classifier.classify_function_calling(user_message)

        intent = intent_result.get("intent", "UNKNOWN")
        slots = intent_result.get("slots", {})
        confidence = intent_result.get("confidence", 0)

        # 根据意图和状态生成回复
        response = self._generate_response(session, intent, slots, intent_result)

        # 添加助手回复
        session.add_message("assistant", response["reply"], {
            "intent": intent,
            "confidence": confidence,
            "slots": slots
        })

        # 构建返回结果
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "reply": response["reply"],
            "intent_result": intent_result,
            "order": self._serialize_order(session.current_order) if session.current_order else None,
            "history": session.history,
            "suggestions": response.get("suggestions", []),
            "actions": response.get("actions", [])
        }

    def _generate_response(self, session: Session, intent: str, slots: Dict, intent_result: Dict) -> Dict:
        """根据意图生成回复"""
        response = {"reply": "", "suggestions": [], "actions": []}

        if intent == "CHITCHAT":
            response = self._handle_chitchat(session, slots)

        elif intent == "ORDER_NEW":
            # 检查是否实际上是确认下单
            if session.current_order and session.current_order.items:
                # 如果已有订单且没有提取到产品名，可能是确认下单
                if not slots.get("product_name"):
                    response = self._handle_payment(session)
                else:
                    response = self._handle_new_order(session, slots)
            else:
                response = self._handle_new_order(session, slots)

        elif intent == "ORDER_MODIFY" or intent == "CUSTOMIZE":
            response = self._handle_modify_order(session, slots)

        elif intent == "ORDER_CANCEL":
            response = self._handle_cancel_order(session)

        elif intent == "ORDER_QUERY":
            response = self._handle_query_order(session)

        elif intent == "PRODUCT_INFO":
            response = self._handle_product_info(slots)

        elif intent == "RECOMMEND":
            response = self._handle_recommend(session)

        elif intent == "PAYMENT":
            response = self._handle_payment(session)

        elif intent == "COMPLAINT":
            response = self._handle_complaint(session)

        else:
            response["reply"] = "抱歉，我没有理解您的意思。您可以说'我要点一杯拿铁'来点单，或者问我'有什么推荐的'。"
            response["suggestions"] = ["我要点杯拿铁", "有什么推荐的", "菜单有什么"]

        return response

    def _handle_chitchat(self, session: Session, slots: Dict) -> Dict:
        greetings = [
            "您好！欢迎光临，请问想喝点什么？",
            "您好！今天想来杯什么咖啡呢？",
            "欢迎光临！我们有多款特色饮品，需要我推荐一下吗？"
        ]
        if session.state == ConversationState.GREETING:
            session.state = ConversationState.TAKING_ORDER

        return {
            "reply": greetings[hash(session.session_id) % len(greetings)],
            "suggestions": ["来杯拿铁", "有什么推荐", "看看菜单"],
            "actions": []
        }

    def _normalize_product_name(self, name: str) -> str:
        """规范化产品名称"""
        if not name:
            return None
        # 处理常见变体
        name_map = {
            "冰美式": "美式咖啡", "热美式": "美式咖啡", "美式": "美式咖啡",
            "冰拿铁": "拿铁", "热拿铁": "拿铁",
            "冰摩卡": "摩卡", "热摩卡": "摩卡",
            "澳白": "馥芮白",
        }
        normalized = name_map.get(name, name)
        # 检查是否在菜单中
        if normalized in MENU:
            return normalized
        # 模糊匹配
        for menu_name in MENU:
            if menu_name in name or name in menu_name:
                return menu_name
        return normalized

    def _handle_new_order(self, session: Session, slots: Dict) -> Dict:
        product_name = self._normalize_product_name(slots.get("product_name"))

        if not product_name:
            return {
                "reply": "请问您想点什么饮品呢？我们有拿铁、美式、卡布奇诺、摩卡等。",
                "suggestions": ["拿铁", "美式咖啡", "卡布奇诺", "有什么推荐"],
                "actions": []
            }

        # 创建订单项
        item = OrderItem(
            product_name=product_name,
            size=slots.get("size", "中杯"),
            temperature=slots.get("temperature", "热"),
            sweetness=slots.get("sweetness", "标准"),
            milk_type=slots.get("milk_type", "全脂奶"),
            extras=slots.get("extras", []),
            quantity=slots.get("quantity", 1)
        )

        # 创建或更新订单
        if not session.current_order:
            session.current_order = Order(order_id=f"ORD{int(time.time()) % 100000:05d}")

        session.current_order.items.append(item)
        session.state = ConversationState.CONFIRMING

        item_desc = item.to_string()
        price = item.get_price()
        total = session.current_order.get_total()

        reply = f"好的，已添加 {item_desc}（¥{price:.0f}）。"
        if len(session.current_order.items) > 1:
            reply += f"\n当前订单共 {len(session.current_order.items)} 件商品，合计 ¥{total:.0f}。"
        reply += "\n\n请问还需要别的吗？或者确认下单？"

        return {
            "reply": reply,
            "suggestions": ["确认下单", "再来一杯", "换成大杯", "取消订单"],
            "actions": ["confirm_order", "add_item", "modify", "cancel"]
        }

    def _handle_modify_order(self, session: Session, slots: Dict) -> Dict:
        if not session.current_order or not session.current_order.items:
            return {
                "reply": "您还没有点单哦，请先告诉我您想喝什么？",
                "suggestions": ["来杯拿铁", "看看菜单", "有什么推荐"],
                "actions": []
            }

        # 修改最后一个订单项
        last_item = session.current_order.items[-1]
        modified = []

        if "size" in slots:
            last_item.size = slots["size"]
            modified.append(f"杯型改为{slots['size']}")

        if "temperature" in slots:
            last_item.temperature = slots["temperature"]
            modified.append(f"温度改为{slots['temperature']}")

        if "sweetness" in slots:
            last_item.sweetness = slots["sweetness"]
            modified.append(f"甜度改为{slots['sweetness']}")

        if "milk_type" in slots:
            last_item.milk_type = slots["milk_type"]
            modified.append(f"奶类改为{slots['milk_type']}")

        if "extras" in slots:
            last_item.extras.extend(slots["extras"])
            modified.append(f"添加{'/'.join(slots['extras'])}")

        # 处理产品名称为配料的情况
        product_in_slot = slots.get("product_name", "")
        if product_in_slot and product_in_slot not in MENU:
            extra_map = {"浓缩": "浓缩shot", "香草": "香草糖浆", "焦糖": "焦糖糖浆"}
            for key, extra in extra_map.items():
                if key in product_in_slot and extra not in last_item.extras:
                    last_item.extras.append(extra)
                    modified.append(f"添加{extra}")

        if modified:
            reply = f"好的，已为您{', '.join(modified)}。\n"
            reply += f"现在是：{last_item.to_string()}（¥{last_item.get_price():.0f}）\n"
            reply += "还需要其他调整吗？"
        else:
            reply = "请问您想修改什么呢？可以换杯型、温度、甜度或奶类。"

        return {
            "reply": reply,
            "suggestions": ["确认下单", "换成冰的", "少糖", "加燕麦奶"],
            "actions": ["confirm_order", "modify"]
        }

    def _handle_cancel_order(self, session: Session) -> Dict:
        if not session.current_order or not session.current_order.items:
            return {
                "reply": "您还没有点单哦，不需要取消。请问想喝点什么？",
                "suggestions": ["来杯拿铁", "有什么推荐"],
                "actions": []
            }

        session.current_order.status = "cancelled"
        items_count = len(session.current_order.items)
        session.current_order = None
        session.state = ConversationState.TAKING_ORDER

        return {
            "reply": f"好的，已为您取消订单（共{items_count}件商品）。\n有需要随时再点哦！",
            "suggestions": ["重新点单", "看看菜单"],
            "actions": []
        }

    def _handle_query_order(self, session: Session) -> Dict:
        if not session.current_order:
            return {
                "reply": "您目前没有进行中的订单。请问需要点单吗？",
                "suggestions": ["我要点单", "看看菜单"],
                "actions": []
            }

        order = session.current_order
        status_text = {
            "pending": "待确认",
            "confirmed": "已确认，准备制作",
            "preparing": "制作中",
            "ready": "已完成，请取餐",
            "completed": "已完成",
            "cancelled": "已取消"
        }

        items_text = "\n".join([f"  * {item.to_string()} ¥{item.get_price():.0f}" for item in order.items])

        return {
            "reply": f"您的订单 {order.order_id}：\n{items_text}\n\n合计：¥{order.get_total():.0f}\n状态：{status_text.get(order.status, order.status)}",
            "suggestions": ["确认下单", "修改订单", "取消订单"],
            "actions": ["confirm_order", "modify", "cancel"]
        }

    def _handle_product_info(self, slots: Dict) -> Dict:
        product_name = slots.get("product_name")

        if product_name and product_name in MENU:
            info = MENU[product_name]
            reply = f"【{product_name}】\n"
            reply += f"价格：¥{info['price']}（中杯）\n"
            reply += f"热量：{info['calories']} 大卡\n"
            reply += f"介绍：{info['desc']}\n\n"
            reply += "需要来一杯吗？"
            suggestions = [f"来杯{product_name}", "看看其他", "有什么推荐"]
        else:
            # 显示菜单
            reply = "我们的菜单：\n\n"
            for name, info in MENU.items():
                reply += f"* {name}  ¥{info['price']}\n"
            reply += "\n升杯：大杯+4元，超大杯+7元\n换奶：燕麦奶/椰奶+6元\n\n请问想喝什么？"
            suggestions = ["拿铁", "美式咖啡", "推荐一下"]

        return {
            "reply": reply,
            "suggestions": suggestions,
            "actions": []
        }

    def _handle_recommend(self, session: Session) -> Dict:
        recommendations = [
            ("拿铁", "最受欢迎的经典选择，奶香与咖啡香完美平衡"),
            ("馥芮白", "澳洲风味，比拿铁更浓郁顺滑"),
            ("美式咖啡", "低卡之选，适合注重健康的您"),
        ]

        reply = "为您推荐：\n\n"
        for name, reason in recommendations:
            info = MENU[name]
            reply += f"* {name}（¥{info['price']}）\n   {reason}\n\n"
        reply += "请问想试试哪一款？"

        return {
            "reply": reply,
            "suggestions": ["来杯拿铁", "来杯馥芮白", "来杯美式"],
            "actions": []
        }

    def _handle_payment(self, session: Session) -> Dict:
        if not session.current_order or not session.current_order.items:
            return {
                "reply": "您还没有点单哦，请先告诉我您想喝什么？",
                "suggestions": ["来杯拿铁", "看看菜单"],
                "actions": []
            }

        order = session.current_order
        total = order.get_total()

        # 模拟确认订单
        order.status = "confirmed"
        session.state = ConversationState.COMPLETED

        items_text = "\n".join([f"  * {item.to_string()} ¥{item.get_price():.0f}" for item in order.items])

        reply = f"订单已确认！\n\n"
        reply += f"订单号：{order.order_id}\n"
        reply += f"{items_text}\n"
        reply += f"──────────\n"
        reply += f"合计：¥{total:.0f}\n\n"
        reply += "支持微信、支付宝、Apple Pay\n"
        reply += "请稍候，您的饮品马上就好！"

        # 模拟制作
        order.status = "preparing"

        return {
            "reply": reply,
            "suggestions": ["再点一单", "查看订单"],
            "actions": ["new_order", "query_order"]
        }

    def _handle_complaint(self, session: Session) -> Dict:
        return {
            "reply": "非常抱歉给您带来不好的体验！我已经记录下您的反馈，会尽快为您处理。\n\n请问具体是什么问题呢？我们会尽力改进。",
            "suggestions": ["重新制作", "取消订单", "联系经理"],
            "actions": []
        }

    def _serialize_order(self, order: Order) -> Dict:
        """序列化订单"""
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
                    "price": item.get_price(),
                    "description": item.to_string()
                }
                for item in order.items
            ],
            "total": order.get_total(),
            "status": order.status,
            "created_at": order.created_at
        }

    def reset_session(self, session_id: str) -> Dict:
        """重置会话"""
        self.session_manager.delete_session(session_id)
        session = self.session_manager.create_session()
        session.add_message("assistant", "您好！欢迎光临，请问想喝点什么？")
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "history": session.history
        }
