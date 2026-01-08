"""会话数据模型"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from .order import Order


class ConversationState(str, Enum):
    """对话状态"""
    GREETING = "greeting"
    TAKING_ORDER = "taking_order"
    CONFIRMING = "confirming"
    MODIFYING = "modifying"
    PAYMENT = "payment"
    COMPLETED = "completed"


@dataclass
class Session:
    """会话"""
    session_id: str
    state: ConversationState = ConversationState.GREETING
    current_order: Optional[Order] = None
    pending_item: Optional[Dict] = None  # 正在收集信息的订单项
    history: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, intent_info: Dict = None):
        self.history.append({
            "role": role,
            "content": content,
            "intent_info": intent_info,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
