"""订单数据模型"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

# 延迟导入避免循环依赖
def _get_menu_data():
    from data.menu import MENU, SIZE_PRICE, MILK_PRICE, EXTRAS_PRICE
    return MENU, SIZE_PRICE, MILK_PRICE, EXTRAS_PRICE


@dataclass
class OrderItem:
    """订单项"""
    product_name: str
    size: str = "中杯"
    temperature: str = "热"
    sweetness: str = "标准"
    milk_type: str = "全脂奶"
    extras: List[str] = field(default_factory=list)
    quantity: int = 1

    def get_price(self) -> float:
        MENU, SIZE_PRICE, MILK_PRICE, EXTRAS_PRICE = _get_menu_data()
        base = MENU.get(self.product_name, {}).get("price", 30)
        size_add = SIZE_PRICE.get(self.size, 0)
        milk_add = MILK_PRICE.get(self.milk_type, 0)
        extras_add = sum(EXTRAS_PRICE.get(e, 0) for e in self.extras)
        return (base + size_add + milk_add + extras_add) * self.quantity

    def to_string(self) -> str:
        parts = []
        if self.quantity > 1:
            parts.append(f"{self.quantity}杯")
        parts.append(self.size)
        parts.append(self.temperature)
        if self.sweetness != "标准":
            parts.append(self.sweetness)
        if self.milk_type != "全脂奶":
            parts.append(self.milk_type)
        parts.append(self.product_name)
        if self.extras:
            parts.append(f"加{'/'.join(self.extras)}")
        return "".join(parts)


@dataclass
class Order:
    """订单"""
    order_id: str
    items: List[OrderItem] = field(default_factory=list)
    status: str = "pending"  # pending, confirmed, preparing, ready, completed, cancelled
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_total(self) -> float:
        return sum(item.get_price() for item in self.items)
