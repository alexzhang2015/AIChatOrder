"""槽位提取器"""

import re
from typing import Dict


class SlotExtractor:
    """槽位提取器"""

    def __init__(self):
        self.product_patterns = [
            (r'拿铁', '拿铁'), (r'美式', '美式咖啡'), (r'卡布奇诺', '卡布奇诺'),
            (r'摩卡', '摩卡'), (r'星冰乐', '星冰乐'), (r'馥芮白|澳白', '馥芮白'),
            (r'抹茶', '抹茶拿铁'), (r'焦糖玛奇朵', '焦糖玛奇朵'), (r'香草拿铁', '香草拿铁'),
        ]
        self.size_patterns = [(r'超大杯', '超大杯'), (r'大杯', '大杯'), (r'中杯', '中杯')]
        self.temperature_patterns = [
            (r'去冰', '去冰'), (r'少冰', '少冰'), (r'多冰', '多冰'),
            (r'冰', '冰'), (r'热', '热'), (r'温', '温'),
        ]
        self.sweetness_patterns = [
            (r'无糖', '无糖'), (r'三分糖', '三分糖'), (r'半糖', '半糖'),
            (r'少糖', '少糖'), (r'全糖', '全糖'),
        ]
        self.milk_patterns = [
            (r'燕麦奶', '燕麦奶'), (r'椰奶', '椰奶'), (r'豆奶', '豆奶'),
            (r'脱脂奶|脱脂', '脱脂奶'), (r'全脂奶|全脂', '全脂奶'),
        ]
        self.extras_patterns = [
            (r'浓缩|extra shot', '浓缩shot'), (r'香草糖浆', '香草糖浆'),
            (r'焦糖糖浆', '焦糖糖浆'), (r'奶油', '奶油'), (r'珍珠', '珍珠'),
        ]
        self.quantity_map = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5}

    def extract(self, text: str) -> Dict:
        slots = {}

        for pattern, name in self.product_patterns:
            if re.search(pattern, text):
                slots['product_name'] = name
                break

        for pattern, size in self.size_patterns:
            if re.search(pattern, text):
                slots['size'] = size
                break

        for pattern, temp in self.temperature_patterns:
            if re.search(pattern, text):
                slots['temperature'] = temp
                break

        for pattern, sweetness in self.sweetness_patterns:
            if re.search(pattern, text):
                slots['sweetness'] = sweetness
                break

        for pattern, milk in self.milk_patterns:
            if re.search(pattern, text):
                slots['milk_type'] = milk
                break

        extras = []
        for pattern, extra in self.extras_patterns:
            if re.search(pattern, text):
                extras.append(extra)
        if extras:
            slots['extras'] = extras

        quantity_match = re.search(r'([一二三四五六七八九十两]|[1-9])[份杯]', text)
        if quantity_match:
            q = quantity_match.group(1)
            qty = self.quantity_map.get(q, int(q) if q.isdigit() else 1)
            if qty != 1:
                slots['quantity'] = qty

        return slots
