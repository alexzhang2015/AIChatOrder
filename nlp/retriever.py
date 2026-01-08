"""向量检索器"""

from typing import Dict, List
import numpy as np


class SimpleVectorRetriever:
    """简化的向量检索器"""

    def __init__(self, examples: List[Dict]):
        self.examples = examples
        self.keywords = {
            'order': ['要', '来', '点', '给我', '帮我', '杯', '份'],
            'modify': ['换', '改', '加', '减', '不要', '少', '多'],
            'cancel': ['取消', '不要了', '算了', '不点'],
            'query': ['查', '到哪', '多久', '状态', '什么时候'],
            'info': ['多少钱', '价格', '什么', '有什么', '卡路里', '成分'],
            'recommend': ['推荐', '好喝', '建议', '适合'],
            'payment': ['支付', '付款', '优惠', '积分', '微信', '支付宝'],
            'complaint': ['投诉', '错了', '太久', '不满意', '差评'],
            'product': ['拿铁', '美式', '卡布奇诺', '摩卡', '星冰乐', '馥芮白', '抹茶']
        }

    def _extract_features(self, text: str) -> Dict[str, float]:
        features = {}
        for category, words in self.keywords.items():
            features[category] = sum(1 for w in words if w in text)
        return features

    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        keys = set(vec1.keys()) | set(vec2.keys())
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
        norm1 = np.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = np.sqrt(sum(v**2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_features = self._extract_features(query)
        scored_examples = []

        for example in self.examples:
            example_features = self._extract_features(example['text'])
            similarity = self._cosine_similarity(query_features, example_features)
            common_chars = len(set(query) & set(example['text']))
            char_bonus = common_chars * 0.02

            scored_examples.append({
                **example,
                'similarity': round(similarity + char_bonus, 4)
            })

        scored_examples.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_examples[:top_k]
