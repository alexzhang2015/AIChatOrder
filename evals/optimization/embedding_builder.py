"""
Embedding 知识库构建器

构建多层次的向量知识库，支持:
1. 商品知识库
2. 意图示例库
3. 对话历史库
4. Bad Case 修复库
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingDocument:
    """Embedding 文档"""
    id: str
    text: str
    layer: str  # product, intent, dialogue, fixed_case
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EmbeddingLayer:
    """Embedding 层"""
    name: str
    description: str
    documents: List[EmbeddingDocument] = field(default_factory=list)
    priority: int = 1  # 检索时的优先级权重


class EmbeddingKnowledgeBase:
    """多层次 Embedding 知识库"""

    def __init__(self, storage_path: str = "evals/optimization/embedding_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 初始化四个层次
        self.layers: Dict[str, EmbeddingLayer] = {
            "product": EmbeddingLayer(
                name="product",
                description="商品知识库 - 商品名称、别名、描述",
                priority=3
            ),
            "intent": EmbeddingLayer(
                name="intent",
                description="意图示例库 - 各意图的典型表达",
                priority=2
            ),
            "dialogue": EmbeddingLayer(
                name="dialogue",
                description="对话历史库 - 成功对话案例",
                priority=1
            ),
            "fixed_case": EmbeddingLayer(
                name="fixed_case",
                description="Bad Case 修复库 - 已修复的边界案例",
                priority=4  # 最高优先级
            )
        }

        self._load()

    def _load(self):
        """加载已存储的知识库"""
        for layer_name in self.layers:
            layer_file = self.storage_path / f"{layer_name}_layer.json"
            if layer_file.exists():
                with open(layer_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents = [EmbeddingDocument(**doc) for doc in data]
                    self.layers[layer_name].documents = documents
                logger.info(f"加载 {layer_name} 层: {len(documents)} 条文档")

    def _save_layer(self, layer_name: str):
        """保存单个层"""
        layer_file = self.storage_path / f"{layer_name}_layer.json"
        documents = [asdict(doc) for doc in self.layers[layer_name].documents]
        # 移除 embedding 以减少存储大小
        for doc in documents:
            doc.pop("embedding", None)
        with open(layer_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

    def _generate_id(self, text: str, layer: str) -> str:
        """生成文档 ID"""
        hash_input = f"{layer}_{text}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    # ========== 商品知识层 ==========

    def add_product(
        self,
        product_name: str,
        description: str = "",
        aliases: List[str] = None,
        category: str = "",
        features: Dict[str, Any] = None
    ):
        """添加商品到知识库"""
        # 主文档
        main_doc = EmbeddingDocument(
            id=self._generate_id(product_name, "product"),
            text=f"{product_name}。{description}" if description else product_name,
            layer="product",
            metadata={
                "type": "product",
                "product_name": product_name,
                "category": category,
                "features": features or {},
                "is_canonical": True
            }
        )
        self._add_to_layer("product", main_doc)

        # 别名文档
        for alias in (aliases or []):
            alias_doc = EmbeddingDocument(
                id=self._generate_id(f"{alias}={product_name}", "product"),
                text=f"{alias}就是{product_name}",
                layer="product",
                metadata={
                    "type": "product_alias",
                    "product_name": product_name,
                    "alias": alias,
                    "is_canonical": False
                }
            )
            self._add_to_layer("product", alias_doc)

    def add_product_knowledge(
        self,
        product_name: str,
        knowledge_type: str,  # price, calories, ingredients, etc.
        content: str
    ):
        """添加商品知识"""
        doc = EmbeddingDocument(
            id=self._generate_id(f"{product_name}_{knowledge_type}", "product"),
            text=f"{product_name}的{knowledge_type}：{content}",
            layer="product",
            metadata={
                "type": "product_knowledge",
                "product_name": product_name,
                "knowledge_type": knowledge_type
            }
        )
        self._add_to_layer("product", doc)

    # ========== 意图示例层 ==========

    def add_intent_example(
        self,
        text: str,
        intent: str,
        slots: Dict[str, Any] = None,
        confidence: float = 1.0,
        is_edge_case: bool = False,
        note: str = ""
    ):
        """添加意图示例"""
        doc = EmbeddingDocument(
            id=self._generate_id(text, "intent"),
            text=text,
            layer="intent",
            metadata={
                "type": "intent_example",
                "intent": intent,
                "slots": slots or {},
                "confidence": confidence,
                "is_edge_case": is_edge_case,
                "note": note
            }
        )
        self._add_to_layer("intent", doc)

    def add_intent_boundary(
        self,
        text: str,
        correct_intent: str,
        wrong_intents: List[str],
        explanation: str
    ):
        """添加意图边界案例（容易混淆的）"""
        doc = EmbeddingDocument(
            id=self._generate_id(text, "intent_boundary"),
            text=text,
            layer="intent",
            metadata={
                "type": "intent_boundary",
                "intent": correct_intent,
                "wrong_intents": wrong_intents,
                "explanation": explanation,
                "is_edge_case": True,
                "priority": "high"
            }
        )
        self._add_to_layer("intent", doc)

    # ========== 对话历史层 ==========

    def add_dialogue(
        self,
        turns: List[Dict[str, str]],
        success: bool = True,
        products: List[str] = None,
        intents: List[str] = None,
        session_id: str = ""
    ):
        """添加对话历史"""
        # 将对话转为单一文本
        dialogue_text = " | ".join([
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in turns
        ])

        doc = EmbeddingDocument(
            id=self._generate_id(session_id or dialogue_text[:50], "dialogue"),
            text=dialogue_text,
            layer="dialogue",
            metadata={
                "type": "dialogue",
                "turns": len(turns),
                "success": success,
                "products": products or [],
                "intents": intents or [],
                "session_id": session_id
            }
        )
        self._add_to_layer("dialogue", doc)

    # ========== Bad Case 修复层 ==========

    def add_fixed_case(
        self,
        text: str,
        original_intent: str,
        correct_intent: str,
        original_slots: Dict[str, Any] = None,
        correct_slots: Dict[str, Any] = None,
        root_cause: str = "",
        fix_description: str = "",
        badcase_id: str = ""
    ):
        """添加已修复的 Bad Case"""
        doc = EmbeddingDocument(
            id=self._generate_id(badcase_id or text, "fixed_case"),
            text=text,
            layer="fixed_case",
            metadata={
                "type": "fixed_case",
                "original_intent": original_intent,
                "correct_intent": correct_intent,
                "original_slots": original_slots or {},
                "correct_slots": correct_slots or {},
                "root_cause": root_cause,
                "fix_description": fix_description,
                "badcase_id": badcase_id,
                "is_important": True
            }
        )
        self._add_to_layer("fixed_case", doc)

    # ========== 通用方法 ==========

    def _add_to_layer(self, layer_name: str, document: EmbeddingDocument):
        """添加文档到指定层"""
        layer = self.layers[layer_name]

        # 检查是否已存在
        existing_ids = {doc.id for doc in layer.documents}
        if document.id not in existing_ids:
            layer.documents.append(document)
            self._save_layer(layer_name)
            logger.debug(f"添加文档到 {layer_name}: {document.id}")

    def get_all_documents(self) -> List[EmbeddingDocument]:
        """获取所有文档"""
        all_docs = []
        for layer in self.layers.values():
            all_docs.extend(layer.documents)
        return all_docs

    def get_layer_documents(self, layer_name: str) -> List[EmbeddingDocument]:
        """获取指定层的文档"""
        return self.layers.get(layer_name, EmbeddingLayer("", "")).documents

    def search(
        self,
        query: str,
        top_k: int = 5,
        layers: List[str] = None,
        filters: Dict[str, Any] = None
    ) -> List[Tuple[EmbeddingDocument, float]]:
        """
        搜索相似文档

        注意: 此方法需要与向量存储集成
        当前返回基于关键词的简单匹配结果
        """
        results = []
        target_layers = layers or list(self.layers.keys())

        for layer_name in target_layers:
            layer = self.layers[layer_name]
            for doc in layer.documents:
                # 简单的关键词匹配评分
                score = self._simple_similarity(query, doc.text)

                # 应用过滤器
                if filters:
                    if not self._match_filters(doc, filters):
                        continue

                # 应用层优先级
                score *= layer.priority

                if score > 0:
                    results.append((doc, score))

        # 排序并返回 top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _simple_similarity(self, query: str, text: str) -> float:
        """简单的相似度计算（关键词匹配）"""
        query_chars = set(query)
        text_chars = set(text)
        common = len(query_chars & text_chars)
        total = len(query_chars | text_chars)
        return common / total if total > 0 else 0

    def _match_filters(self, doc: EmbeddingDocument, filters: Dict[str, Any]) -> bool:
        """检查文档是否匹配过滤器"""
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            if isinstance(value, list):
                if doc.metadata[key] not in value:
                    return False
            elif doc.metadata[key] != value:
                return False
        return True

    def export_for_chroma(self, output_path: str) -> str:
        """导出为 Chroma 兼容格式"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "documents": [],
            "metadatas": [],
            "ids": []
        }

        for doc in self.get_all_documents():
            export_data["documents"].append(doc.text)
            export_data["metadatas"].append({
                **doc.metadata,
                "layer": doc.layer,
                "created_at": doc.created_at
            })
            export_data["ids"].append(doc.id)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return str(output_file)

    def export_for_fine_tuning(
        self,
        output_path: str,
        format: str = "openai"
    ) -> str:
        """导出为微调格式"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "openai":
            # OpenAI Fine-tuning JSONL 格式
            with open(output_file, "w", encoding="utf-8") as f:
                for doc in self.get_layer_documents("intent"):
                    if doc.metadata.get("type") == "intent_example":
                        training_example = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "你是咖啡店点单助手，识别用户意图和提取槽位信息。"
                                },
                                {
                                    "role": "user",
                                    "content": doc.text
                                },
                                {
                                    "role": "assistant",
                                    "content": json.dumps({
                                        "intent": doc.metadata.get("intent", "UNKNOWN"),
                                        "confidence": doc.metadata.get("confidence", 0.9),
                                        "slots": doc.metadata.get("slots", {}),
                                        "reasoning": doc.metadata.get("note", "")
                                    }, ensure_ascii=False)
                                }
                            ]
                        }
                        f.write(json.dumps(training_example, ensure_ascii=False) + "\n")

        return str(output_file)

    def generate_statistics(self) -> Dict[str, Any]:
        """生成知识库统计"""
        stats = {
            "total_documents": 0,
            "by_layer": {},
            "by_type": {}
        }

        for layer_name, layer in self.layers.items():
            layer_count = len(layer.documents)
            stats["total_documents"] += layer_count
            stats["by_layer"][layer_name] = {
                "count": layer_count,
                "priority": layer.priority,
                "description": layer.description
            }

            # 统计文档类型
            for doc in layer.documents:
                doc_type = doc.metadata.get("type", "unknown")
                stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1

        return stats

    def print_summary(self):
        """打印知识库摘要"""
        stats = self.generate_statistics()

        print("\n" + "=" * 60)
        print("Embedding 知识库摘要")
        print("=" * 60)

        print(f"\n总文档数: {stats['total_documents']}")

        print("\n各层统计:")
        for layer_name, layer_stats in stats["by_layer"].items():
            print(f"  {layer_name}:")
            print(f"    - 文档数: {layer_stats['count']}")
            print(f"    - 优先级: {layer_stats['priority']}")
            print(f"    - 描述: {layer_stats['description']}")

        print("\n文档类型分布:")
        for doc_type, count in sorted(stats["by_type"].items()):
            print(f"  {doc_type}: {count}")

        print("=" * 60)


def build_default_knowledge_base() -> EmbeddingKnowledgeBase:
    """构建默认知识库"""
    kb = EmbeddingKnowledgeBase()

    # 1. 添加商品知识
    products = [
        {
            "name": "美式咖啡",
            "description": "浓缩咖啡加水，简单纯粹的咖啡风味",
            "aliases": ["美式", "Americano", "冰美式", "热美式", "续命水", "续命咖啡", "黑咖啡"],
            "category": "espresso_based"
        },
        {
            "name": "拿铁",
            "description": "经典意式浓缩咖啡与蒸奶的完美结合，口感丝滑",
            "aliases": ["Latte", "拿铁咖啡", "冰拿铁", "热拿铁", "奶咖"],
            "category": "espresso_based"
        },
        {
            "name": "卡布奇诺",
            "description": "浓缩咖啡、蒸奶和奶泡的经典意式组合",
            "aliases": ["Cappuccino", "卡布"],
            "category": "espresso_based"
        },
        {
            "name": "摩卡",
            "description": "浓缩咖啡、巧克力和蒸奶的完美融合",
            "aliases": ["Mocha", "摩卡咖啡", "巧克力咖啡"],
            "category": "mocha_based"
        },
        {
            "name": "星冰乐",
            "description": "冰爽混合饮品，清凉解暑",
            "aliases": ["Frappuccino", "冰沙", "肥宅快乐水"],
            "category": "blended"
        },
        {
            "name": "馥芮白",
            "description": "澳洲风格，浓缩咖啡与丝滑奶泡",
            "aliases": ["Flat White", "澳白", "澳瑞白", "dirty"],
            "category": "espresso_based"
        }
    ]

    for product in products:
        kb.add_product(
            product_name=product["name"],
            description=product["description"],
            aliases=product["aliases"],
            category=product["category"]
        )

    # 2. 添加意图示例
    intent_examples = [
        # ORDER_NEW
        ("我要一杯拿铁", "ORDER_NEW", {"product_name": "拿铁"}),
        ("来杯美式咖啡", "ORDER_NEW", {"product_name": "美式咖啡"}),
        ("给我一杯大杯冰星冰乐", "ORDER_NEW", {"product_name": "星冰乐", "size": "大杯", "temperature": "冰"}),
        ("两杯卡布奇诺", "ORDER_NEW", {"product_name": "卡布奇诺", "quantity": 2}),
        ("来个续命水", "ORDER_NEW", {"product_name": "美式咖啡"}),

        # ORDER_MODIFY
        ("换成大杯", "ORDER_MODIFY", {"size": "大杯"}),
        ("改成冰的", "ORDER_MODIFY", {"temperature": "冰"}),
        ("换燕麦奶", "ORDER_MODIFY", {"milk_type": "燕麦奶"}),
        ("加一份浓缩", "ORDER_MODIFY", {"extras": ["浓缩shot"]}),
        ("少糖", "ORDER_MODIFY", {"sweetness": "少糖"}),

        # ORDER_CANCEL
        ("取消订单", "ORDER_CANCEL", {}),
        ("不要了", "ORDER_CANCEL", {}),
        ("算了", "ORDER_CANCEL", {}),
        ("不点了", "ORDER_CANCEL", {}),

        # RECOMMEND
        ("有什么推荐的", "RECOMMEND", {}),
        ("什么好喝", "RECOMMEND", {}),
        ("推荐一下", "RECOMMEND", {}),

        # PRODUCT_INFO
        ("拿铁多少钱", "PRODUCT_INFO", {"product_name": "拿铁"}),
        ("美式有多少卡路里", "PRODUCT_INFO", {"product_name": "美式咖啡"}),

        # CHITCHAT
        ("你好", "CHITCHAT", {}),
        ("谢谢", "CHITCHAT", {}),
        ("好的", "CHITCHAT", {}),

        # PAYMENT
        ("结账", "PAYMENT", {}),
        ("买单", "PAYMENT", {}),
    ]

    for text, intent, slots in intent_examples:
        kb.add_intent_example(text=text, intent=intent, slots=slots)

    # 3. 添加边界案例
    boundaries = [
        {
            "text": "有什么推荐的咖啡吗",
            "correct_intent": "RECOMMEND",
            "wrong_intents": ["ORDER_NEW"],
            "explanation": "询问推荐，不是直接下单"
        },
        {
            "text": "算了还是不喝了",
            "correct_intent": "ORDER_CANCEL",
            "wrong_intents": ["CHITCHAT"],
            "explanation": "隐式取消表达"
        },
        {
            "text": "换一杯拿铁",
            "correct_intent": "ORDER_MODIFY",
            "wrong_intents": ["ORDER_NEW"],
            "explanation": "替换当前订单，不是新订单"
        },
        {
            "text": "再来一杯",
            "correct_intent": "ORDER_NEW",
            "wrong_intents": ["ORDER_MODIFY"],
            "explanation": "追加新订单"
        }
    ]

    for boundary in boundaries:
        kb.add_intent_boundary(**boundary)

    return kb


def main():
    """测试知识库构建"""
    print("构建默认知识库...")
    kb = build_default_knowledge_base()

    # 打印摘要
    kb.print_summary()

    # 导出数据
    output_dir = Path("evals/optimization/embedding_store")

    chroma_file = kb.export_for_chroma(output_dir / "chroma_export.json")
    print(f"\nChroma 格式导出: {chroma_file}")

    fine_tuning_file = kb.export_for_fine_tuning(output_dir / "fine_tuning.jsonl")
    print(f"Fine-tuning 格式导出: {fine_tuning_file}")

    # 测试搜索
    print("\n测试搜索:")
    test_queries = [
        "续命咖啡",
        "取消订单",
        "有什么好喝的推荐"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = kb.search(query, top_k=3)
        for doc, score in results:
            print(f"  [{doc.layer}] {doc.text[:50]}... (score: {score:.3f})")


if __name__ == "__main__":
    main()
