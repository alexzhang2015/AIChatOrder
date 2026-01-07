"""
Chroma 向量数据库封装

使用 Chroma 进行语义检索，替代基于关键词的 SimpleVectorRetriever。
支持检索结果缓存以提升性能。
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from exceptions import VectorStoreError, VectorStoreInitError, EmbeddingError
from cache import get_vector_cache, VectorCache

logger = logging.getLogger(__name__)

# 默认持久化目录
DEFAULT_CHROMA_PATH = Path(__file__).parent / "chroma_data"

# Chroma 可用性检查
CHROMA_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    logger.warning("chromadb 未安装，将使用降级的关键词检索器")


class ChromaRetriever:
    """基于 Chroma 的向量检索器

    使用 Chroma 内置的 embedding 模型进行语义检索。
    API 与 SimpleVectorRetriever 兼容。
    支持检索结果缓存以提升性能。
    """

    def __init__(
        self,
        examples: List[Dict],
        collection_name: str = "training_examples",
        persist_directory: Optional[Path] = None,
        cache: Optional[VectorCache] = None
    ):
        """初始化 Chroma 检索器

        Args:
            examples: 训练示例列表，每个示例包含 text, intent, slots
            collection_name: Chroma 集合名称
            persist_directory: 持久化目录
            cache: 可选的向量缓存实例
        """
        self.examples = examples
        self.collection_name = collection_name
        self.persist_directory = persist_directory or DEFAULT_CHROMA_PATH
        self._cache = cache or get_vector_cache()

        if not CHROMA_AVAILABLE:
            raise VectorStoreInitError("chromadb 未安装，请运行: pip install chromadb")

        try:
            # 确保目录存在
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # 初始化 Chroma 客户端
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )

            # 索引示例（如果集合为空）
            if self.collection.count() == 0:
                self._index_examples()
            else:
                logger.info(f"使用已有的向量索引，共 {self.collection.count()} 条记录")

        except Exception as e:
            raise VectorStoreInitError(f"Chroma 初始化失败: {e}")

    def _index_examples(self):
        """索引所有训练示例"""
        logger.info(f"开始索引 {len(self.examples)} 个训练示例...")

        try:
            documents = []
            metadatas = []
            ids = []

            for i, example in enumerate(self.examples):
                documents.append(example["text"])
                metadatas.append({
                    "intent": example["intent"],
                    "slots": json.dumps(example.get("slots", {}), ensure_ascii=False),
                    "index": i
                })
                ids.append(f"example_{i}")

            # 批量添加
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"索引完成，共 {len(documents)} 条记录")

        except Exception as e:
            raise EmbeddingError(f"索引失败: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索最相似的示例

        Args:
            query: 查询文本
            top_k: 返回的最大结果数

        Returns:
            相似示例列表，每个包含 text, intent, slots, similarity
        """
        # 检查缓存
        cached = self._cache.get(query, top_k)
        if cached is not None:
            return cached

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self.examples)),
                include=["documents", "metadatas", "distances"]
            )

            retrieved = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # 距离转换为相似度 (cosine distance -> similarity)
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # cosine distance to similarity

                    metadata = results["metadatas"][0][i]
                    slots = json.loads(metadata.get("slots", "{}"))

                    retrieved.append({
                        "text": results["documents"][0][i],
                        "intent": metadata["intent"],
                        "slots": slots,
                        "similarity": round(similarity, 4)
                    })

            # 缓存结果
            self._cache.set(query, top_k, retrieved)
            return retrieved

        except Exception as e:
            logger.error(f"检索失败: {e}")
            # 降级：返回空列表
            return []

    def add_example(self, text: str, intent: str, slots: Optional[Dict] = None):
        """动态添加新示例

        Args:
            text: 示例文本
            intent: 意图标签
            slots: 槽位信息
        """
        try:
            new_index = self.collection.count()
            self.collection.add(
                documents=[text],
                metadatas=[{
                    "intent": intent,
                    "slots": json.dumps(slots or {}, ensure_ascii=False),
                    "index": new_index
                }],
                ids=[f"example_{new_index}"]
            )

            # 同步更新内存中的示例
            self.examples.append({
                "text": text,
                "intent": intent,
                "slots": slots or {}
            })

            # 清空缓存（新示例可能影响检索结果）
            self._cache.invalidate_all()

            logger.debug(f"添加新示例: {text[:50]}...")

        except Exception as e:
            logger.error(f"添加示例失败: {e}")
            raise EmbeddingError(f"添加示例失败: {e}")

    def reset(self):
        """重置集合，重新索引"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._index_examples()
            # 清空缓存
            self._cache.invalidate_all()
            logger.info("向量索引已重置")
        except Exception as e:
            raise VectorStoreError(f"重置失败: {e}")

    def count(self) -> int:
        """返回索引的示例数量"""
        return self.collection.count()

    def cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return self._cache.stats()


class FallbackRetriever:
    """降级检索器（当 Chroma 不可用时使用）

    基于关键词匹配的简单检索器，与 SimpleVectorRetriever 类似。
    支持检索结果缓存。
    """

    def __init__(self, examples: List[Dict], cache: Optional[VectorCache] = None):
        self.examples = examples
        self._cache = cache or get_vector_cache()
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
        """提取文本特征"""
        features = {}
        for category, words in self.keywords.items():
            features[category] = sum(1 for w in words if w in text)
        return features

    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """计算余弦相似度"""
        import math
        keys = set(vec1.keys()) | set(vec2.keys())
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索最相似的示例"""
        # 检查缓存
        cached = self._cache.get(query, top_k)
        if cached is not None:
            return cached

        query_features = self._extract_features(query)
        scored_examples = []

        for example in self.examples:
            example_features = self._extract_features(example['text'])
            similarity = self._cosine_similarity(query_features, example_features)

            # 字符级别奖励
            common_chars = len(set(query) & set(example['text']))
            char_bonus = common_chars * 0.02

            scored_examples.append({
                **example,
                'similarity': round(similarity + char_bonus, 4)
            })

        scored_examples.sort(key=lambda x: x['similarity'], reverse=True)
        results = scored_examples[:top_k]

        # 缓存结果
        self._cache.set(query, top_k, results)
        return results

    def add_example(self, text: str, intent: str, slots: Optional[Dict] = None):
        """添加新示例"""
        self.examples.append({
            "text": text,
            "intent": intent,
            "slots": slots or {}
        })
        # 清空缓存
        self._cache.invalidate_all()

    def count(self) -> int:
        """返回示例数量"""
        return len(self.examples)

    def cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return self._cache.stats()


def create_retriever(
    examples: List[Dict],
    use_chroma: bool = True,
    collection_name: str = "training_examples",
    persist_directory: Optional[Path] = None
):
    """创建检索器的工厂函数

    Args:
        examples: 训练示例列表
        use_chroma: 是否使用 Chroma（否则使用降级检索器）
        collection_name: Chroma 集合名称
        persist_directory: Chroma 持久化目录

    Returns:
        ChromaRetriever 或 FallbackRetriever
    """
    if use_chroma and CHROMA_AVAILABLE:
        try:
            return ChromaRetriever(
                examples=examples,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        except VectorStoreInitError as e:
            logger.warning(f"Chroma 初始化失败，使用降级检索器: {e}")
            return FallbackRetriever(examples)
    else:
        if use_chroma and not CHROMA_AVAILABLE:
            logger.warning("chromadb 未安装，使用降级检索器")
        return FallbackRetriever(examples)


def is_chroma_available() -> bool:
    """检查 Chroma 是否可用"""
    return CHROMA_AVAILABLE
