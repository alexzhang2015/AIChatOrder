"""
生产数据收集器

从生产环境收集用户交互数据，用于持续优化模型
支持:
1. 低置信度样本收集
2. 用户反馈收集
3. 自动标注队列管理
4. 数据质量评估
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import random

logger = logging.getLogger(__name__)


class LabelStatus(str, Enum):
    """标注状态"""
    PENDING = "pending"           # 待标注
    AUTO_LABELED = "auto_labeled" # 自动标注
    HUMAN_LABELED = "human_labeled"  # 人工标注
    VERIFIED = "verified"         # 已验证
    REJECTED = "rejected"         # 拒绝（质量差）


class SamplePriority(str, Enum):
    """样本优先级"""
    HIGH = "high"       # 高优先级（低置信度、用户反馈）
    MEDIUM = "medium"   # 中优先级
    LOW = "low"         # 低优先级（高置信度）


@dataclass
class ProductionSample:
    """生产样本"""
    id: str
    timestamp: str
    session_id: str

    # 输入
    user_input: str
    conversation_history: List[Dict] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # 模型输出
    predicted_intent: str = ""
    predicted_confidence: float = 0.0
    predicted_slots: Dict[str, Any] = field(default_factory=dict)

    # 标注信息
    label_status: str = LabelStatus.PENDING.value
    labeled_intent: str = ""
    labeled_slots: Dict[str, Any] = field(default_factory=dict)
    labeler: str = ""
    labeled_at: str = ""

    # 用户反馈
    user_feedback: str = ""
    feedback_type: str = ""  # positive, negative, correction
    feedback_at: str = ""

    # 质量评估
    priority: str = SamplePriority.MEDIUM.value
    quality_score: float = 0.0
    is_edge_case: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductionSample":
        return cls(**data)


@dataclass
class LabelingTask:
    """标注任务"""
    task_id: str
    samples: List[str]  # Sample IDs
    created_at: str
    assigned_to: str = ""
    status: str = "pending"  # pending, in_progress, completed
    completed_at: str = ""
    notes: str = ""


class ProductionDataCollector:
    """生产数据收集器"""

    def __init__(
        self,
        storage_path: str = "evals/optimization/production_data",
        confidence_threshold: float = 0.7,
        auto_label_threshold: float = 0.9
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.confidence_threshold = confidence_threshold
        self.auto_label_threshold = auto_label_threshold

        self.samples: Dict[str, ProductionSample] = {}
        self.labeling_tasks: Dict[str, LabelingTask] = {}

        self._load()

    def _load(self):
        """加载已存储的数据"""
        samples_file = self.storage_path / "samples.json"
        if samples_file.exists():
            with open(samples_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.samples = {
                    s["id"]: ProductionSample.from_dict(s)
                    for s in data
                }
            logger.info(f"加载 {len(self.samples)} 个生产样本")

        tasks_file = self.storage_path / "labeling_tasks.json"
        if tasks_file.exists():
            with open(tasks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.labeling_tasks = {
                    t["task_id"]: LabelingTask(**t)
                    for t in data
                }

    def _save(self):
        """保存数据"""
        samples_file = self.storage_path / "samples.json"
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.samples.values()],
                     f, ensure_ascii=False, indent=2)

        tasks_file = self.storage_path / "labeling_tasks.json"
        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self.labeling_tasks.values()],
                     f, ensure_ascii=False, indent=2)

    def _generate_id(self, content: str) -> str:
        """生成样本 ID"""
        hash_input = f"{content}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def collect(
        self,
        user_input: str,
        session_id: str,
        predicted_intent: str,
        predicted_confidence: float,
        predicted_slots: Dict[str, Any] = None,
        conversation_history: List[Dict] = None,
        context: Dict[str, Any] = None
    ) -> ProductionSample:
        """
        收集生产样本

        Args:
            user_input: 用户输入
            session_id: 会话 ID
            predicted_intent: 预测意图
            predicted_confidence: 置信度
            predicted_slots: 预测槽位
            conversation_history: 对话历史
            context: 上下文信息

        Returns:
            ProductionSample 对象
        """
        sample_id = self._generate_id(f"{session_id}_{user_input}")

        # 确定优先级
        if predicted_confidence < self.confidence_threshold:
            priority = SamplePriority.HIGH
        elif predicted_confidence < self.auto_label_threshold:
            priority = SamplePriority.MEDIUM
        else:
            priority = SamplePriority.LOW

        # 确定标注状态
        if predicted_confidence >= self.auto_label_threshold:
            label_status = LabelStatus.AUTO_LABELED
            labeled_intent = predicted_intent
            labeled_slots = predicted_slots or {}
        else:
            label_status = LabelStatus.PENDING
            labeled_intent = ""
            labeled_slots = {}

        sample = ProductionSample(
            id=sample_id,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            user_input=user_input,
            conversation_history=conversation_history or [],
            context=context or {},
            predicted_intent=predicted_intent,
            predicted_confidence=predicted_confidence,
            predicted_slots=predicted_slots or {},
            label_status=label_status.value,
            labeled_intent=labeled_intent,
            labeled_slots=labeled_slots,
            priority=priority.value
        )

        self.samples[sample_id] = sample
        self._save()

        logger.debug(f"收集样本 {sample_id}: {user_input[:30]}... "
                    f"(置信度: {predicted_confidence:.2f}, 优先级: {priority.value})")

        return sample

    def add_user_feedback(
        self,
        sample_id: str,
        feedback: str,
        feedback_type: str = "negative"
    ):
        """
        添加用户反馈

        Args:
            sample_id: 样本 ID
            feedback: 反馈内容
            feedback_type: 反馈类型 (positive, negative, correction)
        """
        if sample_id not in self.samples:
            logger.warning(f"样本不存在: {sample_id}")
            return

        sample = self.samples[sample_id]
        sample.user_feedback = feedback
        sample.feedback_type = feedback_type
        sample.feedback_at = datetime.now().isoformat()

        # 有用户反馈的样本优先级提升
        if feedback_type in ["negative", "correction"]:
            sample.priority = SamplePriority.HIGH.value
            sample.label_status = LabelStatus.PENDING.value  # 需要重新标注

        self._save()

    def label_sample(
        self,
        sample_id: str,
        intent: str,
        slots: Dict[str, Any] = None,
        labeler: str = "human",
        is_edge_case: bool = False,
        notes: str = ""
    ):
        """
        标注样本

        Args:
            sample_id: 样本 ID
            intent: 标注意图
            slots: 标注槽位
            labeler: 标注者
            is_edge_case: 是否为边界案例
            notes: 备注
        """
        if sample_id not in self.samples:
            logger.warning(f"样本不存在: {sample_id}")
            return

        sample = self.samples[sample_id]
        sample.labeled_intent = intent
        sample.labeled_slots = slots or {}
        sample.labeler = labeler
        sample.labeled_at = datetime.now().isoformat()
        sample.label_status = LabelStatus.HUMAN_LABELED.value
        sample.is_edge_case = is_edge_case
        sample.notes = notes

        # 计算质量分数
        sample.quality_score = self._calculate_quality_score(sample)

        self._save()

    def _calculate_quality_score(self, sample: ProductionSample) -> float:
        """计算样本质量分数"""
        score = 0.5  # 基础分

        # 有人工标注加分
        if sample.label_status == LabelStatus.HUMAN_LABELED.value:
            score += 0.2

        # 有用户反馈加分
        if sample.user_feedback:
            score += 0.1

        # 边界案例加分
        if sample.is_edge_case:
            score += 0.1

        # 预测正确加分（如果已标注）
        if sample.labeled_intent and sample.predicted_intent == sample.labeled_intent:
            score += 0.1

        return min(score, 1.0)

    def get_pending_samples(
        self,
        limit: int = 50,
        priority: str = None
    ) -> List[ProductionSample]:
        """获取待标注样本"""
        pending = [
            s for s in self.samples.values()
            if s.label_status == LabelStatus.PENDING.value
        ]

        if priority:
            pending = [s for s in pending if s.priority == priority]

        # 按优先级排序
        priority_order = {
            SamplePriority.HIGH.value: 0,
            SamplePriority.MEDIUM.value: 1,
            SamplePriority.LOW.value: 2
        }
        pending.sort(key=lambda s: (priority_order.get(s.priority, 2), s.timestamp))

        return pending[:limit]

    def get_labeled_samples(
        self,
        intent: str = None,
        min_quality: float = 0.0,
        limit: int = 100
    ) -> List[ProductionSample]:
        """获取已标注样本"""
        labeled = [
            s for s in self.samples.values()
            if s.label_status in [
                LabelStatus.HUMAN_LABELED.value,
                LabelStatus.VERIFIED.value
            ]
            and s.quality_score >= min_quality
        ]

        if intent:
            labeled = [s for s in labeled if s.labeled_intent == intent]

        return labeled[:limit]

    def get_edge_cases(self, limit: int = 50) -> List[ProductionSample]:
        """获取边界案例"""
        edge_cases = [
            s for s in self.samples.values()
            if s.is_edge_case
        ]
        return edge_cases[:limit]

    def create_labeling_task(
        self,
        sample_ids: List[str],
        assigned_to: str = ""
    ) -> LabelingTask:
        """创建标注任务"""
        task_id = self._generate_id(f"task_{datetime.now().isoformat()}")

        task = LabelingTask(
            task_id=task_id,
            samples=sample_ids,
            created_at=datetime.now().isoformat(),
            assigned_to=assigned_to
        )

        self.labeling_tasks[task_id] = task
        self._save()

        return task

    def export_for_training(
        self,
        output_path: str,
        min_quality: float = 0.5,
        include_auto_labeled: bool = True,
        format: str = "jsonl"
    ) -> str:
        """
        导出训练数据

        Args:
            output_path: 输出路径
            min_quality: 最低质量分数
            include_auto_labeled: 是否包含自动标注
            format: 输出格式 (jsonl, json)

        Returns:
            输出文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        valid_statuses = [LabelStatus.HUMAN_LABELED.value, LabelStatus.VERIFIED.value]
        if include_auto_labeled:
            valid_statuses.append(LabelStatus.AUTO_LABELED.value)

        samples = [
            s for s in self.samples.values()
            if s.label_status in valid_statuses
            and s.quality_score >= min_quality
        ]

        if format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in samples:
                    training_example = {
                        "text": sample.user_input,
                        "intent": sample.labeled_intent or sample.predicted_intent,
                        "slots": sample.labeled_slots or sample.predicted_slots,
                        "source": "production",
                        "quality_score": sample.quality_score,
                        "is_edge_case": sample.is_edge_case
                    }
                    f.write(json.dumps(training_example, ensure_ascii=False) + "\n")

        elif format == "json":
            data = [
                {
                    "text": s.user_input,
                    "intent": s.labeled_intent or s.predicted_intent,
                    "slots": s.labeled_slots or s.predicted_slots,
                    "source": "production",
                    "quality_score": s.quality_score,
                    "is_edge_case": s.is_edge_case
                }
                for s in samples
            ]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"导出 {len(samples)} 个训练样本到 {output_file}")
        return str(output_file)

    def export_fine_tuning_data(
        self,
        output_path: str,
        min_quality: float = 0.7
    ) -> str:
        """导出 OpenAI Fine-tuning 格式数据"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        samples = [
            s for s in self.samples.values()
            if s.label_status in [LabelStatus.HUMAN_LABELED.value, LabelStatus.VERIFIED.value]
            and s.quality_score >= min_quality
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是咖啡店点单助手，识别用户意图和提取槽位信息。"
                        },
                        {
                            "role": "user",
                            "content": sample.user_input
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps({
                                "intent": sample.labeled_intent,
                                "confidence": 0.95,
                                "slots": sample.labeled_slots,
                                "reasoning": sample.notes or ""
                            }, ensure_ascii=False)
                        }
                    ]
                }
                f.write(json.dumps(training_example, ensure_ascii=False) + "\n")

        logger.info(f"导出 {len(samples)} 个 fine-tuning 样本到 {output_file}")
        return str(output_file)

    def generate_statistics(self) -> Dict[str, Any]:
        """生成统计信息"""
        total = len(self.samples)
        if total == 0:
            return {"message": "暂无数据"}

        stats = {
            "total_samples": total,
            "by_label_status": {},
            "by_priority": {},
            "by_intent": {},
            "average_confidence": 0.0,
            "average_quality": 0.0,
            "edge_cases_count": 0,
            "with_feedback": 0
        }

        confidence_sum = 0.0
        quality_sum = 0.0
        labeled_count = 0

        for sample in self.samples.values():
            # 按标注状态统计
            stats["by_label_status"][sample.label_status] = \
                stats["by_label_status"].get(sample.label_status, 0) + 1

            # 按优先级统计
            stats["by_priority"][sample.priority] = \
                stats["by_priority"].get(sample.priority, 0) + 1

            # 按意图统计
            intent = sample.labeled_intent or sample.predicted_intent
            if intent:
                stats["by_intent"][intent] = \
                    stats["by_intent"].get(intent, 0) + 1

            # 置信度
            confidence_sum += sample.predicted_confidence

            # 质量分数（仅已标注）
            if sample.label_status in [LabelStatus.HUMAN_LABELED.value, LabelStatus.VERIFIED.value]:
                quality_sum += sample.quality_score
                labeled_count += 1

            # 边界案例
            if sample.is_edge_case:
                stats["edge_cases_count"] += 1

            # 用户反馈
            if sample.user_feedback:
                stats["with_feedback"] += 1

        stats["average_confidence"] = confidence_sum / total if total > 0 else 0
        stats["average_quality"] = quality_sum / labeled_count if labeled_count > 0 else 0

        return stats

    def print_dashboard(self):
        """打印数据收集仪表盘"""
        stats = self.generate_statistics()

        print("\n" + "=" * 60)
        print("生产数据收集仪表盘")
        print("=" * 60)

        if "message" in stats:
            print(stats["message"])
            return

        print(f"\n总样本数: {stats['total_samples']}")
        print(f"边界案例: {stats['edge_cases_count']}")
        print(f"有用户反馈: {stats['with_feedback']}")
        print(f"平均置信度: {stats['average_confidence']:.2f}")
        print(f"平均质量分: {stats['average_quality']:.2f}")

        print("\n标注状态分布:")
        for status, count in stats["by_label_status"].items():
            pct = count / stats["total_samples"] * 100
            print(f"  {status}: {count} ({pct:.1f}%)")

        print("\n优先级分布:")
        for priority, count in stats["by_priority"].items():
            pct = count / stats["total_samples"] * 100
            print(f"  {priority}: {count} ({pct:.1f}%)")

        print("\n意图分布 (Top 5):")
        sorted_intents = sorted(stats["by_intent"].items(),
                               key=lambda x: x[1], reverse=True)[:5]
        for intent, count in sorted_intents:
            pct = count / stats["total_samples"] * 100
            print(f"  {intent}: {count} ({pct:.1f}%)")

        print("=" * 60)


def simulate_production_data():
    """模拟生产数据收集"""
    collector = ProductionDataCollector()

    # 模拟生产数据
    test_inputs = [
        ("来杯拿铁", "ORDER_NEW", 0.95, {"product_name": "拿铁"}),
        ("美式咖啡", "ORDER_NEW", 0.92, {"product_name": "美式咖啡"}),
        ("有什么推荐的", "RECOMMEND", 0.88, {}),
        ("算了不要了", "ORDER_CANCEL", 0.65, {}),  # 低置信度
        ("换成大杯", "ORDER_MODIFY", 0.85, {"size": "大杯"}),
        ("续命水来一杯", "ORDER_NEW", 0.72, {"product_name": "美式咖啡"}),  # 中等置信度
        ("这个怎么样", "PRODUCT_INFO", 0.55, {}),  # 低置信度
        ("结账", "PAYMENT", 0.93, {}),
        ("再来一个", "ORDER_NEW", 0.68, {}),  # 低置信度
        ("不点了不点了", "ORDER_CANCEL", 0.82, {}),
    ]

    print("模拟收集生产数据...")
    for user_input, intent, confidence, slots in test_inputs:
        sample = collector.collect(
            user_input=user_input,
            session_id=f"session_{random.randint(1000, 9999)}",
            predicted_intent=intent,
            predicted_confidence=confidence,
            predicted_slots=slots
        )
        print(f"  收集: {user_input} -> {intent} (置信度: {confidence:.2f})")

    # 模拟用户反馈
    pending = collector.get_pending_samples(limit=3)
    for sample in pending:
        collector.add_user_feedback(
            sample.id,
            feedback="识别不准确",
            feedback_type="negative"
        )

    # 模拟人工标注
    for sample in pending[:2]:
        collector.label_sample(
            sample.id,
            intent=sample.predicted_intent,
            labeler="human_annotator",
            is_edge_case=sample.predicted_confidence < 0.7,
            notes="已核实"
        )

    # 打印统计
    collector.print_dashboard()

    # 导出训练数据
    output_dir = Path("evals/optimization/production_data")
    training_file = collector.export_for_training(
        output_dir / "training_export.jsonl"
    )
    print(f"\n训练数据导出: {training_file}")


if __name__ == "__main__":
    simulate_production_data()
