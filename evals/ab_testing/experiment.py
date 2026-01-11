"""
A/B 测试实验框架
A/B Testing Experiment Framework

定义和管理 A/B 测试实验
"""

import uuid
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import yaml
import random


class ExperimentStatus(str, Enum):
    """实验状态"""
    DRAFT = "draft"           # 草稿
    SCHEDULED = "scheduled"   # 已计划
    RUNNING = "running"       # 运行中
    PAUSED = "paused"         # 已暂停
    COMPLETED = "completed"   # 已完成
    ABORTED = "aborted"       # 已中止


class AllocationStrategy(str, Enum):
    """流量分配策略"""
    RANDOM = "random"           # 随机分配
    USER_ID_HASH = "user_hash"  # 基于用户ID哈希
    SESSION_HASH = "session_hash"  # 基于会话哈希
    DETERMINISTIC = "deterministic"  # 确定性分配（用于测试）


@dataclass
class Variant:
    """实验变体"""
    id: str
    name: str
    description: str = ""
    weight: float = 0.5  # 流量权重 (0-1)
    config: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "config": self.config,
            "is_control": self.is_control
        }


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    description: str = ""
    metric_type: str = "ratio"  # ratio, mean, count
    higher_is_better: bool = True
    minimum_detectable_effect: float = 0.05  # 最小可检测效应
    is_primary: bool = False
    is_guardrail: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "metric_type": self.metric_type,
            "higher_is_better": self.higher_is_better,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "is_primary": self.is_primary,
            "is_guardrail": self.is_guardrail
        }


@dataclass
class StratificationRule:
    """分层规则"""
    field: str  # 分层字段
    values: List[str]  # 分层值
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "values": self.values,
            "description": self.description
        }


@dataclass
class ExperimentResult:
    """单个变体的实验结果"""
    variant_id: str
    sample_size: int
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, tuple]
    raw_data: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ABExperiment:
    """
    A/B 测试实验

    定义实验的所有配置，包括：
    - 变体定义
    - 指标定义
    - 流量分配
    - 运行时间
    """
    id: str
    name: str
    description: str = ""
    hypothesis: str = ""  # 假设

    # 变体
    variants: List[Variant] = field(default_factory=list)

    # 指标
    primary_metrics: List[MetricDefinition] = field(default_factory=list)
    secondary_metrics: List[MetricDefinition] = field(default_factory=list)
    guardrail_metrics: List[MetricDefinition] = field(default_factory=list)

    # 流量分配
    allocation_strategy: AllocationStrategy = AllocationStrategy.USER_ID_HASH
    traffic_percentage: float = 1.0  # 参与实验的总流量百分比

    # 分层
    stratification: List[StratificationRule] = field(default_factory=list)

    # 时间
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_runtime_days: int = 7
    max_runtime_days: int = 30

    # 样本量
    min_sample_per_variant: int = 100
    target_sample_per_variant: int = 1000

    # 状态
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 结果
    results: Dict[str, ExperimentResult] = field(default_factory=dict)

    # 元数据
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"exp_{uuid.uuid4().hex[:8]}"

    @classmethod
    def create(
        cls,
        name: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        primary_metric: str,
        **kwargs
    ) -> "ABExperiment":
        """
        便捷创建方法

        Args:
            name: 实验名称
            control_config: 对照组配置
            treatment_config: 实验组配置
            primary_metric: 主要指标名称

        Returns:
            ABExperiment
        """
        control = Variant(
            id="control",
            name="对照组",
            config=control_config,
            weight=0.5,
            is_control=True
        )
        treatment = Variant(
            id="treatment",
            name="实验组",
            config=treatment_config,
            weight=0.5,
            is_control=False
        )

        primary = MetricDefinition(
            name=primary_metric,
            is_primary=True
        )

        return cls(
            id=kwargs.get("id", ""),
            name=name,
            variants=[control, treatment],
            primary_metrics=[primary],
            **{k: v for k, v in kwargs.items() if k != "id"}
        )

    def get_control(self) -> Optional[Variant]:
        """获取对照组"""
        for v in self.variants:
            if v.is_control:
                return v
        return self.variants[0] if self.variants else None

    def get_treatment(self) -> Optional[Variant]:
        """获取实验组"""
        for v in self.variants:
            if not v.is_control:
                return v
        return self.variants[1] if len(self.variants) > 1 else None

    def allocate(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **context
    ) -> Variant:
        """
        分配变体

        Args:
            user_id: 用户ID
            session_id: 会话ID
            **context: 额外上下文

        Returns:
            分配的变体
        """
        if self.status != ExperimentStatus.RUNNING:
            # 实验未运行，返回对照组
            return self.get_control() or self.variants[0]

        # 检查是否参与实验
        if random.random() > self.traffic_percentage:
            return self.get_control() or self.variants[0]

        # 根据策略分配
        if self.allocation_strategy == AllocationStrategy.RANDOM:
            return self._allocate_random()
        elif self.allocation_strategy == AllocationStrategy.USER_ID_HASH:
            return self._allocate_by_hash(user_id or session_id or str(uuid.uuid4()))
        elif self.allocation_strategy == AllocationStrategy.SESSION_HASH:
            return self._allocate_by_hash(session_id or str(uuid.uuid4()))
        else:
            return self._allocate_random()

    def _allocate_random(self) -> Variant:
        """随机分配"""
        r = random.random()
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if r < cumulative:
                return variant
        return self.variants[-1]

    def _allocate_by_hash(self, key: str) -> Variant:
        """基于哈希分配（确保同一用户总是分到同一组）"""
        hash_input = f"{self.id}:{key}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant
        return self.variants[-1]

    def start(self):
        """启动实验"""
        if self.status not in [ExperimentStatus.DRAFT, ExperimentStatus.SCHEDULED]:
            raise ValueError(f"Cannot start experiment in {self.status} status")

        self.status = ExperimentStatus.RUNNING
        self.start_date = datetime.now()
        self.updated_at = datetime.now()

    def pause(self):
        """暂停实验"""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot pause experiment in {self.status} status")

        self.status = ExperimentStatus.PAUSED
        self.updated_at = datetime.now()

    def resume(self):
        """恢复实验"""
        if self.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Cannot resume experiment in {self.status} status")

        self.status = ExperimentStatus.RUNNING
        self.updated_at = datetime.now()

    def complete(self):
        """完成实验"""
        if self.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            raise ValueError(f"Cannot complete experiment in {self.status} status")

        self.status = ExperimentStatus.COMPLETED
        self.end_date = datetime.now()
        self.updated_at = datetime.now()

    def abort(self, reason: str = ""):
        """中止实验"""
        self.status = ExperimentStatus.ABORTED
        self.end_date = datetime.now()
        self.updated_at = datetime.now()
        self.metadata["abort_reason"] = reason

    def is_ready_for_analysis(self) -> tuple:
        """
        检查是否满足分析条件

        Returns:
            (is_ready, reason)
        """
        # 检查最小运行时间
        if self.start_date:
            runtime_days = (datetime.now() - self.start_date).days
            if runtime_days < self.min_runtime_days:
                return False, f"实验运行时间不足 {self.min_runtime_days} 天"

        # 检查最小样本量（需要从结果中获取）
        for variant in self.variants:
            if variant.id in self.results:
                if self.results[variant.id].sample_size < self.min_sample_per_variant:
                    return False, f"变体 {variant.name} 样本量不足 {self.min_sample_per_variant}"

        return True, "满足分析条件"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "variants": [v.to_dict() for v in self.variants],
            "primary_metrics": [m.to_dict() for m in self.primary_metrics],
            "secondary_metrics": [m.to_dict() for m in self.secondary_metrics],
            "guardrail_metrics": [m.to_dict() for m in self.guardrail_metrics],
            "allocation_strategy": self.allocation_strategy.value,
            "traffic_percentage": self.traffic_percentage,
            "stratification": [s.to_dict() for s in self.stratification],
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "min_runtime_days": self.min_runtime_days,
            "max_runtime_days": self.max_runtime_days,
            "min_sample_per_variant": self.min_sample_per_variant,
            "target_sample_per_variant": self.target_sample_per_variant,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner": self.owner,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABExperiment":
        """从字典创建"""
        variants = [
            Variant(**v) for v in data.get("variants", [])
        ]
        primary_metrics = [
            MetricDefinition(**m) for m in data.get("primary_metrics", [])
        ]
        secondary_metrics = [
            MetricDefinition(**m) for m in data.get("secondary_metrics", [])
        ]
        guardrail_metrics = [
            MetricDefinition(**m) for m in data.get("guardrail_metrics", [])
        ]
        stratification = [
            StratificationRule(**s) for s in data.get("stratification", [])
        ]

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            hypothesis=data.get("hypothesis", ""),
            variants=variants,
            primary_metrics=primary_metrics,
            secondary_metrics=secondary_metrics,
            guardrail_metrics=guardrail_metrics,
            allocation_strategy=AllocationStrategy(data.get("allocation_strategy", "user_hash")),
            traffic_percentage=data.get("traffic_percentage", 1.0),
            stratification=stratification,
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            min_runtime_days=data.get("min_runtime_days", 7),
            max_runtime_days=data.get("max_runtime_days", 30),
            min_sample_per_variant=data.get("min_sample_per_variant", 100),
            target_sample_per_variant=data.get("target_sample_per_variant", 1000),
            status=ExperimentStatus(data.get("status", "draft")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            owner=data.get("owner", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

    def save(self, path: str):
        """保存实验配置"""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if filepath.suffix == ".yaml":
                yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ABExperiment":
        """加载实验配置"""
        filepath = Path(path)

        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls.from_dict(data)


class ExperimentRegistry:
    """
    实验注册表

    管理所有实验的生命周期
    """

    def __init__(self, storage_path: str = "evals/ab_testing/experiments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, ABExperiment] = {}
        self._load_experiments()

    def _load_experiments(self):
        """加载所有实验"""
        for filepath in self.storage_path.glob("*.yaml"):
            try:
                exp = ABExperiment.load(str(filepath))
                self._experiments[exp.id] = exp
            except Exception as e:
                print(f"Failed to load experiment {filepath}: {e}")

    def register(self, experiment: ABExperiment) -> str:
        """注册实验"""
        self._experiments[experiment.id] = experiment
        experiment.save(str(self.storage_path / f"{experiment.id}.yaml"))
        return experiment.id

    def get(self, experiment_id: str) -> Optional[ABExperiment]:
        """获取实验"""
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ABExperiment]:
        """列出实验"""
        results = list(self._experiments.values())

        if status:
            results = [e for e in results if e.status == status]

        if tags:
            results = [e for e in results if any(t in e.tags for t in tags)]

        return sorted(results, key=lambda e: e.created_at, reverse=True)

    def get_running_experiments(self) -> List[ABExperiment]:
        """获取正在运行的实验"""
        return self.list_experiments(status=ExperimentStatus.RUNNING)

    def update(self, experiment: ABExperiment):
        """更新实验"""
        experiment.updated_at = datetime.now()
        self._experiments[experiment.id] = experiment
        experiment.save(str(self.storage_path / f"{experiment.id}.yaml"))

    def delete(self, experiment_id: str):
        """删除实验"""
        if experiment_id in self._experiments:
            del self._experiments[experiment_id]
            filepath = self.storage_path / f"{experiment_id}.yaml"
            if filepath.exists():
                filepath.unlink()


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("A/B 测试实验框架测试")
    print("=" * 60)

    # 创建实验
    experiment = ABExperiment.create(
        name="意图分类模型对比",
        control_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "method": "zero_shot"
        },
        treatment_config={
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "method": "few_shot"
        },
        primary_metric="intent_accuracy",
        description="对比 GPT-3.5 和 GPT-4o-mini 在意图分类上的表现",
        hypothesis="GPT-4o-mini 的意图分类准确率将比 GPT-3.5 提高 5% 以上"
    )

    # 添加更多指标
    experiment.secondary_metrics = [
        MetricDefinition(name="slot_f1", description="槽位提取 F1"),
        MetricDefinition(name="response_time", description="响应时间", higher_is_better=False)
    ]

    experiment.guardrail_metrics = [
        MetricDefinition(
            name="error_rate",
            description="错误率",
            higher_is_better=False,
            is_guardrail=True
        )
    ]

    # 添加分层
    experiment.stratification = [
        StratificationRule(
            field="intent_type",
            values=["ORDER_NEW", "ORDER_MODIFY", "RECOMMEND"],
            description="按意图类型分层"
        )
    ]

    print(f"\n实验创建成功!")
    print(f"  ID: {experiment.id}")
    print(f"  名称: {experiment.name}")
    print(f"  假设: {experiment.hypothesis}")
    print(f"  状态: {experiment.status.value}")

    # 启动实验
    experiment.start()
    print(f"\n实验已启动: {experiment.status.value}")

    # 模拟分配
    print(f"\n模拟用户分配:")
    allocation_counts = {"control": 0, "treatment": 0}
    for i in range(100):
        variant = experiment.allocate(user_id=f"user_{i}")
        allocation_counts[variant.id] += 1

    print(f"  对照组: {allocation_counts['control']} 人")
    print(f"  实验组: {allocation_counts['treatment']} 人")

    # 验证哈希一致性
    print(f"\n验证分配一致性:")
    user_id = "test_user_123"
    allocations = [experiment.allocate(user_id=user_id).id for _ in range(10)]
    is_consistent = len(set(allocations)) == 1
    print(f"  同一用户多次分配: {'一致' if is_consistent else '不一致'}")

    # 保存实验
    experiment.save("/tmp/test_experiment.yaml")
    print(f"\n实验已保存")

    # 重新加载
    loaded = ABExperiment.load("/tmp/test_experiment.yaml")
    print(f"实验已加载: {loaded.name}")

    print("\n✅ A/B 测试实验框架测试完成!")
