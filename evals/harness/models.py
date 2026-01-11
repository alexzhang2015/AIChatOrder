"""
评估框架核心数据模型

定义评估任务、测试用例、评分配置等数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum


class TaskCategory(str, Enum):
    """任务类别"""
    REGRESSION = "regression"      # 回归测试，要求高通过率
    CAPABILITY = "capability"      # 能力测试，允许低通过率
    EDGE_CASE = "edge_case"        # 边界场景


class GraderType(str, Enum):
    """评分器类型"""
    INTENT_ACCURACY = "intent_accuracy"
    SLOT_F1 = "slot_f1"
    SLOT_NORMALIZATION = "slot_normalization"
    FUZZY_MATCH = "fuzzy_match"
    CONSTRAINT_VALIDATION = "constraint_validation"
    STATE_CHECK = "state_check"
    LLM_RUBRIC = "llm_rubric"
    TRANSCRIPT = "transcript"
    BEHAVIOR_CHECK = "behavior_check"
    CONFUSION_MATRIX = "confusion_matrix"
    SAFETY_CHECK = "safety_check"
    # Phase 5: 性能评分器
    LATENCY = "latency"
    BENCHMARK = "benchmark"
    PERFORMANCE_PROFILE = "performance_profile"


@dataclass
class TestCase:
    """单个测试用例"""
    input: Union[str, Dict[str, Any]]
    expected_intent: Optional[str] = None
    expected_slots: Optional[Dict[str, Any]] = None
    expected_confidence: Optional[float] = None
    expected_behavior: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        return cls(
            input=data.get("input", ""),
            expected_intent=data.get("expected_intent"),
            expected_slots=data.get("expected_slots"),
            expected_confidence=data.get("expected_confidence"),
            expected_behavior=data.get("expected_behavior"),
            metadata=data.get("metadata", {})
        )


@dataclass
class GraderConfig:
    """评分器配置"""
    type: GraderType
    # 通用配置
    weight: float = 1.0
    required: bool = True

    # Intent Grader 配置
    min_accuracy: Optional[float] = None
    per_intent_min: Optional[float] = None

    # Slot Grader 配置
    min_precision: Optional[float] = None
    min_recall: Optional[float] = None
    check_fields: Optional[List[str]] = None

    # LLM Rubric 配置
    model: Optional[str] = None
    rubric: Optional[str] = None
    min_scores: Optional[Dict[str, float]] = None

    # Transcript Grader 配置
    max_turns: Optional[int] = None

    # State Check 配置
    expect: Optional[Dict[str, Any]] = None

    # Fuzzy Match 配置
    min_match_accuracy: Optional[float] = None

    # Confusion Matrix 配置
    alert_pairs: Optional[List[List[str]]] = None
    alert_threshold: Optional[float] = None

    # Safety Check 配置
    must_recommend: Optional[List[str]] = None
    must_not_recommend: Optional[List[str]] = None
    expected_behavior: Optional[str] = None

    # Latency Grader 配置
    p50_target: Optional[float] = None
    p95_target: Optional[float] = None
    p99_target: Optional[float] = None
    critical: Optional[float] = None
    component: Optional[str] = None

    # Benchmark Grader 配置
    metric: Optional[str] = None
    benchmark_source: Optional[str] = None

    # Performance Profile Grader 配置
    latency_weight: Optional[float] = None
    accuracy_weight: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraderConfig":
        grader_type = data.get("type", "")
        try:
            gtype = GraderType(grader_type)
        except ValueError:
            gtype = GraderType.INTENT_ACCURACY  # 默认

        return cls(
            type=gtype,
            weight=data.get("weight", 1.0),
            required=data.get("required", True),
            min_accuracy=data.get("min_accuracy"),
            per_intent_min=data.get("per_intent_min"),
            min_precision=data.get("min_precision"),
            min_recall=data.get("min_recall"),
            check_fields=data.get("check_fields"),
            model=data.get("model"),
            rubric=data.get("rubric"),
            min_scores=data.get("min_scores"),
            max_turns=data.get("max_turns"),
            expect=data.get("expect"),
            min_match_accuracy=data.get("min_match_accuracy"),
            alert_pairs=data.get("alert_pairs"),
            alert_threshold=data.get("alert_threshold"),
            must_recommend=data.get("must_recommend"),
            must_not_recommend=data.get("must_not_recommend"),
            expected_behavior=data.get("expected_behavior"),
            # Performance grader configs
            p50_target=data.get("p50_target"),
            p95_target=data.get("p95_target"),
            p99_target=data.get("p99_target"),
            critical=data.get("critical"),
            component=data.get("component"),
            metric=data.get("metric"),
            benchmark_source=data.get("benchmark_source"),
            latency_weight=data.get("latency_weight"),
            accuracy_weight=data.get("accuracy_weight")
        )


@dataclass
class SimulationConfig:
    """对话模拟配置"""
    user_persona: str
    user_goal: str
    user_simulator: str = "claude-3-5-sonnet"
    max_turns: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        return cls(
            user_persona=data.get("user_persona", ""),
            user_goal=data.get("user_goal", ""),
            user_simulator=data.get("user_simulator", "claude-3-5-sonnet"),
            max_turns=data.get("max_turns", 10)
        )


@dataclass
class TaskConfig:
    """评估任务配置"""
    id: str
    name: str
    description: str = ""
    category: TaskCategory = TaskCategory.REGRESSION

    # 测试用例（批量测试）
    test_cases: List[TestCase] = field(default_factory=list)

    # 对话模拟配置（端到端测试）
    simulation: Optional[SimulationConfig] = None

    # 评分器列表
    graders: List[GraderConfig] = field(default_factory=list)

    # 追踪指标
    tracked_metrics: List[str] = field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        task_data = data.get("task", data)

        # 解析类别
        category_str = task_data.get("metadata", {}).get("category", "regression")
        try:
            category = TaskCategory(category_str)
        except ValueError:
            category = TaskCategory.REGRESSION

        # 解析测试用例
        test_cases = [
            TestCase.from_dict(tc)
            for tc in task_data.get("test_cases", [])
        ]

        # 解析模拟配置
        simulation = None
        if "simulation" in task_data:
            simulation = SimulationConfig.from_dict(task_data["simulation"])

        # 解析评分器
        graders = [
            GraderConfig.from_dict(g)
            for g in task_data.get("graders", [])
        ]

        return cls(
            id=task_data.get("id", "unknown"),
            name=task_data.get("name", "Unnamed Task"),
            description=task_data.get("description", ""),
            category=category,
            test_cases=test_cases,
            simulation=simulation,
            graders=graders,
            tracked_metrics=task_data.get("tracked_metrics", []),
            metadata=task_data.get("metadata", {})
        )


@dataclass
class GraderResult:
    """评分器结果"""
    grader_type: GraderType
    passed: bool
    score: float  # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grader_type": self.grader_type.value,
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            "failures": self.failures
        }


@dataclass
class Trial:
    """单次试验结果"""
    task_id: str
    trial_number: int
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)
    grader_results: Dict[str, GraderResult] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    passed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trial_number": self.trial_number,
            "transcript": self.transcript,
            "outcome": self.outcome,
            "grader_results": {
                k: v.to_dict() for k, v in self.grader_results.items()
            },
            "metrics": self.metrics,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms
        }


@dataclass
class EvalResult:
    """评估任务结果"""
    task_id: str
    task_name: str
    category: TaskCategory
    trials: List[Trial] = field(default_factory=list)

    # 统计指标
    pass_at_1: float = 0.0       # 首次成功率
    pass_at_k: float = 0.0       # k次中至少成功一次
    pass_all: float = 0.0        # 全部成功率

    # 聚合指标
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)

    # 元数据
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "category": self.category.value,
            "trials": [t.to_dict() for t in self.trials],
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "pass_all": self.pass_all,
            "aggregate_metrics": self.aggregate_metrics,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


@dataclass
class EvalSuiteResult:
    """评估套件结果"""
    suite_name: str
    tasks_total: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0
    overall_pass_rate: float = 0.0
    results: List[EvalResult] = field(default_factory=list)

    # 按类别统计
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 执行信息
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    total_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "tasks_total": self.tasks_total,
            "tasks_passed": self.tasks_passed,
            "tasks_failed": self.tasks_failed,
            "overall_pass_rate": self.overall_pass_rate,
            "results": [r.to_dict() for r in self.results],
            "by_category": self.by_category,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms
        }
