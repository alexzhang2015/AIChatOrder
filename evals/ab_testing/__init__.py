"""
A/B 测试框架
A/B Testing Framework

提供完整的 A/B 测试功能：
- 实验定义和管理
- 流量分配
- 统计分析
- 报告生成
"""

from evals.ab_testing.experiment import (
    ABExperiment,
    ExperimentStatus,
    AllocationStrategy,
    Variant,
    MetricDefinition,
    StratificationRule,
    ExperimentResult,
    ExperimentRegistry,
)

from evals.ab_testing.analyzer import (
    ABTestAnalyzer,
    SignificanceLevel,
    TestResult,
    StatisticalResult,
    ExperimentAnalysis,
)

from evals.ab_testing.runner import (
    ABTestRunner,
    TrialRecord,
)

__all__ = [
    # Experiment
    "ABExperiment",
    "ExperimentStatus",
    "AllocationStrategy",
    "Variant",
    "MetricDefinition",
    "StratificationRule",
    "ExperimentResult",
    "ExperimentRegistry",
    # Analyzer
    "ABTestAnalyzer",
    "SignificanceLevel",
    "TestResult",
    "StatisticalResult",
    "ExperimentAnalysis",
    # Runner
    "ABTestRunner",
    "TrialRecord",
]
