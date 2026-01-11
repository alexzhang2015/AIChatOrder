"""
A/B 测试运行器
A/B Test Runner

协调实验执行、数据收集和分析
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

from evals.ab_testing.experiment import (
    ABExperiment, ExperimentStatus, Variant, ExperimentResult,
    ExperimentRegistry, MetricDefinition
)
from evals.ab_testing.analyzer import (
    ABTestAnalyzer, StatisticalResult, ExperimentAnalysis, TestResult
)


@dataclass
class TrialRecord:
    """单次试验记录"""
    experiment_id: str
    variant_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    metrics: Dict[str, float]
    latency_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "metadata": self.metadata
        }


class ABTestRunner:
    """
    A/B 测试运行器

    功能：
    - 运行实验
    - 收集数据
    - 执行分析
    - 生成报告
    """

    def __init__(
        self,
        storage_path: str = "evals/ab_testing/data",
        registry: Optional[ExperimentRegistry] = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.registry = registry or ExperimentRegistry()
        self.analyzer = ABTestAnalyzer()
        self._trial_records: Dict[str, List[TrialRecord]] = {}

    def run_trial(
        self,
        experiment: ABExperiment,
        agent_func: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        input_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> TrialRecord:
        """
        运行单次试验

        Args:
            experiment: 实验配置
            agent_func: Agent 函数 (input, variant_config) -> output
            input_data: 输入数据
            user_id: 用户ID
            session_id: 会话ID

        Returns:
            TrialRecord
        """
        # 分配变体
        variant = experiment.allocate(user_id=user_id, session_id=session_id)

        # 执行试验
        start_time = time.perf_counter()
        success = True
        output_data = {}

        try:
            output_data = agent_func(input_data, variant.config)
        except Exception as e:
            success = False
            output_data = {"error": str(e)}

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 计算指标
        metrics = self._calculate_metrics(input_data, output_data, experiment)

        # 创建记录
        record = TrialRecord(
            experiment_id=experiment.id,
            variant_id=variant.id,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            input_data=input_data,
            output_data=output_data,
            metrics=metrics,
            latency_ms=latency_ms,
            success=success
        )

        # 存储记录
        if experiment.id not in self._trial_records:
            self._trial_records[experiment.id] = []
        self._trial_records[experiment.id].append(record)

        return record

    def run_batch(
        self,
        experiment: ABExperiment,
        agent_func: Callable,
        test_cases: List[Dict[str, Any]],
        user_id_field: str = "user_id"
    ) -> List[TrialRecord]:
        """
        批量运行试验

        Args:
            experiment: 实验配置
            agent_func: Agent 函数
            test_cases: 测试用例列表
            user_id_field: 用户ID字段名

        Returns:
            TrialRecord 列表
        """
        records = []

        for i, test_case in enumerate(test_cases):
            user_id = test_case.get(user_id_field, f"user_{i}")
            record = self.run_trial(
                experiment=experiment,
                agent_func=agent_func,
                input_data=test_case,
                user_id=user_id
            )
            records.append(record)

        return records

    def _calculate_metrics(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        experiment: ABExperiment
    ) -> Dict[str, float]:
        """计算指标"""
        metrics = {}

        # 从输出数据中提取指标
        if "intent" in output_data and "expected_intent" in input_data:
            metrics["intent_correct"] = float(
                output_data.get("intent") == input_data.get("expected_intent")
            )

        if "confidence" in output_data:
            metrics["confidence"] = output_data.get("confidence", 0)

        if "slots" in output_data and "expected_slots" in input_data:
            pred_slots = output_data.get("slots", {})
            expected_slots = input_data.get("expected_slots", {})

            # 简化的槽位准确率
            if expected_slots:
                correct = sum(
                    1 for k, v in expected_slots.items()
                    if pred_slots.get(k) == v
                )
                metrics["slot_accuracy"] = correct / len(expected_slots)

        # 错误率
        metrics["error"] = float(not output_data or "error" in output_data)

        return metrics

    def get_trial_data(
        self,
        experiment_id: str,
        variant_id: Optional[str] = None
    ) -> List[TrialRecord]:
        """获取试验数据"""
        records = self._trial_records.get(experiment_id, [])

        if variant_id:
            records = [r for r in records if r.variant_id == variant_id]

        return records

    def analyze_experiment(
        self,
        experiment: ABExperiment
    ) -> ExperimentAnalysis:
        """
        分析实验结果

        Args:
            experiment: 实验配置

        Returns:
            ExperimentAnalysis
        """
        records = self.get_trial_data(experiment.id)

        # 按变体分组
        variant_data: Dict[str, List[TrialRecord]] = {}
        for record in records:
            if record.variant_id not in variant_data:
                variant_data[record.variant_id] = []
            variant_data[record.variant_id].append(record)

        # 获取对照组和实验组
        control = experiment.get_control()
        treatment = experiment.get_treatment()

        if not control or not treatment:
            raise ValueError("实验缺少对照组或实验组")

        control_records = variant_data.get(control.id, [])
        treatment_records = variant_data.get(treatment.id, [])

        # 分析主要指标
        primary_results = []
        for metric_def in experiment.primary_metrics:
            result = self._analyze_metric(
                metric_def,
                control_records,
                treatment_records
            )
            primary_results.append(result)

        # 分析次要指标
        secondary_results = []
        for metric_def in experiment.secondary_metrics:
            result = self._analyze_metric(
                metric_def,
                control_records,
                treatment_records
            )
            secondary_results.append(result)

        # 分析护栏指标
        guardrail_results = []
        for metric_def in experiment.guardrail_metrics:
            result = self._analyze_metric(
                metric_def,
                control_records,
                treatment_records,
                is_guardrail=True
            )
            guardrail_results.append(result)

        # 生成建议
        recommendation, confidence_summary, is_conclusive, winner = \
            self.analyzer.generate_recommendation(primary_results, guardrail_results)

        return ExperimentAnalysis(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            analysis_date=datetime.now().isoformat(),
            sample_sizes={
                control.id: len(control_records),
                treatment.id: len(treatment_records)
            },
            primary_results=primary_results,
            secondary_results=secondary_results,
            guardrail_results=guardrail_results,
            recommendation=recommendation,
            confidence_summary=confidence_summary,
            is_conclusive=is_conclusive,
            winner=winner
        )

    def _analyze_metric(
        self,
        metric_def: MetricDefinition,
        control_records: List[TrialRecord],
        treatment_records: List[TrialRecord],
        is_guardrail: bool = False
    ) -> StatisticalResult:
        """分析单个指标"""
        metric_name = metric_def.name

        # 提取指标值
        control_values = [
            r.metrics.get(metric_name, 0) for r in control_records
            if metric_name in r.metrics
        ]
        treatment_values = [
            r.metrics.get(metric_name, 0) for r in treatment_records
            if metric_name in r.metrics
        ]

        # 如果是比例类型，使用比例检验
        if metric_def.metric_type == "ratio":
            # 假设值为 0 或 1
            control_successes = sum(1 for v in control_values if v > 0.5)
            treatment_successes = sum(1 for v in treatment_values if v > 0.5)

            return self.analyzer.proportion_test(
                control_successes=control_successes,
                control_total=len(control_values),
                treatment_successes=treatment_successes,
                treatment_total=len(treatment_values),
                metric_name=metric_name,
                higher_is_better=metric_def.higher_is_better,
                is_guardrail=is_guardrail
            )
        else:
            # 使用 t 检验
            return self.analyzer.two_sample_ttest(
                control_data=control_values,
                treatment_data=treatment_values,
                metric_name=metric_name,
                higher_is_better=metric_def.higher_is_better,
                is_guardrail=is_guardrail
            )

    def generate_report(
        self,
        analysis: ExperimentAnalysis,
        format: str = "markdown"
    ) -> str:
        """
        生成实验报告

        Args:
            analysis: 分析结果
            format: 输出格式 ("markdown", "html")

        Returns:
            报告内容
        """
        if format == "markdown":
            return self._generate_markdown_report(analysis)
        else:
            return self._generate_markdown_report(analysis)

    def _generate_markdown_report(self, analysis: ExperimentAnalysis) -> str:
        """生成 Markdown 报告"""
        lines = [
            f"# A/B 测试报告: {analysis.experiment_name}",
            "",
            f"**实验 ID:** {analysis.experiment_id}",
            f"**分析日期:** {analysis.analysis_date}",
            "",
            "## 样本量",
            "",
            "| 变体 | 样本数 |",
            "|------|--------|",
        ]

        for variant_id, count in analysis.sample_sizes.items():
            lines.append(f"| {variant_id} | {count} |")

        lines.extend([
            "",
            "## 主要指标",
            "",
        ])

        for result in analysis.primary_results:
            emoji = "✅" if result.test_result == TestResult.SIGNIFICANT_POSITIVE else \
                    "❌" if result.test_result == TestResult.SIGNIFICANT_NEGATIVE else "➖"
            lines.extend([
                f"### {emoji} {result.metric_name}",
                "",
                f"- 对照组: {result.control_mean:.4f}",
                f"- 实验组: {result.treatment_mean:.4f}",
                f"- 变化: {result.relative_difference:+.2%}",
                f"- p值: {result.p_value:.4f}",
                f"- 置信区间: [{result.confidence_interval[0]:.2%}, {result.confidence_interval[1]:.2%}]",
                f"- 结论: {result.interpret()}",
                "",
            ])

        if analysis.guardrail_results:
            lines.extend([
                "## 护栏指标",
                "",
            ])
            for result in analysis.guardrail_results:
                status = "⚠️ 违规" if result.is_guardrail_violated else "✅ 正常"
                lines.extend([
                    f"### {status} {result.metric_name}",
                    "",
                    f"- 变化: {result.relative_difference:+.2%}",
                    f"- p值: {result.p_value:.4f}",
                    "",
                ])

        lines.extend([
            "## 结论与建议",
            "",
            f"**建议:** {analysis.recommendation}",
            "",
            f"**置信度:** {analysis.confidence_summary}",
            "",
            f"**是否得出结论:** {'是' if analysis.is_conclusive else '否'}",
            "",
        ])

        if analysis.winner:
            lines.append(f"**推荐采用:** {analysis.winner}")

        return "\n".join(lines)

    def save_results(self, experiment_id: str):
        """保存实验结果"""
        records = self.get_trial_data(experiment_id)

        filepath = self.storage_path / f"{experiment_id}_trials.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in records],
                f,
                ensure_ascii=False,
                indent=2
            )

    def load_results(self, experiment_id: str) -> List[TrialRecord]:
        """加载实验结果"""
        filepath = self.storage_path / f"{experiment_id}_trials.json"

        if not filepath.exists():
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for item in data:
            record = TrialRecord(
                experiment_id=item["experiment_id"],
                variant_id=item["variant_id"],
                user_id=item.get("user_id"),
                session_id=item.get("session_id"),
                timestamp=datetime.fromisoformat(item["timestamp"]),
                input_data=item["input_data"],
                output_data=item["output_data"],
                metrics=item["metrics"],
                latency_ms=item["latency_ms"],
                success=item["success"],
                metadata=item.get("metadata", {})
            )
            records.append(record)

        self._trial_records[experiment_id] = records
        return records


# 测试代码
if __name__ == "__main__":
    import random
    import numpy as np

    print("=" * 60)
    print("A/B 测试运行器测试")
    print("=" * 60)

    # 创建实验
    experiment = ABExperiment.create(
        name="分类方法对比",
        control_config={"method": "zero_shot"},
        treatment_config={"method": "few_shot"},
        primary_metric="intent_correct"
    )

    experiment.secondary_metrics = [
        MetricDefinition(name="confidence", description="置信度")
    ]

    experiment.guardrail_metrics = [
        MetricDefinition(
            name="error",
            description="错误率",
            higher_is_better=False,
            is_guardrail=True
        )
    ]

    experiment.start()

    # 创建运行器
    runner = ABTestRunner()

    # 模拟 Agent 函数
    def mock_agent(input_data: Dict, config: Dict) -> Dict:
        # 模拟不同方法的效果
        if config.get("method") == "few_shot":
            # few_shot 方法更准确
            correct_prob = 0.92
        else:
            correct_prob = 0.85

        is_correct = random.random() < correct_prob

        return {
            "intent": input_data.get("expected_intent") if is_correct else "WRONG",
            "confidence": random.gauss(0.85 if is_correct else 0.6, 0.1)
        }

    # 生成测试用例
    intents = ["ORDER_NEW", "ORDER_MODIFY", "RECOMMEND", "PRODUCT_INFO"]
    test_cases = [
        {
            "input": f"测试输入 {i}",
            "expected_intent": random.choice(intents),
            "user_id": f"user_{i}"
        }
        for i in range(200)
    ]

    # 运行批量测试
    print("\n运行 200 个测试用例...")
    records = runner.run_batch(experiment, mock_agent, test_cases)

    # 统计分配
    control_count = sum(1 for r in records if r.variant_id == "control")
    treatment_count = sum(1 for r in records if r.variant_id == "treatment")
    print(f"  对照组: {control_count} 次")
    print(f"  实验组: {treatment_count} 次")

    # 分析实验
    print("\n分析实验结果...")
    analysis = runner.analyze_experiment(experiment)

    # 打印结果
    print(f"\n主要指标结果:")
    for result in analysis.primary_results:
        print(f"  {result.metric_name}:")
        print(f"    对照组: {result.control_mean:.2%}")
        print(f"    实验组: {result.treatment_mean:.2%}")
        print(f"    变化: {result.relative_difference:+.1%}")
        print(f"    显著性: {result.test_result.value}")

    print(f"\n建议: {analysis.recommendation}")
    print(f"置信度: {analysis.confidence_summary}")

    # 生成报告
    report = runner.generate_report(analysis)
    print("\n" + "=" * 60)
    print("生成的报告 (前 40 行):")
    print("=" * 60)
    for line in report.split("\n")[:40]:
        print(line)

    print("\n✅ A/B 测试运行器测试完成!")
