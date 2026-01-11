"""
性能评分器
Performance Graders

包含延迟评分器和基准对比评分器
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


@dataclass
class LatencyThresholds:
    """延迟阈值配置"""
    p50_target: float = 500
    p95_target: float = 1000
    p99_target: float = 2000
    critical: float = 3000


class LatencyGrader(BaseGrader):
    """
    延迟性能评分器

    评估系统响应延迟是否满足 SLA 要求

    配置示例:
    ```yaml
    graders:
      - type: latency
        p50_target: 500
        p95_target: 1000
        p99_target: 2000
        critical: 3000
    ```
    """

    grader_type = GraderType.LATENCY

    def __init__(self, config: Optional[GraderConfig] = None):
        super().__init__(config)
        self.thresholds = LatencyThresholds(
            p50_target=config.p50_target if config and config.p50_target else 500,
            p95_target=config.p95_target if config and config.p95_target else 1000,
            p99_target=config.p99_target if config and config.p99_target else 2000,
            critical=config.critical if config and config.critical else 3000
        )
        self.component_filter = config.component if config else None

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估延迟性能

        Args:
            predictions: 模型预测结果（应包含 latency_ms 字段）
            test_cases: 测试用例
            transcript: 对话记录

        Returns:
            GraderResult
        """
        # 收集延迟测量
        latency_measurements = []
        for pred in predictions:
            latency = pred.get("latency_ms") or pred.get("duration_ms", 0)
            if latency > 0:
                latency_measurements.append(latency)

        if not latency_measurements:
            return GraderResult(
                grader_type=self.grader_type,
                passed=True,
                score=1.0,
                details={"note": "No latency data available"},
                failures=[]
            )

        # 计算百分位数
        sorted_latencies = sorted(latency_measurements)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = (n - 1) * p / 100
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            weight = idx - lower
            return sorted_latencies[lower] * (1 - weight) + sorted_latencies[upper] * weight

        p50 = percentile(50)
        p90 = percentile(90)
        p95 = percentile(95)
        p99 = percentile(99)

        # 检查违规
        violations = []
        failures = []
        score = 100.0

        if p50 > self.thresholds.p50_target:
            ratio = p50 / self.thresholds.p50_target
            deduction = min(30, (ratio - 1) * 30)
            score -= deduction
            violations.append({
                "type": "p50_violation",
                "actual": round(p50, 2),
                "target": self.thresholds.p50_target,
                "deduction": round(deduction, 1)
            })
            failures.append({
                "type": "p50_violation",
                "message": f"P50 延迟 {p50:.1f}ms 超过目标 {self.thresholds.p50_target}ms"
            })

        if p95 > self.thresholds.p95_target:
            ratio = p95 / self.thresholds.p95_target
            deduction = min(40, (ratio - 1) * 40)
            score -= deduction
            violations.append({
                "type": "p95_violation",
                "actual": round(p95, 2),
                "target": self.thresholds.p95_target,
                "deduction": round(deduction, 1)
            })
            failures.append({
                "type": "p95_violation",
                "message": f"P95 延迟 {p95:.1f}ms 超过目标 {self.thresholds.p95_target}ms"
            })

        if p99 > self.thresholds.p99_target:
            ratio = p99 / self.thresholds.p99_target
            deduction = min(30, (ratio - 1) * 30)
            score -= deduction
            violations.append({
                "type": "p99_violation",
                "actual": round(p99, 2),
                "target": self.thresholds.p99_target,
                "deduction": round(deduction, 1)
            })
            failures.append({
                "type": "p99_violation",
                "message": f"P99 延迟 {p99:.1f}ms 超过目标 {self.thresholds.p99_target}ms"
            })

        # 检查严重超时
        max_latency = max(sorted_latencies)
        if max_latency > self.thresholds.critical:
            failures.append({
                "type": "critical_breach",
                "message": f"最大延迟 {max_latency:.1f}ms 超过严重阈值 {self.thresholds.critical}ms"
            })

        score = max(0, score) / 100

        # 确定等级
        if score >= 0.95:
            grade = "A"
        elif score >= 0.90:
            grade = "B"
        elif score >= 0.85:
            grade = "C"
        elif score >= 0.80:
            grade = "D"
        else:
            grade = "F"

        passed = len(violations) == 0 or score >= 0.80

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=score,
            details={
                "grade": grade,
                "statistics": {
                    "count": n,
                    "p50": round(p50, 2),
                    "p90": round(p90, 2),
                    "p95": round(p95, 2),
                    "p99": round(p99, 2),
                    "min": round(min(sorted_latencies), 2),
                    "max": round(max(sorted_latencies), 2),
                    "mean": round(sum(sorted_latencies) / n, 2)
                },
                "thresholds": {
                    "p50_target": self.thresholds.p50_target,
                    "p95_target": self.thresholds.p95_target,
                    "p99_target": self.thresholds.p99_target,
                    "critical": self.thresholds.critical
                },
                "violations": violations
            },
            failures=failures
        )


class BenchmarkGrader(BaseGrader):
    """
    行业基准对比评分器

    将评估结果与行业基准进行对比

    配置示例:
    ```yaml
    graders:
      - type: benchmark
        metric: "intent_accuracy"
        benchmark_source: "industry"
    ```
    """

    grader_type = GraderType.BENCHMARK

    def __init__(self, config: Optional[GraderConfig] = None):
        super().__init__(config)
        self.metric = config.metric if config and config.metric else "intent_accuracy"
        self.benchmark_source = config.benchmark_source if config and config.benchmark_source else "industry"
        self.benchmarks = self._load_benchmarks()

    def _load_benchmarks(self) -> Dict[str, Any]:
        """加载基准配置"""
        path = Path(__file__).parent.parent / "metrics" / "industry_benchmarks.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _get_benchmark_value(self, metric: str) -> Optional[Dict[str, float]]:
        """获取指定指标的基准值"""
        if metric in ["intent_accuracy", "intent_recognition"]:
            ir = self.benchmarks.get("intent_recognition", {})
            overall = ir.get("overall_accuracy", {})
            return {
                "excellent": overall.get("excellent", 0.98),
                "good": overall.get("good", 0.95),
                "acceptable": overall.get("acceptable", 0.90),
                "industry_average": overall.get("industry_average", 0.92)
            }

        if metric in ["slot_f1", "slot_extraction"]:
            se = self.benchmarks.get("slot_extraction", {})
            overall = se.get("overall_f1", {})
            return {
                "excellent": overall.get("excellent", 0.95),
                "good": overall.get("good", 0.90),
                "acceptable": overall.get("acceptable", 0.85),
                "industry_average": overall.get("industry_average", 0.88)
            }

        if metric in ["task_completion", "task_completion_rate"]:
            dm = self.benchmarks.get("dialogue_management", {})
            tcr = dm.get("task_completion_rate", {})
            return {
                "excellent": tcr.get("excellent", 0.92),
                "good": tcr.get("good", 0.85),
                "acceptable": tcr.get("acceptable", 0.80),
                "industry_average": tcr.get("industry_average", 0.83)
            }

        if metric in ["fuzzy_accuracy", "fuzzy_understanding"]:
            fu = self.benchmarks.get("fuzzy_understanding", {})
            overall = fu.get("overall_accuracy", {})
            return {
                "excellent": overall.get("excellent", 0.90),
                "good": overall.get("good", 0.85),
                "acceptable": overall.get("acceptable", 0.80),
                "industry_average": overall.get("industry_average", 0.82)
            }

        return None

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估并与基准对比

        Args:
            predictions: 模型预测结果
            test_cases: 测试用例

        Returns:
            GraderResult
        """
        # 根据指标计算实际值
        actual_value = self._calculate_metric(predictions, test_cases)

        benchmark = self._get_benchmark_value(self.metric)

        if not benchmark:
            return GraderResult(
                grader_type=self.grader_type,
                passed=True,
                score=actual_value,
                details={"note": f"No benchmark available for {self.metric}"},
                failures=[]
            )

        # 确定等级
        if actual_value >= benchmark["excellent"]:
            grade = "A"
            grade_label = "优秀"
            emoji = "🟢"
        elif actual_value >= benchmark["good"]:
            grade = "B"
            grade_label = "良好"
            emoji = "🟡"
        elif actual_value >= benchmark["acceptable"]:
            grade = "C"
            grade_label = "合格"
            emoji = "🟠"
        else:
            grade = "D"
            grade_label = "待改进"
            emoji = "🔴"

        # 与行业平均对比
        vs_industry = actual_value - benchmark["industry_average"]
        vs_industry_pct = (vs_industry / benchmark["industry_average"]) * 100 if benchmark["industry_average"] > 0 else 0

        passed = actual_value >= benchmark["acceptable"]
        failures = []
        if not passed:
            gap = benchmark["acceptable"] - actual_value
            failures.append({
                "type": "below_benchmark",
                "message": f"{self.metric} ({actual_value:.2%}) 低于行业可接受标准 ({benchmark['acceptable']:.2%})，差距 {gap:.2%}"
            })

        interpretation = self._generate_interpretation(actual_value, benchmark, grade)

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=actual_value,
            details={
                "metric": self.metric,
                "grade": grade,
                "grade_label": f"{emoji} {grade_label}",
                "benchmark_comparison": {
                    "actual": round(actual_value, 4),
                    "industry_average": benchmark["industry_average"],
                    "vs_industry": f"{vs_industry_pct:+.1f}%",
                    "excellent_threshold": benchmark["excellent"],
                    "good_threshold": benchmark["good"],
                    "acceptable_threshold": benchmark["acceptable"]
                },
                "interpretation": interpretation
            },
            failures=failures
        )

    def _calculate_metric(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase]
    ) -> float:
        """计算指定指标的值"""
        if self.metric in ["intent_accuracy", "intent_recognition"]:
            # 计算意图准确率
            if not predictions or not test_cases:
                return 0.0
            correct = 0
            total = 0
            for pred, tc in zip(predictions, test_cases):
                if tc.expected_intent:
                    total += 1
                    if pred.get("intent") == tc.expected_intent:
                        correct += 1
            return correct / total if total > 0 else 0.0

        if self.metric in ["slot_f1", "slot_extraction"]:
            # 简化的 F1 计算
            if not predictions or not test_cases:
                return 0.0
            tp = fp = fn = 0
            for pred, tc in zip(predictions, test_cases):
                pred_slots = pred.get("slots", {})
                expected_slots = tc.expected_slots or {}
                for key in set(pred_slots.keys()) | set(expected_slots.keys()):
                    if key in pred_slots and key in expected_slots:
                        if pred_slots[key] == expected_slots[key]:
                            tp += 1
                        else:
                            fp += 1
                            fn += 1
                    elif key in pred_slots:
                        fp += 1
                    else:
                        fn += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return 0.0

    def _generate_interpretation(
        self,
        actual: float,
        benchmark: Dict[str, float],
        grade: str
    ) -> str:
        """生成解读说明"""
        metric_names = {
            "intent_accuracy": "意图识别准确率",
            "slot_f1": "槽位提取 F1",
            "task_completion": "任务完成率",
            "fuzzy_accuracy": "模糊表达理解准确率"
        }
        metric_name = metric_names.get(self.metric, self.metric)

        if grade == "A":
            return f"{metric_name}达到优秀水平，超越行业领先标准"
        elif grade == "B":
            return f"{metric_name}达到良好水平，高于行业平均"
        elif grade == "C":
            return f"{metric_name}达到合格水平，接近行业平均"
        else:
            gap = benchmark["acceptable"] - actual
            return f"{metric_name}低于行业标准，需提升 {gap:.1%} 达到合格线"


class PerformanceProfileGrader(BaseGrader):
    """
    性能画像评分器

    综合评估延迟和准确率，生成性能画像

    配置示例:
    ```yaml
    graders:
      - type: performance_profile
        latency_weight: 0.3
        accuracy_weight: 0.7
    ```
    """

    grader_type = GraderType.PERFORMANCE_PROFILE

    def __init__(self, config: Optional[GraderConfig] = None):
        super().__init__(config)
        self.latency_weight = config.latency_weight if config and config.latency_weight else 0.3
        self.accuracy_weight = config.accuracy_weight if config and config.accuracy_weight else 0.7

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        生成性能画像

        Args:
            predictions: 模型预测结果
            test_cases: 测试用例

        Returns:
            GraderResult
        """
        # 收集延迟数据
        latencies = []
        for pred in predictions:
            latency = pred.get("latency_ms") or pred.get("duration_ms", 0)
            if latency > 0:
                latencies.append(latency)

        # 计算准确率
        correct = 0
        total = 0
        for pred, tc in zip(predictions, test_cases):
            if tc.expected_intent:
                total += 1
                if pred.get("intent") == tc.expected_intent:
                    correct += 1

        accuracy_score = correct / total if total > 0 else 0.0

        # 计算延迟评分 (越低越好)
        latency_score = 1.0
        latency_stats = {}
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[n // 2]
            p95_idx = min(int(n * 0.95), n - 1)
            p95 = sorted_lat[p95_idx]
            # 假设 1000ms 为基准
            latency_score = max(0, 1 - (p95 - 500) / 1500)
            latency_stats = {
                "count": n,
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "min": round(min(sorted_lat), 2),
                "max": round(max(sorted_lat), 2),
                "score": round(latency_score, 3)
            }

        accuracy_stats = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy_score, 4),
            "score": round(accuracy_score, 3)
        }

        # 综合评分
        overall_score = (
            self.latency_weight * latency_score +
            self.accuracy_weight * accuracy_score
        )

        # 确定性能画像类型
        profile = self._determine_profile(latency_score, accuracy_score)

        passed = overall_score >= 0.80
        failures = []
        if not passed:
            if latency_score < 0.7:
                failures.append({
                    "type": "high_latency",
                    "message": f"延迟评分 {latency_score:.2f} 过低"
                })
            if accuracy_score < 0.85:
                failures.append({
                    "type": "low_accuracy",
                    "message": f"准确率 {accuracy_score:.2%} 需提升"
                })

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=overall_score,
            details={
                "profile": profile,
                "latency": latency_stats,
                "accuracy": accuracy_stats,
                "weights": {
                    "latency": self.latency_weight,
                    "accuracy": self.accuracy_weight
                },
                "overall_score": round(overall_score, 3)
            },
            failures=failures
        )

    def _determine_profile(self, latency_score: float, accuracy_score: float) -> Dict[str, str]:
        """确定性能画像类型"""
        if latency_score >= 0.9 and accuracy_score >= 0.95:
            return {
                "type": "高性能",
                "emoji": "🚀",
                "description": "延迟低、准确率高，表现优异"
            }
        elif latency_score >= 0.8 and accuracy_score >= 0.90:
            return {
                "type": "均衡型",
                "emoji": "⚖️",
                "description": "延迟和准确率平衡良好"
            }
        elif latency_score < 0.7 and accuracy_score >= 0.90:
            return {
                "type": "准确优先",
                "emoji": "🎯",
                "description": "准确率高但延迟较大，适合对准确性要求高的场景"
            }
        elif latency_score >= 0.8 and accuracy_score < 0.85:
            return {
                "type": "速度优先",
                "emoji": "⚡",
                "description": "响应快但准确率需提升"
            }
        else:
            return {
                "type": "待优化",
                "emoji": "🔧",
                "description": "延迟和准确率都需要改进"
            }
