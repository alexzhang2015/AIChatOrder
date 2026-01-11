"""
A/B 测试统计分析器
A/B Testing Statistical Analyzer

提供统计显著性检验、置信区间计算、效应量分析等功能
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from scipy import stats
import numpy as np


class SignificanceLevel(float, Enum):
    """显著性水平"""
    VERY_HIGH = 0.001   # 99.9%
    HIGH = 0.01         # 99%
    STANDARD = 0.05     # 95%
    LOW = 0.10          # 90%


class TestResult(str, Enum):
    """检验结果"""
    SIGNIFICANT_POSITIVE = "significant_positive"   # 显著提升
    SIGNIFICANT_NEGATIVE = "significant_negative"   # 显著下降
    NOT_SIGNIFICANT = "not_significant"             # 不显著
    INSUFFICIENT_DATA = "insufficient_data"         # 数据不足


@dataclass
class StatisticalResult:
    """统计检验结果"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_difference: float
    relative_difference: float  # 百分比变化
    p_value: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    test_result: TestResult
    effect_size: float  # Cohen's d 或其他效应量
    sample_size_control: int
    sample_size_treatment: int
    power: float  # 统计功效
    is_guardrail_violated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "control_mean": round(self.control_mean, 4),
            "treatment_mean": round(self.treatment_mean, 4),
            "absolute_difference": round(self.absolute_difference, 4),
            "relative_difference": f"{self.relative_difference:+.2%}",
            "p_value": round(self.p_value, 4),
            "confidence_interval": (
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4)
            ),
            "confidence_level": f"{self.confidence_level:.0%}",
            "test_result": self.test_result.value,
            "effect_size": round(self.effect_size, 3),
            "sample_size": {
                "control": self.sample_size_control,
                "treatment": self.sample_size_treatment
            },
            "power": round(self.power, 3),
            "is_guardrail_violated": self.is_guardrail_violated
        }

    def interpret(self) -> str:
        """生成结果解读"""
        if self.test_result == TestResult.INSUFFICIENT_DATA:
            return f"数据不足，无法得出结论（需要更多样本）"

        direction = "提升" if self.relative_difference > 0 else "下降"
        abs_change = abs(self.relative_difference)

        if self.test_result == TestResult.SIGNIFICANT_POSITIVE:
            return (
                f"{self.metric_name} 显著{direction} {abs_change:.1%}，"
                f"95% 置信区间 [{self.confidence_interval[0]:.2%}, {self.confidence_interval[1]:.2%}]，"
                f"p值={self.p_value:.4f}"
            )
        elif self.test_result == TestResult.SIGNIFICANT_NEGATIVE:
            return (
                f"⚠️ {self.metric_name} 显著{direction} {abs_change:.1%}，"
                f"p值={self.p_value:.4f}，需要关注"
            )
        else:
            return (
                f"{self.metric_name} 变化 {self.relative_difference:+.1%} 不显著，"
                f"p值={self.p_value:.4f}"
            )


@dataclass
class ExperimentAnalysis:
    """完整实验分析结果"""
    experiment_id: str
    experiment_name: str
    analysis_date: str
    sample_sizes: Dict[str, int]
    primary_results: List[StatisticalResult]
    secondary_results: List[StatisticalResult]
    guardrail_results: List[StatisticalResult]
    recommendation: str
    confidence_summary: str
    is_conclusive: bool
    winner: Optional[str] = None  # "control", "treatment", or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "analysis_date": self.analysis_date,
            "sample_sizes": self.sample_sizes,
            "primary_results": [r.to_dict() for r in self.primary_results],
            "secondary_results": [r.to_dict() for r in self.secondary_results],
            "guardrail_results": [r.to_dict() for r in self.guardrail_results],
            "recommendation": self.recommendation,
            "confidence_summary": self.confidence_summary,
            "is_conclusive": self.is_conclusive,
            "winner": self.winner
        }


class ABTestAnalyzer:
    """
    A/B 测试分析器

    提供统计分析功能：
    - t 检验 / z 检验
    - 置信区间计算
    - 效应量计算
    - 功效分析
    - 多重比较校正
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_sample_size: int = 30
    ):
        self.significance_level = significance_level
        self.confidence_level = 1 - significance_level
        self.min_sample_size = min_sample_size

    def two_sample_ttest(
        self,
        control_data: List[float],
        treatment_data: List[float],
        metric_name: str = "metric",
        higher_is_better: bool = True,
        is_guardrail: bool = False
    ) -> StatisticalResult:
        """
        双样本 t 检验

        Args:
            control_data: 对照组数据
            treatment_data: 实验组数据
            metric_name: 指标名称
            higher_is_better: 是否越高越好
            is_guardrail: 是否为护栏指标

        Returns:
            StatisticalResult
        """
        n_control = len(control_data)
        n_treatment = len(treatment_data)

        # 检查数据量
        if n_control < self.min_sample_size or n_treatment < self.min_sample_size:
            return StatisticalResult(
                metric_name=metric_name,
                control_mean=np.mean(control_data) if control_data else 0,
                treatment_mean=np.mean(treatment_data) if treatment_data else 0,
                absolute_difference=0,
                relative_difference=0,
                p_value=1.0,
                confidence_interval=(0, 0),
                confidence_level=self.confidence_level,
                test_result=TestResult.INSUFFICIENT_DATA,
                effect_size=0,
                sample_size_control=n_control,
                sample_size_treatment=n_treatment,
                power=0
            )

        # 计算统计量
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        control_std = np.std(control_data, ddof=1)
        treatment_std = np.std(treatment_data, ddof=1)

        absolute_diff = treatment_mean - control_mean
        relative_diff = absolute_diff / control_mean if control_mean != 0 else 0

        # Welch's t-test (不假设方差相等)
        t_stat, p_value = stats.ttest_ind(
            treatment_data, control_data, equal_var=False
        )

        # 计算置信区间
        pooled_se = math.sqrt(
            (control_std ** 2 / n_control) + (treatment_std ** 2 / n_treatment)
        )
        # 使用 Welch-Satterthwaite 自由度
        df = self._welch_df(control_std, treatment_std, n_control, n_treatment)
        t_critical = stats.t.ppf(1 - self.significance_level / 2, df)

        ci_lower = absolute_diff - t_critical * pooled_se
        ci_upper = absolute_diff + t_critical * pooled_se

        # 转换为相对变化的置信区间
        if control_mean != 0:
            ci_lower_rel = ci_lower / control_mean
            ci_upper_rel = ci_upper / control_mean
        else:
            ci_lower_rel = ci_upper_rel = 0

        # 计算效应量 (Cohen's d)
        pooled_std = math.sqrt(
            ((n_control - 1) * control_std ** 2 + (n_treatment - 1) * treatment_std ** 2)
            / (n_control + n_treatment - 2)
        )
        effect_size = absolute_diff / pooled_std if pooled_std > 0 else 0

        # 计算统计功效
        power = self._calculate_power(effect_size, n_control, n_treatment)

        # 判断结果
        test_result = self._determine_result(
            p_value, relative_diff, higher_is_better
        )

        # 检查护栏指标
        is_guardrail_violated = False
        if is_guardrail:
            if higher_is_better and test_result == TestResult.SIGNIFICANT_NEGATIVE:
                is_guardrail_violated = True
            elif not higher_is_better and test_result == TestResult.SIGNIFICANT_POSITIVE:
                is_guardrail_violated = True

        return StatisticalResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_difference=absolute_diff,
            relative_difference=relative_diff,
            p_value=p_value,
            confidence_interval=(ci_lower_rel, ci_upper_rel),
            confidence_level=self.confidence_level,
            test_result=test_result,
            effect_size=effect_size,
            sample_size_control=n_control,
            sample_size_treatment=n_treatment,
            power=power,
            is_guardrail_violated=is_guardrail_violated
        )

    def proportion_test(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        metric_name: str = "conversion_rate",
        higher_is_better: bool = True,
        is_guardrail: bool = False
    ) -> StatisticalResult:
        """
        比例检验 (z-test for proportions)

        Args:
            control_successes: 对照组成功数
            control_total: 对照组总数
            treatment_successes: 实验组成功数
            treatment_total: 实验组总数
            metric_name: 指标名称
            higher_is_better: 是否越高越好
            is_guardrail: 是否为护栏指标

        Returns:
            StatisticalResult
        """
        # 检查数据量
        if control_total < self.min_sample_size or treatment_total < self.min_sample_size:
            return StatisticalResult(
                metric_name=metric_name,
                control_mean=control_successes / control_total if control_total > 0 else 0,
                treatment_mean=treatment_successes / treatment_total if treatment_total > 0 else 0,
                absolute_difference=0,
                relative_difference=0,
                p_value=1.0,
                confidence_interval=(0, 0),
                confidence_level=self.confidence_level,
                test_result=TestResult.INSUFFICIENT_DATA,
                effect_size=0,
                sample_size_control=control_total,
                sample_size_treatment=treatment_total,
                power=0
            )

        # 计算比例
        p_control = control_successes / control_total
        p_treatment = treatment_successes / treatment_total
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)

        absolute_diff = p_treatment - p_control
        relative_diff = absolute_diff / p_control if p_control > 0 else 0

        # z 检验
        se_pooled = math.sqrt(
            p_pooled * (1 - p_pooled) * (1 / control_total + 1 / treatment_total)
        )
        if se_pooled > 0:
            z_stat = absolute_diff / se_pooled
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1.0

        # 置信区间
        se_diff = math.sqrt(
            (p_control * (1 - p_control) / control_total) +
            (p_treatment * (1 - p_treatment) / treatment_total)
        )
        z_critical = stats.norm.ppf(1 - self.significance_level / 2)
        ci_lower = absolute_diff - z_critical * se_diff
        ci_upper = absolute_diff + z_critical * se_diff

        # 转换为相对变化
        if p_control > 0:
            ci_lower_rel = ci_lower / p_control
            ci_upper_rel = ci_upper / p_control
        else:
            ci_lower_rel = ci_upper_rel = 0

        # 效应量 (Cohen's h)
        effect_size = 2 * (math.asin(math.sqrt(p_treatment)) - math.asin(math.sqrt(p_control)))

        # 功效
        power = self._calculate_power_proportion(
            p_control, p_treatment, control_total, treatment_total
        )

        # 判断结果
        test_result = self._determine_result(p_value, relative_diff, higher_is_better)

        # 检查护栏
        is_guardrail_violated = False
        if is_guardrail:
            if higher_is_better and test_result == TestResult.SIGNIFICANT_NEGATIVE:
                is_guardrail_violated = True
            elif not higher_is_better and test_result == TestResult.SIGNIFICANT_POSITIVE:
                is_guardrail_violated = True

        return StatisticalResult(
            metric_name=metric_name,
            control_mean=p_control,
            treatment_mean=p_treatment,
            absolute_difference=absolute_diff,
            relative_difference=relative_diff,
            p_value=p_value,
            confidence_interval=(ci_lower_rel, ci_upper_rel),
            confidence_level=self.confidence_level,
            test_result=test_result,
            effect_size=effect_size,
            sample_size_control=control_total,
            sample_size_treatment=treatment_total,
            power=power,
            is_guardrail_violated=is_guardrail_violated
        )

    def _welch_df(
        self,
        s1: float,
        s2: float,
        n1: int,
        n2: int
    ) -> float:
        """计算 Welch-Satterthwaite 自由度"""
        v1 = s1 ** 2 / n1
        v2 = s2 ** 2 / n2
        numerator = (v1 + v2) ** 2
        denominator = (v1 ** 2 / (n1 - 1)) + (v2 ** 2 / (n2 - 1))
        return numerator / denominator if denominator > 0 else n1 + n2 - 2

    def _calculate_power(
        self,
        effect_size: float,
        n1: int,
        n2: int
    ) -> float:
        """计算统计功效"""
        if effect_size == 0:
            return 0

        # 简化的功效计算
        se = math.sqrt(1 / n1 + 1 / n2)
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = abs(effect_size) / se - z_alpha

        return stats.norm.cdf(z_beta)

    def _calculate_power_proportion(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int
    ) -> float:
        """计算比例检验的功效"""
        if p1 == p2:
            return 0

        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        se_null = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
        se_alt = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

        if se_null == 0 or se_alt == 0:
            return 0

        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z = (abs(p2 - p1) - z_alpha * se_null) / se_alt

        return stats.norm.cdf(z)

    def _determine_result(
        self,
        p_value: float,
        relative_diff: float,
        higher_is_better: bool
    ) -> TestResult:
        """判断检验结果"""
        if p_value > self.significance_level:
            return TestResult.NOT_SIGNIFICANT

        if higher_is_better:
            if relative_diff > 0:
                return TestResult.SIGNIFICANT_POSITIVE
            else:
                return TestResult.SIGNIFICANT_NEGATIVE
        else:
            # 对于越低越好的指标，下降是正面的
            if relative_diff < 0:
                return TestResult.SIGNIFICANT_POSITIVE
            else:
                return TestResult.SIGNIFICANT_NEGATIVE

    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.8,
        significance_level: float = 0.05
    ) -> int:
        """
        计算所需样本量

        Args:
            baseline_rate: 基线转化率
            minimum_detectable_effect: 最小可检测效应（相对变化）
            power: 统计功效
            significance_level: 显著性水平

        Returns:
            每组所需样本量
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        z_beta = stats.norm.ppf(power)

        p_avg = (p1 + p2) / 2

        numerator = (
            (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
             z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        )
        denominator = (p2 - p1) ** 2

        return math.ceil(numerator / denominator) if denominator > 0 else 0

    def estimate_runtime(
        self,
        required_sample_size: int,
        daily_traffic: int,
        traffic_percentage: float = 1.0,
        num_variants: int = 2
    ) -> int:
        """
        估算实验运行时间

        Args:
            required_sample_size: 每组所需样本量
            daily_traffic: 日均流量
            traffic_percentage: 参与实验的流量比例
            num_variants: 变体数量

        Returns:
            预计运行天数
        """
        samples_per_day = daily_traffic * traffic_percentage / num_variants
        if samples_per_day <= 0:
            return 999

        return math.ceil(required_sample_size / samples_per_day)

    def apply_multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni"
    ) -> List[float]:
        """
        多重比较校正

        Args:
            p_values: 原始 p 值列表
            method: 校正方法 ("bonferroni", "holm", "fdr_bh")

        Returns:
            校正后的 p 值列表
        """
        n = len(p_values)
        if n == 0:
            return []

        if method == "bonferroni":
            return [min(p * n, 1.0) for p in p_values]

        elif method == "holm":
            # Holm-Bonferroni 方法
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros(n)
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (n - i), 1.0)
            return list(corrected)

        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR 控制
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros(n)
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * n / (i + 1), 1.0)
            # 确保单调性
            for i in range(n - 2, -1, -1):
                idx = sorted_indices[i]
                next_idx = sorted_indices[i + 1]
                corrected[idx] = min(corrected[idx], corrected[next_idx])
            return list(corrected)

        return p_values

    def generate_recommendation(
        self,
        primary_results: List[StatisticalResult],
        guardrail_results: List[StatisticalResult]
    ) -> Tuple[str, str, bool, Optional[str]]:
        """
        生成实验建议

        Args:
            primary_results: 主要指标结果
            guardrail_results: 护栏指标结果

        Returns:
            (recommendation, confidence_summary, is_conclusive, winner)
        """
        # 检查护栏指标
        guardrail_violations = [r for r in guardrail_results if r.is_guardrail_violated]
        if guardrail_violations:
            violated_metrics = [r.metric_name for r in guardrail_violations]
            return (
                f"⚠️ 建议不采用实验组：护栏指标违规 ({', '.join(violated_metrics)})",
                "护栏指标出现显著负面变化，建议立即停止实验",
                True,
                "control"
            )

        # 分析主要指标
        significant_positive = [
            r for r in primary_results
            if r.test_result == TestResult.SIGNIFICANT_POSITIVE
        ]
        significant_negative = [
            r for r in primary_results
            if r.test_result == TestResult.SIGNIFICANT_NEGATIVE
        ]
        insufficient = [
            r for r in primary_results
            if r.test_result == TestResult.INSUFFICIENT_DATA
        ]

        if insufficient:
            return (
                "继续收集数据",
                f"数据不足：{len(insufficient)}/{len(primary_results)} 个主要指标无法得出结论",
                False,
                None
            )

        if significant_negative:
            negative_metrics = [r.metric_name for r in significant_negative]
            return (
                f"建议不采用实验组：主要指标下降 ({', '.join(negative_metrics)})",
                f"主要指标中有 {len(significant_negative)} 个显著下降",
                True,
                "control"
            )

        if significant_positive:
            positive_metrics = [r.metric_name for r in significant_positive]
            avg_lift = np.mean([r.relative_difference for r in significant_positive])
            return (
                f"✅ 建议采用实验组：主要指标提升 ({', '.join(positive_metrics)})",
                f"主要指标平均提升 {avg_lift:.1%}，统计显著",
                True,
                "treatment"
            )

        # 不显著
        return (
            "暂无明确结论",
            "主要指标变化不显著，可能需要更长的实验周期或更大的样本量",
            False,
            None
        )


# 测试代码
if __name__ == "__main__":
    import random

    print("=" * 60)
    print("A/B 测试统计分析器测试")
    print("=" * 60)

    analyzer = ABTestAnalyzer(significance_level=0.05)

    # 生成模拟数据
    np.random.seed(42)

    # 场景 1: 显著提升
    print("\n场景 1: 显著提升")
    control_data = np.random.normal(0.85, 0.1, 500).tolist()
    treatment_data = np.random.normal(0.92, 0.1, 500).tolist()

    result = analyzer.two_sample_ttest(
        control_data, treatment_data,
        metric_name="intent_accuracy",
        higher_is_better=True
    )
    print(f"  {result.interpret()}")
    print(f"  效应量: {result.effect_size:.3f}")
    print(f"  功效: {result.power:.3f}")

    # 场景 2: 不显著
    print("\n场景 2: 不显著")
    control_data = np.random.normal(0.85, 0.15, 200).tolist()
    treatment_data = np.random.normal(0.86, 0.15, 200).tolist()

    result = analyzer.two_sample_ttest(
        control_data, treatment_data,
        metric_name="slot_f1",
        higher_is_better=True
    )
    print(f"  {result.interpret()}")

    # 场景 3: 比例检验
    print("\n场景 3: 比例检验 (转化率)")
    result = analyzer.proportion_test(
        control_successes=85,
        control_total=1000,
        treatment_successes=95,
        treatment_total=1000,
        metric_name="order_completion_rate"
    )
    print(f"  对照组: {result.control_mean:.1%}")
    print(f"  实验组: {result.treatment_mean:.1%}")
    print(f"  {result.interpret()}")

    # 场景 4: 样本量计算
    print("\n场景 4: 样本量计算")
    required_n = analyzer.calculate_sample_size(
        baseline_rate=0.85,
        minimum_detectable_effect=0.05,  # 5% 相对提升
        power=0.8
    )
    print(f"  基线转化率: 85%")
    print(f"  最小可检测效应: 5%")
    print(f"  所需每组样本量: {required_n}")

    runtime = analyzer.estimate_runtime(
        required_sample_size=required_n,
        daily_traffic=1000
    )
    print(f"  预计运行天数: {runtime} 天")

    # 场景 5: 多重比较校正
    print("\n场景 5: 多重比较校正")
    p_values = [0.01, 0.03, 0.04, 0.08]
    corrected_bonf = analyzer.apply_multiple_testing_correction(p_values, "bonferroni")
    corrected_bh = analyzer.apply_multiple_testing_correction(p_values, "fdr_bh")

    print(f"  原始 p 值: {p_values}")
    print(f"  Bonferroni 校正: {[round(p, 3) for p in corrected_bonf]}")
    print(f"  BH FDR 校正: {[round(p, 3) for p in corrected_bh]}")

    print("\n✅ 统计分析器测试完成!")
