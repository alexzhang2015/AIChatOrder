"""
业务报告生成器

从评估结果生成面向业务的报告，使用业务语言而非技术术语
"""

import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from evals.metrics.business_impact import (
    BusinessImpactCalculator,
    BusinessAssumptions,
    format_impact_report_markdown
)


class TrendDirection(Enum):
    """趋势方向"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class MetricStatus:
    """指标状态"""
    name: str
    name_cn: str
    current: float
    previous: Optional[float] = None
    target: Optional[float] = None
    trend: Optional[TrendDirection] = None
    trend_value: Optional[float] = None
    status: str = "unknown"  # excellent, good, warning, critical
    status_emoji: str = "⚪"
    is_percentage: bool = True  # 是否为百分比指标

    def to_dict(self) -> dict:
        # 格式化当前值
        if self.is_percentage:
            current_str = f"{self.current:.1%}"
            prev_str = f"{self.previous:.1%}" if self.previous else "-"
            target_str = f"{self.target:.1%}" if self.target else "-"
            trend_str = self._format_trend_percentage()
        else:
            current_str = f"{self.current:.1f}"
            prev_str = f"{self.previous:.1f}" if self.previous else "-"
            target_str = f"{self.target:.1f}" if self.target else "-"
            trend_str = self._format_trend_absolute()

        return {
            "name": self.name,
            "name_cn": self.name_cn,
            "current": current_str,
            "previous": prev_str,
            "target": target_str,
            "trend": trend_str,
            "status": self.status,
            "status_emoji": self.status_emoji
        }

    def _format_trend_percentage(self) -> str:
        """格式化百分比趋势"""
        if self.trend_value is None:
            return "-"
        arrow = "↑" if self.trend == TrendDirection.UP else "↓" if self.trend == TrendDirection.DOWN else "→"
        sign = "+" if self.trend_value > 0 else ""
        return f"{arrow} {sign}{self.trend_value:.1%}"

    def _format_trend_absolute(self) -> str:
        """格式化绝对值趋势"""
        if self.trend_value is None:
            return "-"
        arrow = "↑" if self.trend == TrendDirection.UP else "↓" if self.trend == TrendDirection.DOWN else "→"
        sign = "+" if self.trend_value > 0 else ""
        return f"{arrow} {sign}{self.trend_value:.1f}"


@dataclass
class FailureCategory:
    """失败分类"""
    category: str
    category_cn: str
    count: int
    percentage: float
    typical_cases: List[str]
    recommended_action: str


@dataclass
class BusinessWeeklyReport:
    """业务周报"""
    report_date: str
    period: str  # e.g., "2024-01-08 ~ 2024-01-14"

    # 核心业务指标
    business_metrics: Dict[str, MetricStatus]

    # 失败分析
    failure_analysis: List[FailureCategory]

    # 本周优化效果
    improvements: List[Dict[str, Any]]

    # 下周建议
    recommendations: List[str]

    # 业务影响汇总
    business_impact_summary: Dict[str, Any]


class BusinessReporter:
    """
    业务报告生成器

    功能:
    1. 从评估结果提取业务指标
    2. 分析失败原因（使用业务语言）
    3. 生成周报
    4. 计算业务影响
    """

    # 指标配置
    METRIC_CONFIG = {
        "intent_accuracy": {
            "name_cn": "意图识别准确率",
            "business_name": "订单理解准确率",
            "target": 0.95,
            "thresholds": {"excellent": 0.95, "good": 0.90, "warning": 0.85}
        },
        "slot_f1": {
            "name_cn": "槽位提取 F1",
            "business_name": "订单详情准确率",
            "target": 0.92,
            "thresholds": {"excellent": 0.92, "good": 0.88, "warning": 0.82}
        },
        "order_completion_rate": {
            "name_cn": "订单完成率",
            "business_name": "点单成功率",
            "target": 0.85,
            "thresholds": {"excellent": 0.90, "good": 0.85, "warning": 0.75}
        },
        "first_call_resolution": {
            "name_cn": "首次解决率",
            "business_name": "AI 自主完成率",
            "target": 0.90,
            "thresholds": {"excellent": 0.95, "good": 0.90, "warning": 0.80}
        },
        "escalation_rate": {
            "name_cn": "转人工率",
            "business_name": "人工介入率",
            "target": 0.10,
            "thresholds": {"excellent": 0.05, "good": 0.10, "warning": 0.15},
            "inverse": True  # 越低越好
        },
        "avg_turns": {
            "name_cn": "平均对话轮数",
            "business_name": "平均点单轮数",
            "target": 4.0,
            "thresholds": {"excellent": 3.0, "good": 4.0, "warning": 6.0},
            "inverse": True,
            "is_percentage": False  # 非百分比指标
        }
    }

    # 失败原因映射（技术原因 → 业务语言）
    FAILURE_MAPPING = {
        "intent_confusion": {
            "category_cn": "理解错用户意图",
            "description": "系统误解了用户想要做什么",
            "action": "增加意图边界训练数据"
        },
        "slot_extraction_error": {
            "category_cn": "订单详情搞错",
            "description": "商品名、杯型、温度等信息提取错误",
            "action": "强化槽位提取规则"
        },
        "context_lost": {
            "category_cn": "对话上下文丢失",
            "description": "没有记住之前说的内容",
            "action": "优化对话状态管理"
        },
        "fuzzy_expression_miss": {
            "category_cn": "听不懂口语表达",
            "description": "用户用网络用语或口语，系统没听懂",
            "action": "扩充网络用语/口语映射"
        },
        "constraint_violation": {
            "category_cn": "商品规则搞错",
            "description": "如星冰乐给了热的",
            "action": "完善产品约束规则"
        },
        "user_abandon": {
            "category_cn": "用户主动放弃",
            "description": "对话太长或多次没听懂导致用户放弃",
            "action": "优化对话效率 + 错误恢复"
        }
    }

    def __init__(
        self,
        results_dir: str = "evals/results",
        metrics_config_path: str = "evals/metrics/business_metrics.yaml",
        assumptions: Optional[BusinessAssumptions] = None
    ):
        """
        初始化报告生成器

        Args:
            results_dir: 评估结果目录
            metrics_config_path: 业务指标配置路径
            assumptions: 业务假设参数
        """
        self.results_dir = Path(results_dir)
        self.metrics_config_path = Path(metrics_config_path)
        self.assumptions = assumptions or BusinessAssumptions()
        self.impact_calculator = BusinessImpactCalculator(self.assumptions)

        # 加载指标配置
        self._load_metrics_config()

    def _load_metrics_config(self):
        """加载业务指标配置"""
        if self.metrics_config_path.exists():
            with open(self.metrics_config_path, "r", encoding="utf-8") as f:
                self.metrics_yaml = yaml.safe_load(f)
        else:
            self.metrics_yaml = {}

    def extract_metrics_from_results(
        self,
        result_file: Optional[str] = None
    ) -> Dict[str, float]:
        """
        从评估结果中提取指标

        Args:
            result_file: 结果文件路径，不传则使用最新的

        Returns:
            指标字典
        """
        if result_file:
            result_path = Path(result_file)
        else:
            # 找最新的结果文件
            result_files = list(self.results_dir.glob("*.json"))
            if not result_files:
                return {}
            result_path = max(result_files, key=lambda p: p.stat().st_mtime)

        with open(result_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        metrics = {}

        # 从不同格式的结果中提取指标
        if "results" in result_data:
            # 套件结果
            for task_result in result_data.get("results", []):
                self._extract_from_task_result(task_result, metrics)
        else:
            # 单任务结果
            self._extract_from_task_result(result_data, metrics)

        return metrics

    def _extract_from_task_result(
        self,
        task_result: Dict,
        metrics: Dict[str, float]
    ):
        """从单个任务结果中提取指标"""
        # 从 trials 中提取
        for trial in task_result.get("trials", []):
            for grader_type, grader_result in trial.get("grader_results", {}).items():
                if grader_type == "intent_accuracy":
                    if "accuracy" in grader_result.get("details", {}):
                        metrics["intent_accuracy"] = grader_result["details"]["accuracy"]
                elif grader_type == "slot_f1":
                    if "f1" in grader_result.get("details", {}):
                        metrics["slot_f1"] = grader_result["details"]["f1"]

            # 从 metrics 中提取
            if "avg_confidence" in trial.get("metrics", {}):
                metrics["avg_confidence"] = trial["metrics"]["avg_confidence"]

        # 从 aggregate_metrics 中提取
        agg = task_result.get("aggregate_metrics", {})
        if "avg_confidence_avg" in agg:
            metrics["avg_confidence"] = agg["avg_confidence_avg"]

    def calculate_metric_status(
        self,
        metric_name: str,
        current_value: float,
        previous_value: Optional[float] = None
    ) -> MetricStatus:
        """
        计算指标状态

        Args:
            metric_name: 指标名称
            current_value: 当前值
            previous_value: 上期值（可选）

        Returns:
            MetricStatus
        """
        config = self.METRIC_CONFIG.get(metric_name, {})
        name_cn = config.get("name_cn", metric_name)
        target = config.get("target")
        thresholds = config.get("thresholds", {})
        inverse = config.get("inverse", False)

        # 计算趋势
        trend = None
        trend_value = None
        if previous_value is not None:
            trend_value = current_value - previous_value
            if abs(trend_value) < 0.001:
                trend = TrendDirection.STABLE
            elif trend_value > 0:
                trend = TrendDirection.DOWN if inverse else TrendDirection.UP
            else:
                trend = TrendDirection.UP if inverse else TrendDirection.DOWN

        # 计算状态
        status = "unknown"
        status_emoji = "⚪"

        if thresholds:
            if inverse:
                # 逆向指标（越低越好）
                if current_value <= thresholds.get("excellent", 0):
                    status, status_emoji = "excellent", "🟢"
                elif current_value <= thresholds.get("good", 0):
                    status, status_emoji = "good", "🟢"
                elif current_value <= thresholds.get("warning", 0):
                    status, status_emoji = "warning", "🟡"
                else:
                    status, status_emoji = "critical", "🔴"
            else:
                # 正向指标（越高越好）
                if current_value >= thresholds.get("excellent", 1):
                    status, status_emoji = "excellent", "🟢"
                elif current_value >= thresholds.get("good", 0):
                    status, status_emoji = "good", "🟢"
                elif current_value >= thresholds.get("warning", 0):
                    status, status_emoji = "warning", "🟡"
                else:
                    status, status_emoji = "critical", "🔴"

        # 判断是否为百分比指标
        is_percentage = config.get("is_percentage", True)

        return MetricStatus(
            name=metric_name,
            name_cn=name_cn,
            current=current_value,
            previous=previous_value,
            target=target,
            trend=trend,
            trend_value=trend_value,
            status=status,
            status_emoji=status_emoji,
            is_percentage=is_percentage
        )

    def analyze_failures(
        self,
        eval_results: Dict[str, Any]
    ) -> List[FailureCategory]:
        """
        分析失败原因

        Args:
            eval_results: 评估结果

        Returns:
            失败分类列表
        """
        failure_counts = {key: 0 for key in self.FAILURE_MAPPING}
        typical_cases = {key: [] for key in self.FAILURE_MAPPING}
        total_failures = 0

        # 分析结果中的失败案例
        for result in eval_results.get("results", []):
            for trial in result.get("trials", []):
                if not trial.get("passed", True):
                    total_failures += 1

                    # 根据 grader_results 分类失败原因
                    grader_results = trial.get("grader_results", {})

                    if "intent_accuracy" in grader_results and not grader_results["intent_accuracy"].get("passed"):
                        failure_counts["intent_confusion"] += 1
                        # 添加典型案例
                        details = grader_results["intent_accuracy"].get("details", {})
                        if "mismatches" in details:
                            for mismatch in details["mismatches"][:2]:
                                typical_cases["intent_confusion"].append(
                                    f"输入: '{mismatch.get('input', '')}', "
                                    f"期望: {mismatch.get('expected', '')}, "
                                    f"实际: {mismatch.get('actual', '')}"
                                )

                    if "slot_f1" in grader_results and not grader_results["slot_f1"].get("passed"):
                        failure_counts["slot_extraction_error"] += 1

                    if "fuzzy_match" in grader_results and not grader_results["fuzzy_match"].get("passed"):
                        failure_counts["fuzzy_expression_miss"] += 1

                    if "constraint_validation" in grader_results and not grader_results["constraint_validation"].get("passed"):
                        failure_counts["constraint_violation"] += 1

        # 转换为 FailureCategory 列表
        categories = []
        for key, count in failure_counts.items():
            if count > 0:
                mapping = self.FAILURE_MAPPING[key]
                percentage = count / total_failures if total_failures > 0 else 0
                categories.append(FailureCategory(
                    category=key,
                    category_cn=mapping["category_cn"],
                    count=count,
                    percentage=percentage,
                    typical_cases=typical_cases[key][:3],  # 最多 3 个案例
                    recommended_action=mapping["action"]
                ))

        # 按数量排序
        categories.sort(key=lambda x: x.count, reverse=True)
        return categories

    def generate_weekly_report(
        self,
        current_metrics: Dict[str, float],
        previous_metrics: Optional[Dict[str, float]] = None,
        eval_results: Optional[Dict[str, Any]] = None,
        improvements: Optional[List[Dict[str, Any]]] = None
    ) -> BusinessWeeklyReport:
        """
        生成业务周报

        Args:
            current_metrics: 本周指标
            previous_metrics: 上周指标
            eval_results: 评估结果（用于失败分析）
            improvements: 本周改进项

        Returns:
            BusinessWeeklyReport
        """
        now = datetime.now()
        week_start = now - timedelta(days=now.weekday())
        week_end = week_start + timedelta(days=6)

        # 计算各指标状态
        business_metrics = {}
        for metric_name, value in current_metrics.items():
            prev_value = previous_metrics.get(metric_name) if previous_metrics else None
            status = self.calculate_metric_status(metric_name, value, prev_value)
            business_metrics[metric_name] = status

        # 分析失败原因
        failure_analysis = []
        if eval_results:
            failure_analysis = self.analyze_failures(eval_results)

        # 生成建议
        recommendations = self._generate_recommendations(business_metrics, failure_analysis)

        # 计算业务影响
        target_metrics = {name: cfg.get("target", 0.95) for name, cfg in self.METRIC_CONFIG.items()}
        impact_report = self.impact_calculator.generate_report(current_metrics, target_metrics)

        return BusinessWeeklyReport(
            report_date=now.strftime("%Y-%m-%d"),
            period=f"{week_start.strftime('%Y-%m-%d')} ~ {week_end.strftime('%Y-%m-%d')}",
            business_metrics=business_metrics,
            failure_analysis=failure_analysis,
            improvements=improvements or [],
            recommendations=recommendations,
            business_impact_summary=impact_report.summary
        )

    def _generate_recommendations(
        self,
        metrics: Dict[str, MetricStatus],
        failures: List[FailureCategory]
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于指标状态的建议
        for name, status in metrics.items():
            if status.status in ["warning", "critical"]:
                config = self.METRIC_CONFIG.get(name, {})
                business_name = config.get("business_name", status.name_cn)

                # 根据是否为百分比指标格式化
                if status.is_percentage:
                    current_str = f"{status.current:.1%}"
                    target_str = f"{status.target:.1%}"
                else:
                    current_str = f"{status.current:.1f}"
                    target_str = f"{status.target:.1f}"

                # 根据指标类型选择描述词
                is_inverse = config.get("inverse", False)
                if is_inverse:
                    compare_word = "高于" if status.current > status.target else "接近"
                else:
                    compare_word = "低于"

                recommendations.append(
                    f"【{status.status_emoji} {business_name}】当前 {current_str}，"
                    f"{compare_word}目标 {target_str}，建议重点优化"
                )

        # 基于失败分析的建议
        for failure in failures[:3]:  # 只取前 3 个
            recommendations.append(
                f"【{failure.category_cn}】占失败的 {failure.percentage:.0%}，"
                f"建议：{failure.recommended_action}"
            )

        return recommendations

    def format_report_markdown(self, report: BusinessWeeklyReport) -> str:
        """
        格式化报告为 Markdown
        """
        lines = [
            "# AI 点单 Agent 业务周报",
            "",
            f"**报告日期**: {report.report_date}",
            f"**统计周期**: {report.period}",
            "",
            "---",
            "",
            "## 一、核心业务指标",
            "",
            "| 指标 | 当前值 | 上期值 | 趋势 | 目标值 | 状态 |",
            "|------|--------|--------|------|--------|------|",
        ]

        for name, status in report.business_metrics.items():
            d = status.to_dict()
            lines.append(
                f"| {d['name_cn']} | {d['current']} | {d['previous']} | {d['trend']} | {d['target']} | {d['status_emoji']} {d['status']} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 二、失败原因分析",
            "",
        ])

        if report.failure_analysis:
            for failure in report.failure_analysis:
                lines.extend([
                    f"### {failure.category_cn} ({failure.count} 次, {failure.percentage:.0%})",
                    "",
                ])
                if failure.typical_cases:
                    lines.append("**典型案例**:")
                    for case in failure.typical_cases:
                        lines.append(f"- {case}")
                lines.extend([
                    "",
                    f"**建议操作**: {failure.recommended_action}",
                    "",
                ])
        else:
            lines.append("暂无失败案例分析数据。")

        lines.extend([
            "---",
            "",
            "## 三、本周优化效果",
            "",
        ])

        if report.improvements:
            for imp in report.improvements:
                lines.extend([
                    f"### {imp.get('title', '优化项')}",
                    f"- **内容**: {imp.get('description', '-')}",
                    f"- **效果**: {imp.get('effect', '-')}",
                    f"- **业务影响**: {imp.get('business_impact', '-')}",
                    "",
                ])
        else:
            lines.append("暂无本周优化记录。")

        lines.extend([
            "---",
            "",
            "## 四、下周建议",
            "",
        ])

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.extend([
            "",
            "---",
            "",
            "## 五、业务影响预估",
            "",
            f"- **预估月 GMV 影响**: ¥{report.business_impact_summary.get('estimated_monthly_gmv_impact', 0):,.0f}",
            f"- **预估月成本节省**: ¥{report.business_impact_summary.get('estimated_monthly_cost_saved', 0):,.0f}",
            f"- **预估月总价值**: ¥{report.business_impact_summary.get('estimated_total_monthly_value', 0):,.0f}",
            "",
            "*注：以上为基于当前指标与目标差距的预估，实际影响需根据业务数据验证*",
        ])

        return "\n".join(lines)

    def format_report_json(self, report: BusinessWeeklyReport) -> Dict[str, Any]:
        """
        格式化报告为 JSON
        """
        return {
            "report_date": report.report_date,
            "period": report.period,
            "business_metrics": {
                name: status.to_dict()
                for name, status in report.business_metrics.items()
            },
            "failure_analysis": [
                {
                    "category": f.category,
                    "category_cn": f.category_cn,
                    "count": f.count,
                    "percentage": f"{f.percentage:.1%}",
                    "typical_cases": f.typical_cases,
                    "recommended_action": f.recommended_action
                }
                for f in report.failure_analysis
            ],
            "improvements": report.improvements,
            "recommendations": report.recommendations,
            "business_impact_summary": report.business_impact_summary
        }


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("业务报告生成器测试")
    print("=" * 60)

    # 创建报告生成器
    reporter = BusinessReporter()

    # 模拟指标数据
    current_metrics = {
        "intent_accuracy": 0.92,
        "slot_f1": 0.88,
        "order_completion_rate": 0.85,
        "first_call_resolution": 0.88,
        "escalation_rate": 0.12,
        "avg_turns": 4.5
    }

    previous_metrics = {
        "intent_accuracy": 0.89,
        "slot_f1": 0.85,
        "order_completion_rate": 0.82,
        "first_call_resolution": 0.85,
        "escalation_rate": 0.15,
        "avg_turns": 5.0
    }

    # 模拟改进项
    improvements = [
        {
            "title": "网络用语映射扩展",
            "description": "增加 50 个网络用语映射（续命水、肥宅快乐水等）",
            "effect": "'续命水' 识别率从 60% → 95%",
            "business_impact": "预计每日减少 20 单识别失败"
        }
    ]

    # 生成周报
    report = reporter.generate_weekly_report(
        current_metrics=current_metrics,
        previous_metrics=previous_metrics,
        improvements=improvements
    )

    # 输出 Markdown
    md_report = reporter.format_report_markdown(report)
    print(md_report)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
