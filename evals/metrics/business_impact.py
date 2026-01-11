"""
技术指标 → 业务影响映射

将技术指标的变化转换为业务可理解的影响
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml
from pathlib import Path


class ImpactLevel(Enum):
    """影响等级"""
    CRITICAL = "critical"      # 关键影响
    HIGH = "high"              # 高影响
    MEDIUM = "medium"          # 中等影响
    LOW = "low"                # 低影响


@dataclass
class BusinessAssumptions:
    """业务假设参数（可配置）"""
    daily_orders: int = 1000           # 日均订单数
    avg_order_value: float = 40.0      # 平均客单价（元）
    manual_cost_per_order: float = 5.0 # 人工处理成本（元/单）
    operating_days_per_month: int = 30 # 月运营天数

    # 计算属性
    @property
    def monthly_orders(self) -> int:
        return self.daily_orders * self.operating_days_per_month

    @property
    def monthly_gmv(self) -> float:
        return self.monthly_orders * self.avg_order_value


@dataclass
class ImpactMapping:
    """单个指标的影响映射"""
    metric_name: str
    metric_name_cn: str
    from_value: float
    to_value: float
    improvement: float

    # 业务影响
    business_metric: str
    business_impact: str
    financial_impact: str
    impact_level: ImpactLevel

    # 说明
    explanation: str
    assumptions: str


@dataclass
class BusinessImpactReport:
    """业务影响报告"""
    generated_at: str
    assumptions: BusinessAssumptions
    impacts: List[ImpactMapping]
    summary: Dict[str, Any]


class BusinessImpactCalculator:
    """
    业务影响计算器

    将技术指标变化转换为业务影响
    """

    # 指标影响映射规则
    IMPACT_RULES = {
        "intent_accuracy": {
            "name_cn": "意图识别准确率",
            "business_metric": "订单正确率",
            "impact_formula": lambda delta, assumptions: {
                "orders_affected": int(assumptions.monthly_orders * delta),
                "gmv_impact": assumptions.monthly_orders * delta * assumptions.avg_order_value,
                "description": f"每月约 {int(assumptions.monthly_orders * delta)} 单从错误变为正确"
            },
            "level": ImpactLevel.CRITICAL
        },
        "slot_f1": {
            "name_cn": "槽位提取 F1",
            "business_metric": "订单修改率",
            "impact_formula": lambda delta, assumptions: {
                "orders_affected": int(assumptions.monthly_orders * delta * 0.5),  # 假设 50% 转化
                "time_saved_hours": assumptions.monthly_orders * delta * 0.5 * 0.5 / 60,  # 每次修改节省 30 秒
                "description": f"每月减少约 {int(assumptions.monthly_orders * delta * 0.5)} 单需要用户纠正"
            },
            "level": ImpactLevel.HIGH
        },
        "first_call_resolution": {
            "name_cn": "首次解决率",
            "business_metric": "人工介入成本",
            "impact_formula": lambda delta, assumptions: {
                "orders_affected": int(assumptions.monthly_orders * delta),
                "cost_saved": assumptions.monthly_orders * delta * assumptions.manual_cost_per_order,
                "description": f"每月减少 {int(assumptions.monthly_orders * delta)} 单人工介入，节省约 ¥{assumptions.monthly_orders * delta * assumptions.manual_cost_per_order:,.0f}"
            },
            "level": ImpactLevel.HIGH
        },
        "order_completion_rate": {
            "name_cn": "订单完成率",
            "business_metric": "流失订单",
            "impact_formula": lambda delta, assumptions: {
                "orders_recovered": int(assumptions.monthly_orders * delta),
                "gmv_recovered": assumptions.monthly_orders * delta * assumptions.avg_order_value,
                "description": f"每月挽回约 {int(assumptions.monthly_orders * delta)} 单流失订单，GMV 约 ¥{assumptions.monthly_orders * delta * assumptions.avg_order_value:,.0f}"
            },
            "level": ImpactLevel.CRITICAL
        },
        "escalation_rate": {
            "name_cn": "转人工率",
            "business_metric": "人力成本",
            "impact_formula": lambda delta, assumptions: {
                # 注意：转人工率降低是好事，所以 delta 应该是负数表示改善
                "orders_affected": int(assumptions.monthly_orders * abs(delta)),
                "cost_saved": assumptions.monthly_orders * abs(delta) * assumptions.manual_cost_per_order,
                "description": f"每月减少 {int(assumptions.monthly_orders * abs(delta))} 单转人工，节省约 ¥{assumptions.monthly_orders * abs(delta) * assumptions.manual_cost_per_order:,.0f}"
            },
            "level": ImpactLevel.HIGH
        },
        "average_handling_time": {
            "name_cn": "平均处理时长",
            "business_metric": "用户体验/门店效率",
            "impact_formula": lambda delta, assumptions: {
                # delta 是秒数差异（正数表示时间减少）
                "time_saved_per_order": abs(delta),
                "total_hours_saved": assumptions.monthly_orders * abs(delta) / 3600,
                "description": f"每单节省 {abs(delta):.0f} 秒，每月共节省 {assumptions.monthly_orders * abs(delta) / 3600:.1f} 小时"
            },
            "level": ImpactLevel.MEDIUM
        },
        "confusion_rate": {
            "name_cn": "意图混淆率",
            "business_metric": "理解准确性",
            "impact_formula": lambda delta, assumptions: {
                "orders_affected": int(assumptions.monthly_orders * abs(delta)),
                "description": f"每月减少约 {int(assumptions.monthly_orders * abs(delta))} 单意图误判"
            },
            "level": ImpactLevel.MEDIUM
        }
    }

    def __init__(self, assumptions: Optional[BusinessAssumptions] = None):
        """
        初始化计算器

        Args:
            assumptions: 业务假设参数，不传则使用默认值
        """
        self.assumptions = assumptions or BusinessAssumptions()

    def calculate_impact(
        self,
        metric_name: str,
        from_value: float,
        to_value: float
    ) -> Optional[ImpactMapping]:
        """
        计算单个指标变化的业务影响

        Args:
            metric_name: 指标名称
            from_value: 原始值
            to_value: 目标值

        Returns:
            ImpactMapping 或 None（如果指标未定义）
        """
        if metric_name not in self.IMPACT_RULES:
            return None

        rule = self.IMPACT_RULES[metric_name]
        delta = to_value - from_value
        improvement = delta / from_value if from_value > 0 else 0

        # 计算影响
        impact_result = rule["impact_formula"](abs(delta), self.assumptions)

        # 格式化财务影响
        financial_impact = self._format_financial_impact(metric_name, impact_result)

        return ImpactMapping(
            metric_name=metric_name,
            metric_name_cn=rule["name_cn"],
            from_value=from_value,
            to_value=to_value,
            improvement=improvement,
            business_metric=rule["business_metric"],
            business_impact=impact_result["description"],
            financial_impact=financial_impact,
            impact_level=rule["level"],
            explanation=self._generate_explanation(metric_name, from_value, to_value, impact_result),
            assumptions=f"日均 {self.assumptions.daily_orders} 单，客单价 ¥{self.assumptions.avg_order_value}"
        )

    def calculate_multiple_impacts(
        self,
        metrics: Dict[str, Dict[str, float]]
    ) -> List[ImpactMapping]:
        """
        批量计算多个指标的影响

        Args:
            metrics: {"metric_name": {"from": 0.85, "to": 0.95}, ...}

        Returns:
            影响映射列表
        """
        impacts = []
        for metric_name, values in metrics.items():
            impact = self.calculate_impact(
                metric_name,
                values.get("from", 0),
                values.get("to", 0)
            )
            if impact:
                impacts.append(impact)

        # 按影响等级排序
        level_order = {
            ImpactLevel.CRITICAL: 0,
            ImpactLevel.HIGH: 1,
            ImpactLevel.MEDIUM: 2,
            ImpactLevel.LOW: 3
        }
        impacts.sort(key=lambda x: level_order[x.impact_level])

        return impacts

    def generate_report(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> BusinessImpactReport:
        """
        生成完整的业务影响报告

        Args:
            current_metrics: 当前指标值
            target_metrics: 目标指标值

        Returns:
            BusinessImpactReport
        """
        from datetime import datetime

        # 构建输入格式
        metrics_input = {}
        for name in current_metrics:
            if name in target_metrics:
                metrics_input[name] = {
                    "from": current_metrics[name],
                    "to": target_metrics[name]
                }

        impacts = self.calculate_multiple_impacts(metrics_input)

        # 计算汇总
        summary = self._calculate_summary(impacts)

        return BusinessImpactReport(
            generated_at=datetime.now().isoformat(),
            assumptions=self.assumptions,
            impacts=impacts,
            summary=summary
        )

    def _format_financial_impact(
        self,
        metric_name: str,
        impact_result: Dict
    ) -> str:
        """格式化财务影响描述"""
        if "gmv_impact" in impact_result:
            return f"月 GMV 影响: +¥{impact_result['gmv_impact']:,.0f}"
        elif "gmv_recovered" in impact_result:
            return f"月 GMV 挽回: +¥{impact_result['gmv_recovered']:,.0f}"
        elif "cost_saved" in impact_result:
            return f"月成本节省: ¥{impact_result['cost_saved']:,.0f}"
        elif "total_hours_saved" in impact_result:
            return f"月时间节省: {impact_result['total_hours_saved']:.1f} 小时"
        else:
            return "影响待量化"

    def _generate_explanation(
        self,
        metric_name: str,
        from_value: float,
        to_value: float,
        impact_result: Dict
    ) -> str:
        """生成解释说明"""
        delta = to_value - from_value
        direction = "提升" if delta > 0 else "降低"

        # 特殊处理转人工率等逆向指标
        if metric_name in ["escalation_rate", "confusion_rate", "modification_rate"]:
            direction = "降低" if delta < 0 else "提升"

        return (
            f"{self.IMPACT_RULES[metric_name]['name_cn']}从 {from_value:.1%} {direction}到 {to_value:.1%}，"
            f"变化 {abs(delta):.1%}。{impact_result['description']}。"
        )

    def _calculate_summary(self, impacts: List[ImpactMapping]) -> Dict[str, Any]:
        """计算汇总统计"""
        total_gmv_impact = 0
        total_cost_saved = 0
        critical_count = 0
        high_count = 0

        for impact in impacts:
            if impact.impact_level == ImpactLevel.CRITICAL:
                critical_count += 1
            elif impact.impact_level == ImpactLevel.HIGH:
                high_count += 1

            # 尝试从财务影响中提取数字
            if "GMV" in impact.financial_impact:
                try:
                    amount = float(impact.financial_impact.split("¥")[1].replace(",", "").split()[0])
                    total_gmv_impact += amount
                except (IndexError, ValueError):
                    pass
            elif "成本节省" in impact.financial_impact:
                try:
                    amount = float(impact.financial_impact.split("¥")[1].replace(",", "").split()[0])
                    total_cost_saved += amount
                except (IndexError, ValueError):
                    pass

        return {
            "total_impacts": len(impacts),
            "critical_impacts": critical_count,
            "high_impacts": high_count,
            "estimated_monthly_gmv_impact": total_gmv_impact,
            "estimated_monthly_cost_saved": total_cost_saved,
            "estimated_total_monthly_value": total_gmv_impact + total_cost_saved
        }


def format_impact_report_markdown(report: BusinessImpactReport) -> str:
    """
    将影响报告格式化为 Markdown
    """
    lines = [
        "# 业务影响分析报告",
        "",
        f"**生成时间**: {report.generated_at}",
        "",
        "## 业务假设",
        "",
        f"- 日均订单数: {report.assumptions.daily_orders} 单",
        f"- 平均客单价: ¥{report.assumptions.avg_order_value}",
        f"- 人工处理成本: ¥{report.assumptions.manual_cost_per_order}/单",
        f"- 月运营天数: {report.assumptions.operating_days_per_month} 天",
        "",
        "## 指标影响详情",
        "",
    ]

    # 按影响等级分组
    for level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH, ImpactLevel.MEDIUM, ImpactLevel.LOW]:
        level_impacts = [i for i in report.impacts if i.impact_level == level]
        if level_impacts:
            level_emoji = {
                ImpactLevel.CRITICAL: "🔴",
                ImpactLevel.HIGH: "🟠",
                ImpactLevel.MEDIUM: "🟡",
                ImpactLevel.LOW: "🟢"
            }
            lines.append(f"### {level_emoji[level]} {level.value.upper()} 级影响")
            lines.append("")

            for impact in level_impacts:
                lines.extend([
                    f"#### {impact.metric_name_cn}",
                    "",
                    f"- **变化**: {impact.from_value:.1%} → {impact.to_value:.1%} ({'+' if impact.improvement > 0 else ''}{impact.improvement:.1%})",
                    f"- **业务指标**: {impact.business_metric}",
                    f"- **业务影响**: {impact.business_impact}",
                    f"- **财务影响**: {impact.financial_impact}",
                    f"- **假设**: {impact.assumptions}",
                    "",
                ])

    # 汇总
    lines.extend([
        "## 汇总",
        "",
        f"- **总影响项**: {report.summary['total_impacts']} 项",
        f"- **关键影响**: {report.summary['critical_impacts']} 项",
        f"- **高影响**: {report.summary['high_impacts']} 项",
        f"- **预估月 GMV 影响**: ¥{report.summary['estimated_monthly_gmv_impact']:,.0f}",
        f"- **预估月成本节省**: ¥{report.summary['estimated_monthly_cost_saved']:,.0f}",
        f"- **预估月总价值**: ¥{report.summary['estimated_total_monthly_value']:,.0f}",
        "",
    ])

    return "\n".join(lines)


# 便捷函数
def calculate_business_impact(
    metric_name: str,
    from_value: float,
    to_value: float,
    daily_orders: int = 1000,
    avg_order_value: float = 40.0
) -> Optional[ImpactMapping]:
    """
    便捷函数：计算单个指标的业务影响

    Example:
        >>> impact = calculate_business_impact("intent_accuracy", 0.85, 0.95)
        >>> print(impact.financial_impact)
        "月 GMV 影响: +¥120,000"
    """
    assumptions = BusinessAssumptions(
        daily_orders=daily_orders,
        avg_order_value=avg_order_value
    )
    calculator = BusinessImpactCalculator(assumptions)
    return calculator.calculate_impact(metric_name, from_value, to_value)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("业务影响计算器测试")
    print("=" * 60)

    # 创建计算器（使用默认假设）
    calculator = BusinessImpactCalculator()

    # 测试单个指标
    print("\n--- 单指标测试 ---")
    impact = calculator.calculate_impact("intent_accuracy", 0.85, 0.95)
    if impact:
        print(f"指标: {impact.metric_name_cn}")
        print(f"变化: {impact.from_value:.1%} → {impact.to_value:.1%}")
        print(f"业务影响: {impact.business_impact}")
        print(f"财务影响: {impact.financial_impact}")

    # 测试多指标报告
    print("\n--- 完整报告测试 ---")
    current = {
        "intent_accuracy": 0.85,
        "slot_f1": 0.80,
        "first_call_resolution": 0.80,
        "order_completion_rate": 0.75,
        "escalation_rate": 0.20
    }
    target = {
        "intent_accuracy": 0.95,
        "slot_f1": 0.92,
        "first_call_resolution": 0.95,
        "order_completion_rate": 0.90,
        "escalation_rate": 0.10
    }

    report = calculator.generate_report(current, target)

    # 输出 Markdown 报告
    md_report = format_impact_report_markdown(report)
    print(md_report)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
