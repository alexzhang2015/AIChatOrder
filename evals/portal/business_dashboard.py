"""
业务看板

面向业务团队的可视化看板，提供:
- 核心指标卡片
- 趋势图表数据
- 失败分析摘要
- 改进建议
- 业务影响预估
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from evals.harness.business_reporter import (
    BusinessReporter,
    BusinessWeeklyReport,
    MetricStatus,
    TrendDirection
)
from evals.metrics.business_impact import BusinessImpactCalculator, BusinessAssumptions


class WidgetType(Enum):
    """看板组件类型"""
    METRIC_CARD = "metric_card"
    TREND_CHART = "trend_chart"
    PIE_CHART = "pie_chart"
    TABLE = "table"
    ALERT_LIST = "alert_list"
    ACTION_LIST = "action_list"


@dataclass
class DashboardWidget:
    """看板组件"""
    id: str
    title: str
    widget_type: WidgetType
    data: Dict[str, Any]
    size: str = "medium"  # small, medium, large, full
    position: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "type": self.widget_type.value,
            "data": self.data,
            "size": self.size,
            "position": self.position
        }


@dataclass
class DashboardView:
    """看板视图"""
    title: str
    updated_at: str
    widgets: List[DashboardWidget]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "updated_at": self.updated_at,
            "widgets": [w.to_dict() for w in self.widgets],
            "summary": self.summary
        }


class BusinessDashboard:
    """
    业务看板

    提供面向业务的可视化数据，包括：
    1. 核心指标概览
    2. 趋势变化
    3. 问题诊断
    4. 改进建议
    """

    def __init__(
        self,
        results_dir: str = "evals/results",
        history_days: int = 30
    ):
        """
        初始化业务看板

        Args:
            results_dir: 评估结果目录
            history_days: 历史数据天数
        """
        self.results_dir = Path(results_dir)
        self.history_days = history_days
        self.reporter = BusinessReporter(results_dir=results_dir)

        # 模拟历史数据（实际应从数据库获取）
        self._history_cache: Dict[str, List[Dict]] = {}

    def get_dashboard(
        self,
        current_metrics: Optional[Dict[str, float]] = None,
        previous_metrics: Optional[Dict[str, float]] = None
    ) -> DashboardView:
        """
        获取完整的看板视图

        Args:
            current_metrics: 当前指标
            previous_metrics: 上期指标

        Returns:
            DashboardView
        """
        # 如果没有提供指标，使用模拟数据
        if current_metrics is None:
            current_metrics = self._get_mock_current_metrics()
        if previous_metrics is None:
            previous_metrics = self._get_mock_previous_metrics()

        widgets = []

        # 1. 核心指标卡片
        widgets.append(self._create_metrics_cards(current_metrics, previous_metrics))

        # 2. 趋势图表
        widgets.append(self._create_trend_chart())

        # 3. 失败原因分布
        widgets.append(self._create_failure_distribution())

        # 4. 告警列表
        widgets.append(self._create_alerts(current_metrics))

        # 5. 改进建议
        widgets.append(self._create_action_items(current_metrics, previous_metrics))

        # 6. 业务影响预估
        widgets.append(self._create_business_impact(current_metrics))

        # 计算汇总
        summary = self._calculate_summary(current_metrics, previous_metrics)

        return DashboardView(
            title="AI 点单 Agent 业务看板",
            updated_at=datetime.now().isoformat(),
            widgets=widgets,
            summary=summary
        )

    def _create_metrics_cards(
        self,
        current: Dict[str, float],
        previous: Dict[str, float]
    ) -> DashboardWidget:
        """创建指标卡片"""
        cards = []

        metric_display = [
            ("intent_accuracy", "订单理解准确率", "primary"),
            ("slot_f1", "订单详情准确率", "info"),
            ("order_completion_rate", "点单成功率", "success"),
            ("escalation_rate", "人工介入率", "warning"),
            ("avg_turns", "平均点单轮数", "secondary"),
        ]

        for metric_name, display_name, color in metric_display:
            if metric_name not in current:
                continue

            status = self.reporter.calculate_metric_status(
                metric_name,
                current[metric_name],
                previous.get(metric_name)
            )

            cards.append({
                "name": display_name,
                "value": status.to_dict()["current"],
                "trend": status.to_dict()["trend"],
                "status": status.status,
                "status_emoji": status.status_emoji,
                "target": status.to_dict()["target"],
                "color": color
            })

        return DashboardWidget(
            id="metrics_cards",
            title="核心业务指标",
            widget_type=WidgetType.METRIC_CARD,
            data={"cards": cards},
            size="full",
            position=0
        )

    def _create_trend_chart(self) -> DashboardWidget:
        """创建趋势图表数据"""
        # 生成最近 14 天的模拟数据
        dates = []
        intent_accuracy = []
        order_completion = []
        escalation_rate = []

        base_date = datetime.now()
        for i in range(14, 0, -1):
            date = base_date - timedelta(days=i)
            dates.append(date.strftime("%m-%d"))

            # 模拟有波动的数据
            import random
            intent_accuracy.append(round(0.90 + random.uniform(-0.03, 0.05), 3))
            order_completion.append(round(0.82 + random.uniform(-0.04, 0.06), 3))
            escalation_rate.append(round(0.12 + random.uniform(-0.03, 0.03), 3))

        return DashboardWidget(
            id="trend_chart",
            title="指标趋势（近 14 天）",
            widget_type=WidgetType.TREND_CHART,
            data={
                "labels": dates,
                "datasets": [
                    {
                        "label": "订单理解准确率",
                        "data": intent_accuracy,
                        "color": "#4CAF50"
                    },
                    {
                        "label": "点单成功率",
                        "data": order_completion,
                        "color": "#2196F3"
                    },
                    {
                        "label": "人工介入率",
                        "data": escalation_rate,
                        "color": "#FF9800"
                    }
                ]
            },
            size="large",
            position=1
        )

    def _create_failure_distribution(self) -> DashboardWidget:
        """创建失败原因分布"""
        # 模拟失败分布数据
        distribution = [
            {"name": "理解错用户意图", "value": 35, "color": "#f44336"},
            {"name": "订单详情搞错", "value": 25, "color": "#ff9800"},
            {"name": "听不懂口语表达", "value": 20, "color": "#2196f3"},
            {"name": "商品规则搞错", "value": 12, "color": "#9c27b0"},
            {"name": "用户主动放弃", "value": 8, "color": "#607d8b"},
        ]

        return DashboardWidget(
            id="failure_distribution",
            title="失败原因分布",
            widget_type=WidgetType.PIE_CHART,
            data={
                "items": distribution,
                "total_failures": 100
            },
            size="medium",
            position=2
        )

    def _create_alerts(self, metrics: Dict[str, float]) -> DashboardWidget:
        """创建告警列表"""
        alerts = []

        # 检查各指标是否触发告警
        config = self.reporter.METRIC_CONFIG

        for metric_name, value in metrics.items():
            if metric_name not in config:
                continue

            cfg = config[metric_name]
            thresholds = cfg.get("thresholds", {})
            inverse = cfg.get("inverse", False)
            business_name = cfg.get("business_name", metric_name)

            # 判断是否告警
            if inverse:
                if value > thresholds.get("warning", 1):
                    level = "critical" if value > thresholds.get("warning", 1) * 1.2 else "warning"
                    alerts.append({
                        "level": level,
                        "metric": business_name,
                        "message": f"{business_name}达到 {value:.1%}，高于警戒线",
                        "action": f"建议降至 {thresholds.get('good', 0.1):.1%} 以下"
                    })
            else:
                if value < thresholds.get("warning", 0):
                    level = "critical" if value < thresholds.get("warning", 0) * 0.9 else "warning"
                    alerts.append({
                        "level": level,
                        "metric": business_name,
                        "message": f"{business_name}仅 {value:.1%}，低于目标",
                        "action": f"建议提升至 {thresholds.get('good', 0.9):.1%} 以上"
                    })

        # 按严重程度排序
        alerts.sort(key=lambda x: 0 if x["level"] == "critical" else 1)

        return DashboardWidget(
            id="alerts",
            title="告警提醒",
            widget_type=WidgetType.ALERT_LIST,
            data={"alerts": alerts},
            size="medium",
            position=3
        )

    def _create_action_items(
        self,
        current: Dict[str, float],
        previous: Dict[str, float]
    ) -> DashboardWidget:
        """创建改进建议"""
        actions = []

        # 基于指标变化生成建议
        for metric_name, current_value in current.items():
            prev_value = previous.get(metric_name)
            if prev_value is None:
                continue

            config = self.reporter.METRIC_CONFIG.get(metric_name, {})
            business_name = config.get("business_name", metric_name)
            target = config.get("target", 0.95)
            inverse = config.get("inverse", False)

            # 计算差距
            gap = target - current_value if not inverse else current_value - target

            if gap > 0.05:  # 距离目标超过 5%
                priority = "high" if gap > 0.10 else "medium"

                # 生成具体建议
                if metric_name == "intent_accuracy":
                    suggestion = "增加意图边界训练数据，特别是 ORDER_NEW 和 RECOMMEND 的区分"
                elif metric_name == "slot_f1":
                    suggestion = "强化槽位提取规则，检查产品名和属性的识别"
                elif metric_name == "escalation_rate":
                    suggestion = "优化对话引导策略，减少用户迷惑导致的转人工"
                elif metric_name == "avg_turns":
                    suggestion = "简化点单流程，支持一句话下单"
                else:
                    suggestion = f"提升{business_name}"

                actions.append({
                    "priority": priority,
                    "metric": business_name,
                    "current_gap": f"{gap:.1%}",
                    "suggestion": suggestion,
                    "expected_impact": self._estimate_impact(metric_name, gap)
                })

        # 按优先级排序
        actions.sort(key=lambda x: 0 if x["priority"] == "high" else 1)

        return DashboardWidget(
            id="action_items",
            title="改进建议",
            widget_type=WidgetType.ACTION_LIST,
            data={"actions": actions[:5]},  # 最多显示 5 条
            size="large",
            position=4
        )

    def _create_business_impact(self, metrics: Dict[str, float]) -> DashboardWidget:
        """创建业务影响预估"""
        calculator = BusinessImpactCalculator()
        target_metrics = {
            name: cfg.get("target", 0.95)
            for name, cfg in self.reporter.METRIC_CONFIG.items()
        }

        report = calculator.generate_report(metrics, target_metrics)

        impact_items = []
        for impact in report.impacts:
            # 获取财务影响
            financial_str = impact.financial_impact
            # 解析财务影响数字（从字符串如 "¥120,000" 中提取）
            try:
                financial_num = float(financial_str.replace("¥", "").replace(",", "").replace("元", ""))
            except (ValueError, AttributeError):
                financial_num = 0

            if financial_num != 0:
                impact_items.append({
                    "metric": impact.metric_name_cn,
                    "current": f"{impact.from_value:.1%}",
                    "target": f"{impact.to_value:.1%}",
                    "monthly_impact": financial_str,
                    "description": impact.explanation
                })

        return DashboardWidget(
            id="business_impact",
            title="业务影响预估",
            widget_type=WidgetType.TABLE,
            data={
                "columns": ["指标", "当前", "目标", "月影响", "说明"],
                "rows": impact_items,
                "summary": {
                    "total_gmv_impact": f"¥{report.summary.get('estimated_monthly_gmv_impact', 0):,.0f}",
                    "total_cost_saved": f"¥{report.summary.get('estimated_monthly_cost_saved', 0):,.0f}",
                    "total_value": f"¥{report.summary.get('estimated_total_monthly_value', 0):,.0f}"
                }
            },
            size="full",
            position=5
        )

    def _calculate_summary(
        self,
        current: Dict[str, float],
        previous: Dict[str, float]
    ) -> Dict[str, Any]:
        """计算汇总数据"""
        # 健康度评分（0-100）
        health_score = 0
        total_weight = 0

        weights = {
            "intent_accuracy": 30,
            "slot_f1": 20,
            "order_completion_rate": 25,
            "escalation_rate": 15,
            "avg_turns": 10
        }

        for metric, weight in weights.items():
            if metric not in current:
                continue

            config = self.reporter.METRIC_CONFIG.get(metric, {})
            target = config.get("target", 0.95)
            inverse = config.get("inverse", False)

            if inverse:
                # 越低越好
                score = max(0, min(100, (target / current[metric]) * 100)) if current[metric] > 0 else 100
            else:
                # 越高越好
                score = max(0, min(100, (current[metric] / target) * 100)) if target > 0 else 0

            health_score += score * weight
            total_weight += weight

        health_score = health_score / total_weight if total_weight > 0 else 0

        # 趋势分析
        improving = 0
        declining = 0
        for metric in current:
            if metric in previous:
                config = self.reporter.METRIC_CONFIG.get(metric, {})
                inverse = config.get("inverse", False)
                diff = current[metric] - previous[metric]
                if inverse:
                    diff = -diff  # 反转
                if diff > 0.01:
                    improving += 1
                elif diff < -0.01:
                    declining += 1

        # 确定健康状态
        if health_score >= 90:
            health_status = "excellent"
            health_emoji = "🟢"
            health_text = "优秀"
        elif health_score >= 75:
            health_status = "good"
            health_emoji = "🟢"
            health_text = "良好"
        elif health_score >= 60:
            health_status = "warning"
            health_emoji = "🟡"
            health_text = "需关注"
        else:
            health_status = "critical"
            health_emoji = "🔴"
            health_text = "需改进"

        return {
            "health_score": round(health_score, 1),
            "health_status": health_status,
            "health_emoji": health_emoji,
            "health_text": health_text,
            "metrics_improving": improving,
            "metrics_declining": declining,
            "metrics_stable": len(current) - improving - declining,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _estimate_impact(self, metric_name: str, gap: float) -> str:
        """估算改进影响"""
        assumptions = BusinessAssumptions()

        if metric_name == "intent_accuracy":
            orders_affected = int(assumptions.monthly_orders * gap)
            return f"预计每月减少 {orders_affected:,} 单识别错误"
        elif metric_name == "slot_f1":
            orders_affected = int(assumptions.monthly_orders * gap * 0.5)
            return f"预计每月减少 {orders_affected:,} 单详情错误"
        elif metric_name == "escalation_rate":
            cost_saved = assumptions.monthly_orders * gap * assumptions.human_cost_per_order
            return f"预计每月节省人工成本 ¥{cost_saved:,.0f}"
        elif metric_name == "avg_turns":
            return "预计提升用户体验和效率"
        else:
            return "预计提升整体服务质量"

    def _get_mock_current_metrics(self) -> Dict[str, float]:
        """获取模拟当前指标"""
        return {
            "intent_accuracy": 0.92,
            "slot_f1": 0.88,
            "order_completion_rate": 0.83,
            "escalation_rate": 0.12,
            "avg_turns": 4.5
        }

    def _get_mock_previous_metrics(self) -> Dict[str, float]:
        """获取模拟上期指标"""
        return {
            "intent_accuracy": 0.89,
            "slot_f1": 0.85,
            "order_completion_rate": 0.80,
            "escalation_rate": 0.15,
            "avg_turns": 5.2
        }

    def export_dashboard_json(self, dashboard: DashboardView) -> str:
        """导出看板为 JSON"""
        return json.dumps(dashboard.to_dict(), ensure_ascii=False, indent=2)


# =============================================================================
# 便捷函数
# =============================================================================

def get_business_dashboard(
    current_metrics: Optional[Dict[str, float]] = None,
    previous_metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    获取业务看板数据

    Args:
        current_metrics: 当前指标
        previous_metrics: 上期指标

    Returns:
        看板数据字典
    """
    dashboard = BusinessDashboard()
    view = dashboard.get_dashboard(current_metrics, previous_metrics)
    return view.to_dict()


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("业务看板测试")
    print("=" * 60)

    dashboard = BusinessDashboard()

    # 获取看板视图
    view = dashboard.get_dashboard()

    print(f"\n看板标题: {view.title}")
    print(f"更新时间: {view.updated_at}")
    print(f"组件数量: {len(view.widgets)}")

    print("\n--- 组件列表 ---")
    for widget in view.widgets:
        print(f"  [{widget.widget_type.value}] {widget.title} (size: {widget.size})")

    print("\n--- 汇总信息 ---")
    print(f"  健康度评分: {view.summary['health_score']}")
    print(f"  健康状态: {view.summary['health_emoji']} {view.summary['health_text']}")
    print(f"  指标趋势: ↑{view.summary['metrics_improving']} ↓{view.summary['metrics_declining']} →{view.summary['metrics_stable']}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
