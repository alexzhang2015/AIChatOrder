"""
延迟性能收集器
Latency Performance Collector

收集、分析和报告系统延迟性能指标
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import yaml


class LatencyComponent(str, Enum):
    """延迟组件类型"""
    END_TO_END = "end_to_end"
    INTENT_CLASSIFICATION = "intent_classification"
    SLOT_EXTRACTION = "slot_extraction"
    LLM_GENERATION = "llm_generation"
    RULES_VALIDATION = "rules_validation"
    SKILL_EXECUTION = "skill_execution"
    RAG_RETRIEVAL = "rag_retrieval"


@dataclass
class LatencyMeasurement:
    """单次延迟测量"""
    component: LatencyComponent
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    intent: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PercentileStats:
    """百分位统计"""
    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    mean: float
    std: float
    count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "p50": round(self.p50, 2),
            "p90": round(self.p90, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "count": self.count
        }


@dataclass
class LatencyTarget:
    """延迟目标"""
    p50: float
    p95: float
    p99: float
    critical: Optional[float] = None
    warning: Optional[float] = None


@dataclass
class LatencyReport:
    """延迟报告"""
    overall_stats: PercentileStats
    component_stats: Dict[str, PercentileStats]
    intent_stats: Dict[str, PercentileStats]
    violations: List[Dict[str, Any]]
    grade: str
    health_score: float
    recommendations: List[str]


class LatencyCollector:
    """延迟收集器"""

    def __init__(self, config_path: Optional[str] = None):
        self.measurements: List[LatencyMeasurement] = []
        self.targets = self._load_targets(config_path)
        self.benchmarks = self._load_benchmarks(config_path)
        self._active_timers: Dict[str, float] = {}

    def _load_targets(self, config_path: Optional[str]) -> Dict[str, LatencyTarget]:
        """加载延迟目标配置"""
        if config_path:
            path = Path(config_path) / "latency_requirements.yaml"
        else:
            path = Path(__file__).parent.parent / "metrics" / "latency_requirements.yaml"

        targets = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 端到端目标
            e2e = config.get("end_to_end", {}).get("targets", {})
            targets["end_to_end"] = LatencyTarget(
                p50=e2e.get("p50", 500),
                p95=e2e.get("p95", 1000),
                p99=e2e.get("p99", 2000),
                critical=config.get("end_to_end", {}).get("sla", {}).get("critical", 3000),
                warning=config.get("end_to_end", {}).get("sla", {}).get("warning", 1500)
            )

            # 组件目标
            for comp_name, comp_config in config.get("component_breakdown", {}).items():
                comp_targets = comp_config.get("targets", {})
                targets[comp_name] = LatencyTarget(
                    p50=comp_targets.get("p50", 200),
                    p95=comp_targets.get("p95", 500),
                    p99=comp_targets.get("p99", 1000)
                )
        else:
            # 默认目标
            targets["end_to_end"] = LatencyTarget(p50=500, p95=1000, p99=2000, critical=3000, warning=1500)

        return targets

    def _load_benchmarks(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载行业基准配置"""
        if config_path:
            path = Path(config_path) / "industry_benchmarks.yaml"
        else:
            path = Path(__file__).parent.parent / "metrics" / "industry_benchmarks.yaml"

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def record(
        self,
        component: LatencyComponent,
        duration_ms: float,
        intent: Optional[str] = None,
        success: bool = True,
        **metadata
    ):
        """记录延迟测量"""
        measurement = LatencyMeasurement(
            component=component,
            duration_ms=duration_ms,
            intent=intent,
            success=success,
            metadata=metadata
        )
        self.measurements.append(measurement)

    @contextmanager
    def measure(
        self,
        component: LatencyComponent,
        intent: Optional[str] = None,
        **metadata
    ):
        """上下文管理器方式测量延迟"""
        start = time.perf_counter()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.record(component, duration_ms, intent, success, **metadata)

    def start_timer(self, timer_id: str):
        """开始计时"""
        self._active_timers[timer_id] = time.perf_counter()

    def stop_timer(
        self,
        timer_id: str,
        component: LatencyComponent,
        intent: Optional[str] = None,
        **metadata
    ) -> float:
        """停止计时并记录"""
        if timer_id not in self._active_timers:
            raise ValueError(f"Timer '{timer_id}' not started")

        start = self._active_timers.pop(timer_id)
        duration_ms = (time.perf_counter() - start) * 1000
        self.record(component, duration_ms, intent, **metadata)
        return duration_ms

    def _calculate_percentiles(self, values: List[float]) -> PercentileStats:
        """计算百分位统计"""
        if not values:
            return PercentileStats(
                p50=0, p90=0, p95=0, p99=0,
                min=0, max=0, mean=0, std=0, count=0
            )

        sorted_values = sorted(values)
        n = len(sorted_values)

        def percentile(p: float) -> float:
            idx = (n - 1) * p / 100
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            weight = idx - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

        return PercentileStats(
            p50=percentile(50),
            p90=percentile(90),
            p95=percentile(95),
            p99=percentile(99),
            min=min(sorted_values),
            max=max(sorted_values),
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0,
            count=n
        )

    def get_stats(self, component: Optional[LatencyComponent] = None) -> PercentileStats:
        """获取统计信息"""
        if component:
            values = [m.duration_ms for m in self.measurements if m.component == component]
        else:
            values = [m.duration_ms for m in self.measurements]
        return self._calculate_percentiles(values)

    def get_stats_by_intent(self) -> Dict[str, PercentileStats]:
        """按意图获取统计"""
        intent_values: Dict[str, List[float]] = {}
        for m in self.measurements:
            if m.intent:
                if m.intent not in intent_values:
                    intent_values[m.intent] = []
                intent_values[m.intent].append(m.duration_ms)

        return {
            intent: self._calculate_percentiles(values)
            for intent, values in intent_values.items()
        }

    def check_violations(self) -> List[Dict[str, Any]]:
        """检查 SLA 违反"""
        violations = []

        # 检查端到端延迟
        e2e_stats = self.get_stats(LatencyComponent.END_TO_END)
        e2e_target = self.targets.get("end_to_end")

        if e2e_target and e2e_stats.count > 0:
            if e2e_stats.p95 > e2e_target.p95:
                violations.append({
                    "type": "p95_violation",
                    "component": "end_to_end",
                    "actual": e2e_stats.p95,
                    "target": e2e_target.p95,
                    "severity": "warning"
                })

            if e2e_stats.p99 > e2e_target.p99:
                violations.append({
                    "type": "p99_violation",
                    "component": "end_to_end",
                    "actual": e2e_stats.p99,
                    "target": e2e_target.p99,
                    "severity": "error"
                })

            if e2e_target.critical and e2e_stats.max > e2e_target.critical:
                violations.append({
                    "type": "critical_breach",
                    "component": "end_to_end",
                    "actual": e2e_stats.max,
                    "target": e2e_target.critical,
                    "severity": "critical"
                })

        # 检查各组件延迟
        for comp in LatencyComponent:
            if comp == LatencyComponent.END_TO_END:
                continue

            comp_stats = self.get_stats(comp)
            comp_target = self.targets.get(comp.value)

            if comp_target and comp_stats.count > 0:
                if comp_stats.p95 > comp_target.p95:
                    violations.append({
                        "type": "component_p95_violation",
                        "component": comp.value,
                        "actual": comp_stats.p95,
                        "target": comp_target.p95,
                        "severity": "warning"
                    })

        return violations

    def calculate_health_score(self) -> float:
        """计算延迟健康度评分 (0-100)"""
        e2e_stats = self.get_stats(LatencyComponent.END_TO_END)
        e2e_target = self.targets.get("end_to_end")

        if not e2e_target or e2e_stats.count == 0:
            return 100.0

        score = 100.0

        # P50 评分 (权重 30%)
        if e2e_stats.p50 > e2e_target.p50:
            ratio = e2e_stats.p50 / e2e_target.p50
            score -= min(30, (ratio - 1) * 30)

        # P95 评分 (权重 40%)
        if e2e_stats.p95 > e2e_target.p95:
            ratio = e2e_stats.p95 / e2e_target.p95
            score -= min(40, (ratio - 1) * 40)

        # P99 评分 (权重 30%)
        if e2e_stats.p99 > e2e_target.p99:
            ratio = e2e_stats.p99 / e2e_target.p99
            score -= min(30, (ratio - 1) * 30)

        return max(0, score)

    def get_grade(self, health_score: float) -> str:
        """根据健康度评分获取等级"""
        if health_score >= 95:
            return "A"
        elif health_score >= 90:
            return "B"
        elif health_score >= 85:
            return "C"
        elif health_score >= 80:
            return "D"
        else:
            return "F"

    def generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        violations = self.check_violations()

        # 分析组件延迟
        component_stats = {
            comp: self.get_stats(comp)
            for comp in LatencyComponent
            if self.get_stats(comp).count > 0
        }

        # 找出最慢的组件
        if component_stats:
            slowest = max(
                [(c, s) for c, s in component_stats.items() if c != LatencyComponent.END_TO_END],
                key=lambda x: x[1].p95,
                default=(None, None)
            )

            if slowest[0]:
                comp_name = slowest[0].value
                if slowest[1].p95 > 300:
                    if comp_name == "llm_generation":
                        recommendations.append(
                            f"LLM 生成延迟较高 (P95={slowest[1].p95:.0f}ms)，"
                            "建议：使用流式输出、考虑更快的模型、或添加响应缓存"
                        )
                    elif comp_name == "intent_classification":
                        recommendations.append(
                            f"意图分类延迟较高 (P95={slowest[1].p95:.0f}ms)，"
                            "建议：使用本地小模型、添加热门意图缓存"
                        )
                    elif comp_name == "slot_extraction":
                        recommendations.append(
                            f"槽位提取延迟较高 (P95={slowest[1].p95:.0f}ms)，"
                            "建议：批量处理、使用更快的 NER 模型"
                        )

        # 基于违规生成建议
        critical_violations = [v for v in violations if v["severity"] == "critical"]
        if critical_violations:
            recommendations.append(
                "存在严重延迟违规，建议：检查系统资源、增加服务实例、优化数据库查询"
            )

        # 基于 P99/P50 比值
        e2e_stats = self.get_stats(LatencyComponent.END_TO_END)
        if e2e_stats.count > 0 and e2e_stats.p50 > 0:
            tail_ratio = e2e_stats.p99 / e2e_stats.p50
            if tail_ratio > 4:
                recommendations.append(
                    f"长尾延迟严重 (P99/P50={tail_ratio:.1f}x)，"
                    "建议：检查偶发的慢查询、添加超时控制、优化 GC"
                )

        if not recommendations:
            recommendations.append("延迟性能良好，继续保持！")

        return recommendations

    def generate_report(self) -> LatencyReport:
        """生成完整延迟报告"""
        overall_stats = self.get_stats(LatencyComponent.END_TO_END)
        if overall_stats.count == 0:
            overall_stats = self.get_stats()

        component_stats = {
            comp.value: self.get_stats(comp)
            for comp in LatencyComponent
            if self.get_stats(comp).count > 0
        }

        intent_stats = self.get_stats_by_intent()
        violations = self.check_violations()
        health_score = self.calculate_health_score()
        grade = self.get_grade(health_score)
        recommendations = self.generate_recommendations()

        return LatencyReport(
            overall_stats=overall_stats,
            component_stats=component_stats,
            intent_stats=intent_stats,
            violations=violations,
            grade=grade,
            health_score=health_score,
            recommendations=recommendations
        )

    def compare_with_benchmarks(self) -> Dict[str, Any]:
        """与行业基准对比"""
        comparison = {}

        latency_benchmarks = self.benchmarks.get("latency_benchmarks", {})
        task_oriented = latency_benchmarks.get("task_oriented_dialogue", {})

        e2e_stats = self.get_stats(LatencyComponent.END_TO_END)

        if task_oriented and e2e_stats.count > 0:
            benchmark_p50 = task_oriented.get("p50", 600)
            benchmark_p95 = task_oriented.get("p95", 1200)

            comparison["latency"] = {
                "our_p50": round(e2e_stats.p50, 2),
                "benchmark_p50": benchmark_p50,
                "p50_vs_benchmark": f"{((e2e_stats.p50 / benchmark_p50) - 1) * 100:+.1f}%",
                "our_p95": round(e2e_stats.p95, 2),
                "benchmark_p95": benchmark_p95,
                "p95_vs_benchmark": f"{((e2e_stats.p95 / benchmark_p95) - 1) * 100:+.1f}%",
            }

        return comparison

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        report = self.generate_report()
        return {
            "overall": report.overall_stats.to_dict(),
            "by_component": {
                k: v.to_dict() for k, v in report.component_stats.items()
            },
            "by_intent": {
                k: v.to_dict() for k, v in report.intent_stats.items()
            },
            "violations": report.violations,
            "grade": report.grade,
            "health_score": round(report.health_score, 1),
            "recommendations": report.recommendations,
            "benchmark_comparison": self.compare_with_benchmarks()
        }

    def reset(self):
        """重置所有测量"""
        self.measurements.clear()
        self._active_timers.clear()


class LatencyDecorator:
    """延迟测量装饰器"""

    def __init__(self, collector: LatencyCollector):
        self.collector = collector

    def measure(
        self,
        component: LatencyComponent,
        intent_getter: Optional[Callable] = None
    ):
        """装饰器方式测量函数延迟"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                intent = intent_getter(*args, **kwargs) if intent_getter else None
                with self.collector.measure(component, intent):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# 全局收集器实例
_global_collector: Optional[LatencyCollector] = None


def get_latency_collector() -> LatencyCollector:
    """获取全局延迟收集器"""
    global _global_collector
    if _global_collector is None:
        _global_collector = LatencyCollector()
    return _global_collector


def reset_latency_collector():
    """重置全局延迟收集器"""
    global _global_collector
    if _global_collector:
        _global_collector.reset()


# 便捷函数
def record_latency(
    component: LatencyComponent,
    duration_ms: float,
    intent: Optional[str] = None,
    **metadata
):
    """记录延迟（使用全局收集器）"""
    get_latency_collector().record(component, duration_ms, intent, **metadata)


@contextmanager
def measure_latency(
    component: LatencyComponent,
    intent: Optional[str] = None,
    **metadata
):
    """测量延迟（使用全局收集器）"""
    with get_latency_collector().measure(component, intent, **metadata):
        yield


# 测试代码
if __name__ == "__main__":
    import random

    print("=" * 60)
    print("延迟收集器测试")
    print("=" * 60)

    collector = LatencyCollector()

    # 模拟一些延迟测量
    intents = ["ORDER_NEW", "ORDER_MODIFY", "RECOMMEND", "PRODUCT_INFO", "CHITCHAT"]

    print("\n模拟 100 次请求...")
    for i in range(100):
        intent = random.choice(intents)

        # 端到端延迟
        e2e_base = random.gauss(450, 100)
        # 添加一些长尾
        if random.random() < 0.05:
            e2e_base *= 2.5

        collector.record(
            LatencyComponent.END_TO_END,
            max(100, e2e_base),
            intent=intent
        )

        # 组件延迟
        collector.record(
            LatencyComponent.INTENT_CLASSIFICATION,
            random.gauss(120, 30),
            intent=intent
        )
        collector.record(
            LatencyComponent.SLOT_EXTRACTION,
            random.gauss(80, 20),
            intent=intent
        )
        collector.record(
            LatencyComponent.LLM_GENERATION,
            random.gauss(200, 50),
            intent=intent
        )

    # 生成报告
    report = collector.generate_report()

    print(f"\n整体统计:")
    print(f"  样本数: {report.overall_stats.count}")
    print(f"  P50: {report.overall_stats.p50:.1f}ms")
    print(f"  P90: {report.overall_stats.p90:.1f}ms")
    print(f"  P95: {report.overall_stats.p95:.1f}ms")
    print(f"  P99: {report.overall_stats.p99:.1f}ms")

    print(f"\n健康度: {report.health_score:.1f}/100 (等级: {report.grade})")

    print(f"\n组件延迟 (P95):")
    for comp, stats in report.component_stats.items():
        if comp != "end_to_end":
            print(f"  {comp}: {stats.p95:.1f}ms")

    print(f"\n按意图延迟 (P95):")
    for intent, stats in report.intent_stats.items():
        print(f"  {intent}: {stats.p95:.1f}ms")

    if report.violations:
        print(f"\n⚠️ SLA 违规:")
        for v in report.violations:
            print(f"  [{v['severity']}] {v['type']}: {v['actual']:.1f}ms > {v['target']}ms")

    print(f"\n优化建议:")
    for rec in report.recommendations:
        print(f"  • {rec}")

    # 基准对比
    comparison = collector.compare_with_benchmarks()
    if comparison.get("latency"):
        print(f"\n行业基准对比:")
        lat = comparison["latency"]
        print(f"  P50: {lat['our_p50']:.1f}ms vs {lat['benchmark_p50']}ms ({lat['p50_vs_benchmark']})")
        print(f"  P95: {lat['our_p95']:.1f}ms vs {lat['benchmark_p95']}ms ({lat['p95_vs_benchmark']})")

    print("\n✅ 延迟收集器测试完成!")
