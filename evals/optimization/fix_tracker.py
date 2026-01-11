"""
优化追踪器

追踪优化轮次、版本历史、效果变化
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional


@dataclass
class OptimizationAction:
    """单个优化动作"""
    action_id: str
    strategy: str
    description: str
    affected_badcases: List[str]
    changes: Dict[str, Any]  # 具体变更内容
    status: str = "pending"  # pending, applied, verified, reverted


@dataclass
class OptimizationRound:
    """优化轮次"""
    round_id: str
    version: str
    started_at: str
    completed_at: Optional[str] = None

    # 优化前指标
    baseline_metrics: Dict[str, float] = field(default_factory=dict)

    # 优化后指标
    result_metrics: Dict[str, float] = field(default_factory=dict)

    # 优化动作
    actions: List[OptimizationAction] = field(default_factory=list)

    # Bad Case 统计
    badcases_before: int = 0
    badcases_after: int = 0
    badcases_fixed: int = 0
    badcases_new: int = 0

    # 备注
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["actions"] = [asdict(a) for a in self.actions]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationRound":
        actions = [OptimizationAction(**a) for a in data.pop("actions", [])]
        return cls(**data, actions=actions)


class FixTracker:
    """修复追踪器"""

    def __init__(self, storage_path: str = "evals/optimization/history"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_path / "optimization_history.json"
        self._history: List[OptimizationRound] = []
        self._load()

    def _load(self):
        """加载历史记录"""
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._history = [OptimizationRound.from_dict(d) for d in data]

    def _save(self):
        """保存历史记录"""
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._history], f, ensure_ascii=False, indent=2)

    def start_round(self, version: str, baseline_metrics: Dict[str, float]) -> OptimizationRound:
        """
        开始新一轮优化

        Args:
            version: 版本号
            baseline_metrics: 基线指标

        Returns:
            新的优化轮次
        """
        round_id = f"round_{len(self._history) + 1:03d}"

        new_round = OptimizationRound(
            round_id=round_id,
            version=version,
            started_at=datetime.now().isoformat(),
            baseline_metrics=baseline_metrics,
            badcases_before=int(baseline_metrics.get("total_badcases", 0))
        )

        self._history.append(new_round)
        self._save()

        return new_round

    def add_action(
        self,
        round_id: str,
        strategy: str,
        description: str,
        affected_badcases: List[str],
        changes: Dict[str, Any]
    ) -> OptimizationAction:
        """
        添加优化动作

        Args:
            round_id: 轮次 ID
            strategy: 修复策略
            description: 描述
            affected_badcases: 受影响的 Bad Case ID
            changes: 具体变更

        Returns:
            优化动作
        """
        for round_ in self._history:
            if round_.round_id == round_id:
                action = OptimizationAction(
                    action_id=f"{round_id}_action_{len(round_.actions) + 1:02d}",
                    strategy=strategy,
                    description=description,
                    affected_badcases=affected_badcases,
                    changes=changes,
                    status="pending"
                )
                round_.actions.append(action)
                self._save()
                return action

        raise ValueError(f"Round not found: {round_id}")

    def mark_action_applied(self, action_id: str):
        """标记动作已应用"""
        for round_ in self._history:
            for action in round_.actions:
                if action.action_id == action_id:
                    action.status = "applied"
                    self._save()
                    return

    def complete_round(
        self,
        round_id: str,
        result_metrics: Dict[str, float],
        badcases_fixed: int,
        badcases_new: int,
        notes: str = ""
    ):
        """
        完成优化轮次

        Args:
            round_id: 轮次 ID
            result_metrics: 结果指标
            badcases_fixed: 修复的 Bad Case 数
            badcases_new: 新增的 Bad Case 数
            notes: 备注
        """
        for round_ in self._history:
            if round_.round_id == round_id:
                round_.completed_at = datetime.now().isoformat()
                round_.result_metrics = result_metrics
                round_.badcases_after = int(result_metrics.get("total_badcases", 0))
                round_.badcases_fixed = badcases_fixed
                round_.badcases_new = badcases_new
                round_.notes = notes
                self._save()
                return

        raise ValueError(f"Round not found: {round_id}")

    def get_current_round(self) -> Optional[OptimizationRound]:
        """获取当前进行中的轮次"""
        for round_ in reversed(self._history):
            if round_.completed_at is None:
                return round_
        return None

    def get_round(self, round_id: str) -> Optional[OptimizationRound]:
        """获取指定轮次"""
        for round_ in self._history:
            if round_.round_id == round_id:
                return round_
        return None

    def get_history(self) -> List[OptimizationRound]:
        """获取全部历史"""
        return self._history

    def get_latest_version(self) -> str:
        """获取最新版本号"""
        if self._history:
            return self._history[-1].version
        return "1.0.0"

    def get_trend(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        获取指标趋势

        Args:
            metric_name: 指标名称

        Returns:
            趋势数据列表
        """
        trend = []
        for round_ in self._history:
            if round_.completed_at:
                baseline_value = round_.baseline_metrics.get(metric_name)
                result_value = round_.result_metrics.get(metric_name)
                if baseline_value is not None and result_value is not None:
                    trend.append({
                        "round": round_.round_id,
                        "version": round_.version,
                        "date": round_.completed_at[:10],
                        "baseline": baseline_value,
                        "result": result_value,
                        "change": result_value - baseline_value,
                        "change_pct": (result_value - baseline_value) / baseline_value * 100
                            if baseline_value > 0 else 0
                    })
        return trend

    def generate_progress_report(self) -> Dict[str, Any]:
        """生成进度报告"""
        if not self._history:
            return {"message": "暂无优化历史"}

        completed_rounds = [r for r in self._history if r.completed_at]

        # 总体统计
        total_fixed = sum(r.badcases_fixed for r in completed_rounds)
        total_new = sum(r.badcases_new for r in completed_rounds)
        total_actions = sum(len(r.actions) for r in completed_rounds)

        # 指标变化
        if len(completed_rounds) >= 2:
            first = completed_rounds[0]
            last = completed_rounds[-1]
            metric_changes = {}
            for metric in first.baseline_metrics:
                if metric in last.result_metrics:
                    initial = first.baseline_metrics[metric]
                    current = last.result_metrics[metric]
                    metric_changes[metric] = {
                        "initial": initial,
                        "current": current,
                        "change": current - initial,
                        "change_pct": (current - initial) / initial * 100 if initial > 0 else 0
                    }
        else:
            metric_changes = {}

        return {
            "total_rounds": len(completed_rounds),
            "current_version": self.get_latest_version(),
            "total_badcases_fixed": total_fixed,
            "total_badcases_new": total_new,
            "net_reduction": total_fixed - total_new,
            "total_actions": total_actions,
            "metric_changes": metric_changes,
            "rounds": [
                {
                    "round_id": r.round_id,
                    "version": r.version,
                    "completed": r.completed_at[:10] if r.completed_at else "进行中",
                    "fixed": r.badcases_fixed,
                    "new": r.badcases_new,
                    "actions": len(r.actions)
                }
                for r in self._history[-5:]  # 最近5轮
            ]
        }

    def print_dashboard(self):
        """打印仪表盘"""
        report = self.generate_progress_report()

        print("\n" + "=" * 60)
        print("📊 AI 点单 Agent 优化进度")
        print("=" * 60)

        if "message" in report:
            print(report["message"])
            return

        print(f"当前版本: {report['current_version']}")
        print(f"优化轮次: {report['total_rounds']}")
        print(f"累计修复: {report['total_badcases_fixed']} Bad Cases")
        print(f"净减少: {report['net_reduction']} Bad Cases")
        print(f"执行动作: {report['total_actions']}")

        if report.get("metric_changes"):
            print("\n📈 指标变化:")
            for metric, change in report["metric_changes"].items():
                direction = "↑" if change["change"] > 0 else "↓" if change["change"] < 0 else "→"
                print(f"  {metric}: {change['initial']:.2f} → {change['current']:.2f} "
                      f"({direction} {abs(change['change_pct']):.1f}%)")

        print("\n📋 最近轮次:")
        for r in report.get("rounds", []):
            print(f"  {r['round_id']} (v{r['version']}): "
                  f"修复 {r['fixed']}, 新增 {r['new']}, "
                  f"{r['actions']} 个动作 - {r['completed']}")

        print("=" * 60)
