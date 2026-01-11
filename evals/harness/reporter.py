"""
评估报告生成器

生成多种格式的评估报告:
- 控制台输出
- JSON 报告
- Markdown 报告
- HTML 报告（可选）
"""

import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from evals.harness.models import EvalResult, EvalSuiteResult, Trial, TaskCategory


class EvalReporter:
    """评估报告生成器"""

    def __init__(self, output_dir: str = "evals/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_task_result(self, result: EvalResult, verbose: bool = False):
        """打印单个任务结果到控制台"""
        status = "✅ PASS" if result.pass_all >= 1.0 else "❌ FAIL"
        print(f"\n{'='*60}")
        print(f"任务: {result.task_name} ({result.task_id})")
        print(f"状态: {status}")
        print(f"通过率: pass@1={result.pass_at_1:.0%}, pass@k={result.pass_at_k:.0%}, pass_all={result.pass_all:.0%}")

        if result.aggregate_metrics:
            print(f"\n指标:")
            for key, value in result.aggregate_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        if verbose and result.trials:
            print(f"\n试验详情:")
            for trial in result.trials:
                trial_status = "✅" if trial.passed else "❌"
                print(f"  Trial {trial.trial_number + 1}: {trial_status} ({trial.duration_ms:.0f}ms)")

                for grader_type, grader_result in trial.grader_results.items():
                    if hasattr(grader_result, 'score'):
                        gr_status = "✅" if grader_result.passed else "❌"
                        print(f"    {grader_type}: {gr_status} (score={grader_result.score:.4f})")

                # 显示失败详情
                if not trial.passed:
                    for grader_type, grader_result in trial.grader_results.items():
                        if hasattr(grader_result, 'failures') and grader_result.failures:
                            print(f"    失败 ({grader_type}):")
                            for failure in grader_result.failures[:3]:  # 最多显示3个
                                print(f"      - {failure.get('type', 'unknown')}: {failure.get('input', '')[:50]}")

    def print_suite_result(self, result: EvalSuiteResult, verbose: bool = False):
        """打印套件结果到控制台"""
        print(f"\n{'='*60}")
        print(f"评估套件: {result.suite_name}")
        print(f"{'='*60}")
        print(f"总任务数: {result.tasks_total}")
        print(f"通过: {result.tasks_passed}")
        print(f"失败: {result.tasks_failed}")
        print(f"通过率: {result.overall_pass_rate:.1%}")
        print(f"总耗时: {result.total_duration_ms/1000:.2f}s")

        if result.by_category:
            print(f"\n按类别统计:")
            for category, stats in result.by_category.items():
                print(f"  {category}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.0%})")

        print(f"\n任务详情:")
        for task_result in result.results:
            status = "✅" if task_result.pass_all >= 1.0 else "❌"
            print(f"  {status} {task_result.task_id}: {task_result.task_name}")

            if verbose and task_result.pass_all < 1.0:
                # 显示失败的试验
                for trial in task_result.trials:
                    if not trial.passed:
                        for grader_type, grader_result in trial.grader_results.items():
                            if hasattr(grader_result, 'failures') and grader_result.failures:
                                for failure in grader_result.failures[:2]:
                                    print(f"      - {failure.get('type', 'unknown')}")

        print(f"\n{'='*60}")

    def generate_markdown_report(
        self,
        result: EvalSuiteResult,
        filename: Optional[str] = None
    ) -> str:
        """生成 Markdown 格式报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# 评估报告: {result.suite_name}",
            f"",
            f"**生成时间**: {timestamp}",
            f"",
            f"## 概览",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总任务数 | {result.tasks_total} |",
            f"| 通过 | {result.tasks_passed} |",
            f"| 失败 | {result.tasks_failed} |",
            f"| 通过率 | {result.overall_pass_rate:.1%} |",
            f"| 总耗时 | {result.total_duration_ms/1000:.2f}s |",
            f"",
        ]

        # 按类别统计
        if result.by_category:
            lines.extend([
                f"## 按类别统计",
                f"",
                f"| 类别 | 通过/总数 | 通过率 |",
                f"|------|----------|--------|",
            ])
            for category, stats in result.by_category.items():
                lines.append(f"| {category} | {stats['passed']}/{stats['total']} | {stats['pass_rate']:.0%} |")
            lines.append("")

        # 任务详情
        lines.extend([
            f"## 任务详情",
            f"",
        ])

        for task_result in result.results:
            status = "✅ PASS" if task_result.pass_all >= 1.0 else "❌ FAIL"
            lines.extend([
                f"### {task_result.task_id}: {task_result.task_name}",
                f"",
                f"**状态**: {status}",
                f"",
                f"- pass@1: {task_result.pass_at_1:.0%}",
                f"- pass@k: {task_result.pass_at_k:.0%}",
                f"- pass_all: {task_result.pass_all:.0%}",
                f"",
            ])

            # 失败详情
            if task_result.pass_all < 1.0:
                lines.append("**失败详情**:")
                lines.append("")
                for trial in task_result.trials:
                    if not trial.passed:
                        for grader_type, grader_result in trial.grader_results.items():
                            if hasattr(grader_result, 'failures') and grader_result.failures:
                                lines.append(f"- {grader_type}:")
                                for failure in grader_result.failures[:5]:
                                    lines.append(f"  - {failure.get('type', 'unknown')}: `{str(failure.get('input', ''))[:50]}`")
                lines.append("")

        content = "\n".join(lines)

        # 保存文件
        if filename:
            filepath = self.output_dir / filename
        else:
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"report_{result.suite_name}_{timestamp_file}.md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return str(filepath)

    def generate_json_report(
        self,
        result: EvalSuiteResult,
        filename: Optional[str] = None
    ) -> str:
        """生成 JSON 格式报告"""
        if filename:
            filepath = self.output_dir / filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"report_{result.suite_name}_{timestamp}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        return str(filepath)

    def generate_summary(self, results: List[EvalSuiteResult]) -> str:
        """生成多个套件的汇总报告"""
        lines = [
            "# 评估汇总报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 套件概览",
            "",
            "| 套件 | 通过率 | 通过/总数 | 耗时 |",
            "|------|--------|----------|------|",
        ]

        for result in results:
            lines.append(
                f"| {result.suite_name} | {result.overall_pass_rate:.1%} | "
                f"{result.tasks_passed}/{result.tasks_total} | "
                f"{result.total_duration_ms/1000:.2f}s |"
            )

        lines.append("")

        return "\n".join(lines)
