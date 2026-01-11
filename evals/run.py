"""
评估框架 CLI 入口

用法:
    # 运行回归测试套件
    python -m evals.run --suite regression

    # 运行能力测试套件
    python -m evals.run --suite capability --trials 5

    # 运行单个任务
    python -m evals.run --task intent/basic_intent_classification.yaml

    # 运行所有任务
    python -m evals.run --suite all --verbose

    # 设置失败阈值
    python -m evals.run --suite regression --fail-threshold 0.95
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evals.harness.runner import EvalRunner
from evals.harness.reporter import EvalReporter


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(
        description="AI 点单系统评估框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m evals.run --suite regression
  python -m evals.run --task intent/basic_intent_classification.yaml
  python -m evals.run --suite all --verbose --report markdown
        """
    )

    # 任务选择
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--suite",
        choices=["regression", "capability", "edge_case", "all"],
        help="运行评估套件"
    )
    task_group.add_argument(
        "--task",
        type=str,
        help="运行单个任务 (相对于 evals/tasks 的路径)"
    )

    # 执行参数
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="每个任务的试验次数 (默认: 3)"
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.0,
        help="失败阈值，低于此值返回非零退出码 (默认: 0)"
    )

    # 输出参数
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    parser.add_argument(
        "--report",
        choices=["none", "json", "markdown", "both"],
        default="json",
        help="报告格式 (默认: json)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="仅生成报告，不阻塞 CI (即使失败也返回 0)"
    )

    # 目录参数
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="evals/tasks",
        help="任务目录 (默认: evals/tasks)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evals/results",
        help="结果目录 (默认: evals/results)"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.verbose)

    # 创建执行器和报告器
    runner = EvalRunner(
        tasks_dir=args.tasks_dir,
        results_dir=args.results_dir
    )
    reporter = EvalReporter(output_dir=args.results_dir)

    exit_code = 0

    try:
        if args.suite:
            # 运行套件
            print(f"\n🚀 开始运行评估套件: {args.suite}")
            print(f"   试验次数: {args.trials}")
            print(f"   失败阈值: {args.fail_threshold:.0%}")
            print()

            result = runner.run_suite(
                suite_name=args.suite,
                num_trials=args.trials,
                fail_threshold=args.fail_threshold,
                save_transcript=True
            )

            # 打印结果
            reporter.print_suite_result(result, verbose=args.verbose)

            # 生成报告
            if args.report in ["json", "both"]:
                json_path = reporter.generate_json_report(result)
                print(f"\n📄 JSON 报告: {json_path}")

            if args.report in ["markdown", "both"]:
                md_path = reporter.generate_markdown_report(result)
                print(f"📄 Markdown 报告: {md_path}")

            # 检查阈值
            if args.fail_threshold > 0 and result.overall_pass_rate < args.fail_threshold:
                if not args.report_only:
                    exit_code = 1
                    print(f"\n❌ 评估失败: 通过率 {result.overall_pass_rate:.1%} < 阈值 {args.fail_threshold:.1%}")
                else:
                    print(f"\n⚠️ 通过率低于阈值，但使用了 --report-only，不阻塞")

        else:
            # 运行单个任务
            print(f"\n🚀 开始运行任务: {args.task}")
            print(f"   试验次数: {args.trials}")
            print()

            task = runner.load_task(args.task)
            result = runner.run_task(task, num_trials=args.trials)

            # 打印结果
            reporter.print_task_result(result, verbose=args.verbose)

            if result.pass_all < 1.0 and not args.report_only:
                exit_code = 1

    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ 执行错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit_code = 1

    print()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
