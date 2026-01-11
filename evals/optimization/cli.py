#!/usr/bin/env python3
"""
优化流程 CLI 工具

提供命令行接口管理 AI 点单 Agent 的持续优化流程:
- 数据增强
- Embedding 知识库管理
- Bad Case 分析
- 优化追踪
- 生产数据收集
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def cmd_augment(args):
    """数据增强命令"""
    from evals.optimization.data_augmenter import DataAugmenter

    print("=" * 60)
    print("数据增强")
    print("=" * 60)

    augmenter = DataAugmenter()

    if args.action == "generate":
        # 生成训练样本
        print(f"\n生成模板样本 (每意图 {args.num} 个)...")
        template_examples = augmenter.generate_from_templates(num_per_intent=args.num)
        print(f"  生成 {len(template_examples)} 个模板样本")

        # 生成边界案例
        print(f"\n生成边界案例 ({args.edge_cases} 个)...")
        edge_cases = augmenter.generate_edge_cases(num_cases=args.edge_cases)
        augmenter.generated_examples.extend(edge_cases)
        print(f"  生成 {len(edge_cases)} 个边界案例")

        # 统计
        stats = augmenter.generate_statistics()
        print(f"\n总样本数: {stats['total_examples']}")
        print("\n意图分布:")
        for intent, count in sorted(stats["by_intent"].items()):
            print(f"  {intent}: {count}")

        # 导出
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            training_file = augmenter.export_training_data(
                output_dir / "training_examples.jsonl"
            )
            print(f"\n训练数据已导出: {training_file}")

            embedding_file = augmenter.export_embedding_data(
                output_dir / "embedding_data.json"
            )
            print(f"Embedding 数据已导出: {embedding_file}")

    elif args.action == "stats":
        # 显示现有统计
        augmenter.generate_from_templates(num_per_intent=10)
        stats = augmenter.generate_statistics()

        print(f"\n商品语料数: {stats['product_corpus_size']}")
        print(f"商品变体总数: {stats['total_product_variants']}")

        print("\n商品别名示例:")
        for product, corpus in list(augmenter.product_corpus.items())[:3]:
            print(f"  {product}: {', '.join(corpus.all_variants()[:5])}...")


def cmd_embedding(args):
    """Embedding 知识库管理命令"""
    from evals.optimization.embedding_builder import (
        EmbeddingKnowledgeBase, build_default_knowledge_base
    )

    print("=" * 60)
    print("Embedding 知识库管理")
    print("=" * 60)

    if args.action == "build":
        print("\n构建默认知识库...")
        kb = build_default_knowledge_base()
        kb.print_summary()

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            chroma_file = kb.export_for_chroma(output_dir / "chroma_export.json")
            print(f"\nChroma 格式导出: {chroma_file}")

            ft_file = kb.export_for_fine_tuning(output_dir / "fine_tuning.jsonl")
            print(f"Fine-tuning 格式导出: {ft_file}")

    elif args.action == "stats":
        kb = EmbeddingKnowledgeBase()
        kb.print_summary()

    elif args.action == "search":
        if not args.query:
            print("错误: 搜索需要提供 --query 参数")
            return

        kb = EmbeddingKnowledgeBase()
        print(f"\n搜索: {args.query}")
        results = kb.search(args.query, top_k=args.top_k)

        if not results:
            print("未找到相关结果")
            return

        for doc, score in results:
            print(f"\n[{doc.layer}] score={score:.3f}")
            print(f"  {doc.text[:80]}...")
            print(f"  metadata: {doc.metadata}")


def cmd_analyze(args):
    """Bad Case 分析命令"""
    from evals.optimization.badcase_collector import BadCaseCollector
    from evals.optimization.badcase_analyzer import BadCaseAnalyzer

    print("=" * 60)
    print("Bad Case 分析")
    print("=" * 60)

    collector = BadCaseCollector()
    analyzer = BadCaseAnalyzer()

    # 加载 Bad Cases
    badcases = collector.load()

    if not badcases:
        print("\n暂无 Bad Case 数据")

        # 检查是否有评估结果可以导入
        results_dir = Path("evals/results")
        if results_dir.exists():
            print(f"\n提示: 可以从 {results_dir} 导入评估结果")
        return

    print(f"\n加载 {len(badcases)} 个 Bad Case")

    # 分析
    report = analyzer.analyze(badcases)

    print(f"\n总 Bad Case 数: {report.total_cases}")

    print("\n按类别分布:")
    for category, count in sorted(report.by_category.items(), key=lambda x: -x[1]):
        pct = count / report.total_cases * 100
        print(f"  {category}: {count} ({pct:.1f}%)")

    print("\n按严重程度:")
    for severity, count in sorted(report.by_severity.items(), key=lambda x: -x[1]):
        pct = count / report.total_cases * 100
        print(f"  {severity}: {count} ({pct:.1f}%)")

    print("\n发现的模式:")
    for pattern in report.top_patterns[:5]:
        print(f"  [{pattern.priority}] {pattern.description}")
        print(f"      影响 {pattern.count} 个案例")
        print(f"      建议: {pattern.suggested_fix}")

    print("\n修复建议:")
    for rec in report.recommendations[:3]:
        print(f"\n  策略: {rec.strategy}")
        print(f"    描述: {rec.description}")
        print(f"    影响: {len(rec.affected_cases)} 个案例")
        print(f"    工作量: {rec.effort}, 影响: {rec.impact}")
        print(f"    优先级分数: {rec.priority_score:.2f}")

    print("\n洞察:")
    for insight in report.insights:
        print(f"  - {insight}")

    # 导出报告
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "generated_at": datetime.now().isoformat(),
            "total_cases": report.total_cases,
            "by_category": report.by_category,
            "by_severity": report.by_severity,
            "patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "count": p.count,
                    "suggested_fix": p.suggested_fix,
                    "priority": p.priority
                }
                for p in report.top_patterns
            ],
            "insights": report.insights
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n报告已导出: {output_file}")


def cmd_track(args):
    """优化追踪命令"""
    from evals.optimization.fix_tracker import FixTracker

    print("=" * 60)
    print("优化追踪")
    print("=" * 60)

    tracker = FixTracker()

    if args.action == "dashboard":
        tracker.print_dashboard()

    elif args.action == "start":
        if not args.version:
            print("错误: 开始新轮次需要提供 --version 参数")
            return

        # 获取基线指标（示例）
        baseline = {
            "intent_accuracy": args.accuracy or 0.85,
            "slot_f1": args.slot_f1 or 0.80,
            "total_badcases": args.badcases or 10
        }

        round_ = tracker.start_round(args.version, baseline)
        print(f"\n开始新一轮优化:")
        print(f"  轮次 ID: {round_.round_id}")
        print(f"  版本: {round_.version}")
        print(f"  开始时间: {round_.started_at}")
        print(f"\n基线指标:")
        for metric, value in baseline.items():
            print(f"  {metric}: {value}")

    elif args.action == "complete":
        current = tracker.get_current_round()
        if not current:
            print("没有进行中的优化轮次")
            return

        # 获取结果指标（示例）
        result = {
            "intent_accuracy": args.accuracy or 0.90,
            "slot_f1": args.slot_f1 or 0.85,
            "total_badcases": args.badcases or 5
        }

        tracker.complete_round(
            round_id=current.round_id,
            result_metrics=result,
            badcases_fixed=args.fixed or 5,
            badcases_new=args.new or 0,
            notes=args.notes or ""
        )

        print(f"\n完成优化轮次: {current.round_id}")
        print(f"\n结果指标:")
        for metric, value in result.items():
            print(f"  {metric}: {value}")

        print(f"\nBad Case 变化:")
        print(f"  修复: {args.fixed or 5}")
        print(f"  新增: {args.new or 0}")

    elif args.action == "trend":
        metric = args.metric or "intent_accuracy"
        trend = tracker.get_trend(metric)

        if not trend:
            print(f"暂无 {metric} 的趋势数据")
            return

        print(f"\n{metric} 趋势:")
        for point in trend:
            direction = "↑" if point["change"] > 0 else "↓" if point["change"] < 0 else "→"
            print(f"  {point['round']} (v{point['version']}): "
                  f"{point['baseline']:.2f} → {point['result']:.2f} "
                  f"({direction} {abs(point['change_pct']):.1f}%)")


def cmd_production(args):
    """生产数据收集命令"""
    from evals.optimization.production_collector import ProductionDataCollector

    print("=" * 60)
    print("生产数据收集")
    print("=" * 60)

    collector = ProductionDataCollector()

    if args.action == "stats":
        collector.print_dashboard()

    elif args.action == "pending":
        pending = collector.get_pending_samples(limit=args.limit)
        print(f"\n待标注样本 ({len(pending)} 个):")
        for sample in pending:
            print(f"\n  [{sample.priority}] {sample.user_input}")
            print(f"    预测: {sample.predicted_intent} ({sample.predicted_confidence:.2f})")
            if sample.user_feedback:
                print(f"    反馈: {sample.user_feedback}")

    elif args.action == "export":
        if not args.output:
            args.output = "evals/optimization/production_data/export.jsonl"

        output_file = collector.export_for_training(
            args.output,
            min_quality=args.min_quality,
            include_auto_labeled=args.include_auto
        )
        print(f"\n训练数据已导出: {output_file}")

    elif args.action == "fine-tune":
        if not args.output:
            args.output = "evals/optimization/production_data/fine_tuning.jsonl"

        output_file = collector.export_fine_tuning_data(
            args.output,
            min_quality=args.min_quality
        )
        print(f"\nFine-tuning 数据已导出: {output_file}")


def cmd_workflow(args):
    """完整优化工作流"""
    print("=" * 60)
    print("AI 点单 Agent 优化工作流")
    print("=" * 60)

    steps = [
        ("1. 运行评估", "python -m evals.run_standalone --suite full"),
        ("2. 收集 Bad Case", "python -m evals.optimization.cli analyze"),
        ("3. 数据增强", "python -m evals.optimization.cli augment generate"),
        ("4. 更新 Embedding", "python -m evals.optimization.cli embedding build"),
        ("5. 重新评估", "python -m evals.run_standalone --suite full"),
        ("6. 查看进度", "python -m evals.optimization.cli track dashboard"),
    ]

    print("\n优化工作流步骤:")
    for step_name, command in steps:
        print(f"\n  {step_name}")
        print(f"    $ {command}")

    print("\n" + "-" * 60)
    print("快速开始:")
    print("  python -m evals.optimization.cli workflow --auto")
    print("\n详细文档: evals/optimization/OPTIMIZATION_STRATEGY.md")


def main():
    parser = argparse.ArgumentParser(
        description="AI 点单 Agent 优化 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # augment 命令
    augment_parser = subparsers.add_parser("augment", help="数据增强")
    augment_parser.add_argument("action", choices=["generate", "stats"],
                                help="操作类型")
    augment_parser.add_argument("--num", type=int, default=30,
                                help="每意图生成样本数")
    augment_parser.add_argument("--edge-cases", type=int, default=30,
                                help="边界案例数")
    augment_parser.add_argument("--output", "-o", help="输出目录")
    augment_parser.set_defaults(func=cmd_augment)

    # embedding 命令
    embedding_parser = subparsers.add_parser("embedding", help="Embedding 知识库管理")
    embedding_parser.add_argument("action", choices=["build", "stats", "search"],
                                  help="操作类型")
    embedding_parser.add_argument("--query", "-q", help="搜索查询")
    embedding_parser.add_argument("--top-k", type=int, default=5, help="返回结果数")
    embedding_parser.add_argument("--output", "-o", help="输出目录")
    embedding_parser.set_defaults(func=cmd_embedding)

    # analyze 命令
    analyze_parser = subparsers.add_parser("analyze", help="Bad Case 分析")
    analyze_parser.add_argument("--input", "-i", help="输入目录")
    analyze_parser.add_argument("--output", "-o", help="输出报告路径")
    analyze_parser.set_defaults(func=cmd_analyze)

    # track 命令
    track_parser = subparsers.add_parser("track", help="优化追踪")
    track_parser.add_argument("action", choices=["dashboard", "start", "complete", "trend"],
                              help="操作类型")
    track_parser.add_argument("--version", "-v", help="版本号")
    track_parser.add_argument("--metric", "-m", help="指标名称")
    track_parser.add_argument("--accuracy", type=float, help="准确率")
    track_parser.add_argument("--slot-f1", type=float, help="槽位 F1")
    track_parser.add_argument("--badcases", type=int, help="Bad Case 数")
    track_parser.add_argument("--fixed", type=int, help="修复数")
    track_parser.add_argument("--new", type=int, help="新增数")
    track_parser.add_argument("--notes", help="备注")
    track_parser.set_defaults(func=cmd_track)

    # production 命令
    prod_parser = subparsers.add_parser("production", help="生产数据收集")
    prod_parser.add_argument("action", choices=["stats", "pending", "export", "fine-tune"],
                             help="操作类型")
    prod_parser.add_argument("--limit", type=int, default=20, help="返回数量限制")
    prod_parser.add_argument("--min-quality", type=float, default=0.5, help="最低质量分")
    prod_parser.add_argument("--include-auto", action="store_true", help="包含自动标注")
    prod_parser.add_argument("--output", "-o", help="输出路径")
    prod_parser.set_defaults(func=cmd_production)

    # workflow 命令
    workflow_parser = subparsers.add_parser("workflow", help="完整优化工作流")
    workflow_parser.add_argument("--auto", action="store_true", help="自动执行")
    workflow_parser.set_defaults(func=cmd_workflow)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
