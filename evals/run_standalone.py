#!/usr/bin/env python3
"""
独立评估运行器

不依赖完整项目环境，可单独运行评估任务
用于演示和测试评估框架
"""

import sys
import json
import time
import yaml
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evals.harness.models import (
    TaskConfig, TestCase, GraderConfig, GraderType,
    Trial, EvalResult, EvalSuiteResult, GraderResult, TaskCategory
)
from evals.graders.intent_grader import IntentGrader
from evals.graders.slot_grader import SlotGrader
from evals.graders.fuzzy_grader import FuzzyMatchGrader
from evals.graders.constraint_grader import ConstraintGrader
from evals.graders.transcript_grader import TranscriptGrader
from evals.harness.reporter import EvalReporter
from evals.harness.html_reporter import HTMLReporter


# ============ 模拟分类器 ============

class MockClassifier:
    """模拟分类器，用于演示"""

    INTENT_KEYWORDS = {
        "ORDER_NEW": ["来一杯", "我想点", "给我", "要一杯", "点一杯", "来个", "来两杯"],
        "ORDER_MODIFY": ["改成", "换成", "加", "不要", "少", "多"],
        "ORDER_CANCEL": ["取消", "不要了", "算了", "不点了", "不想要"],
        "ORDER_QUERY": ["订单", "到哪了", "查一下", "状态", "准备好"],
        "RECOMMEND": ["推荐", "好喝", "什么好", "新品", "最好卖"],
        "PRODUCT_INFO": ["多少钱", "价格", "卡路里", "热量", "配料"],
        "CHITCHAT": ["天气", "你好", "谢谢", "好的", "嗯"],
        "PAYMENT": ["付款", "支付", "买单", "结账"],
        "COMPLAINT": ["投诉", "做错", "不对", "太慢"],
    }

    SLOT_PATTERNS = {
        "product_name": {
            "拿铁": ["拿铁", "latte"],
            "美式咖啡": ["美式", "americano"],
            "卡布奇诺": ["卡布奇诺", "cappuccino"],
            "摩卡": ["摩卡", "mocha"],
            "星冰乐": ["星冰乐", "frappuccino"],
            "馥芮白": ["馥芮白", "澳白", "flat white"],
            "冷萃咖啡": ["冷萃", "cold brew"],
            "焦糖玛奇朵": ["焦糖玛奇朵", "玛奇朵"],
        },
        "size": {
            "小杯": ["小杯", "tall"],
            "中杯": ["中杯", "grande"],
            "大杯": ["大杯", "venti"],
            "超大杯": ["超大杯", "trenta"],
        },
        "temperature": {
            "热": ["热", "hot"],
            "冰": ["冰", "iced", "冷"],
            "温": ["温", "常温"],
        },
        "sweetness": {
            "无糖": ["无糖", "不加糖"],
            "少糖": ["少糖", "微糖"],
            "半糖": ["半糖"],
            "标准": ["标准糖"],
            "多糖": ["多糖"],
        },
        "milk_type": {
            "牛奶": ["牛奶"],
            "燕麦奶": ["燕麦奶", "燕麦"],
            "豆奶": ["豆奶", "豆浆"],
            "椰奶": ["椰奶"],
        }
    }

    def classify(self, text: str) -> Dict[str, Any]:
        """模拟意图分类"""
        text_lower = text.lower()

        # 识别意图
        intent = "UNKNOWN"
        confidence = 0.5

        for intent_type, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    intent = intent_type
                    confidence = 0.85 + random.uniform(0, 0.1)
                    break
            if intent != "UNKNOWN":
                break

        # 提取槽位
        slots = {}
        for slot_name, values in self.SLOT_PATTERNS.items():
            for normalized, patterns in values.items():
                for pattern in patterns:
                    if pattern in text_lower or pattern in text:
                        slots[slot_name] = normalized
                        break

        # 提取数量
        for i in range(1, 10):
            if f"{i}杯" in text or f"来{i}" in text:
                slots["quantity"] = i
                break
        if "quantity" not in slots and ("一杯" in text or "来一" in text or "来个" in text):
            slots["quantity"] = 1

        # 提取加料
        extras = []
        if "加浓" in text or "浓一点" in text:
            extras.append("加浓")
        if "奶油" in text:
            extras.append("奶油")
        if extras:
            slots["extras"] = extras

        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots
        }

    def validate_constraints(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        """模拟约束验证"""
        product = slots.get("product_name", "")
        temperature = slots.get("temperature", "")

        # 冰饮产品约束
        cold_only = ["星冰乐", "冷萃咖啡"]
        valid = True
        warnings = []
        auto_corrections = []
        adjusted_slots = slots.copy()

        if product in cold_only and temperature == "热":
            valid = False
            adjusted_slots["temperature"] = "冰"
            warnings.append(f"{product}只能做冰的")
            auto_corrections.append({
                "slot": "temperature",
                "from": "热",
                "to": "冰",
                "message": f"{product}只能做冰的，已自动调整为冰的"
            })

        return {
            "valid": valid,
            "adjusted_slots": adjusted_slots,
            "warnings": warnings,
            "auto_corrections": auto_corrections
        }

    def match_fuzzy(self, text: str) -> Dict[str, Any]:
        """模拟模糊匹配"""
        fuzzy_patterns = {
            "不要那么甜|没那么甜": ("sweetness", "半糖", 0.85),
            "甜一点": ("sweetness", "多糖", 0.8),
            "健康一点": ("sweetness", "无糖", 0.75),
            "浓一点|加浓": (None, None, 0.85, "add_extra_shot"),
            "续命水": ("product_name", "美式咖啡", 0.9),
            "常温|不要太烫": ("temperature", "温", 0.8),
        }

        for pattern, result in fuzzy_patterns.items():
            import re
            if re.search(pattern, text):
                if len(result) == 3:
                    return {
                        "matched": True,
                        "slot_name": result[0],
                        "value": result[1],
                        "confidence": result[2],
                        "pattern": pattern,
                        "action": None,
                        "extra_mappings": {}
                    }
                else:
                    return {
                        "matched": True,
                        "slot_name": result[0],
                        "value": result[1],
                        "confidence": result[2],
                        "pattern": pattern,
                        "action": result[3],
                        "extra_mappings": {}
                    }

        return {"matched": False}


# ============ 独立评估运行器 ============

class StandaloneRunner:
    """独立评估运行器"""

    def __init__(self, tasks_dir: str = "evals/tasks", results_dir: str = "evals/results"):
        self.tasks_dir = Path(tasks_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = MockClassifier()
        self.graders = {
            GraderType.INTENT_ACCURACY: IntentGrader,
            GraderType.SLOT_F1: SlotGrader,
            GraderType.FUZZY_MATCH: FuzzyMatchGrader,
            GraderType.CONSTRAINT_VALIDATION: ConstraintGrader,
            GraderType.TRANSCRIPT: TranscriptGrader,
        }

    def load_task(self, task_path: str) -> TaskConfig:
        """加载任务"""
        path = Path(task_path)
        # 如果路径已经是绝对路径或已经存在，直接使用
        if path.is_absolute() or path.exists():
            pass
        elif not str(task_path).startswith(str(self.tasks_dir)):
            path = self.tasks_dir / path

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return TaskConfig.from_dict(data)

    def load_suite(self, suite_name: str) -> List[TaskConfig]:
        """加载套件"""
        tasks = []
        for task_file in self.tasks_dir.rglob("*.yaml"):
            try:
                # 直接用绝对路径读取
                with open(task_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                task = TaskConfig.from_dict(data)

                if suite_name == "all":
                    tasks.append(task)
                elif suite_name == "regression" and task.category == TaskCategory.REGRESSION:
                    tasks.append(task)
                elif suite_name == "capability" and task.category == TaskCategory.CAPABILITY:
                    tasks.append(task)
                elif suite_name == "edge_case" and task.category == TaskCategory.EDGE_CASE:
                    tasks.append(task)
            except Exception as e:
                print(f"⚠️ 加载任务失败 {task_file}: {e}")
        return tasks

    def run_task(self, task: TaskConfig, num_trials: int = 3) -> EvalResult:
        """运行单个任务"""
        print(f"\n📝 运行任务: {task.id} - {task.name}")
        trials = []

        for i in range(num_trials):
            trial = self._run_trial(task, i)
            trials.append(trial)
            status = "✅" if trial.passed else "❌"
            print(f"   Trial {i+1}: {status} ({trial.duration_ms:.0f}ms)")

        # 计算结果
        passes = [t.passed for t in trials]
        result = EvalResult(
            task_id=task.id,
            task_name=task.name,
            category=task.category,
            trials=trials,
            pass_at_1=1.0 if passes[0] else 0.0,
            pass_at_k=1.0 if any(passes) else 0.0,
            pass_all=1.0 if all(passes) else 0.0,
            completed_at=datetime.now().isoformat()
        )

        status = "✅ PASS" if result.pass_all >= 1.0 else "❌ FAIL"
        print(f"   结果: {status} (pass@1={result.pass_at_1:.0%}, pass_all={result.pass_all:.0%})")

        return result

    def _run_trial(self, task: TaskConfig, trial_num: int) -> Trial:
        """运行单次试验"""
        start_time = time.time()

        predictions = []
        grader_results = {}
        all_passed = True

        # 运行测试用例
        for test_case in task.test_cases:
            if isinstance(test_case.input, str):
                # 文本输入 - 意图分类
                result = self.classifier.classify(test_case.input)
                predictions.append({
                    "input": test_case.input,
                    "intent": result["intent"],
                    "confidence": result["confidence"],
                    "slots": result["slots"]
                })
            elif isinstance(test_case.input, dict):
                # 结构化输入 - 约束验证
                result = self.classifier.validate_constraints(test_case.input)
                predictions.append({
                    "input": test_case.input,
                    **result
                })

        # 运行评分器
        for grader_config in task.graders:
            grader_class = self.graders.get(grader_config.type)
            if grader_class:
                grader = grader_class(config=grader_config)
                result = grader.evaluate(predictions, task.test_cases)
                grader_results[grader_config.type.value] = result
                if not result.passed and grader_config.required:
                    all_passed = False

        duration_ms = (time.time() - start_time) * 1000

        return Trial(
            task_id=task.id,
            trial_number=trial_num,
            outcome={"predictions": predictions},
            grader_results=grader_results,
            passed=all_passed,
            duration_ms=duration_ms
        )

    def run_suite(self, suite_name: str, num_trials: int = 3) -> EvalSuiteResult:
        """运行套件"""
        tasks = self.load_suite(suite_name)
        print(f"\n🚀 运行评估套件: {suite_name}")
        print(f"   任务数: {len(tasks)}")
        print(f"   试验次数: {num_trials}")

        results = []
        started_at = datetime.now()

        for task in tasks:
            result = self.run_task(task, num_trials)
            results.append(result)

        # 统计
        tasks_passed = sum(1 for r in results if r.pass_all >= 1.0)
        tasks_failed = len(results) - tasks_passed
        overall_pass_rate = tasks_passed / len(results) if results else 0

        # 按类别统计
        by_category = {}
        for category in TaskCategory:
            cat_results = [r for r in results if r.category == category]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r.pass_all >= 1.0)
                by_category[category.value] = {
                    "total": len(cat_results),
                    "passed": cat_passed,
                    "pass_rate": cat_passed / len(cat_results)
                }

        completed_at = datetime.now()
        total_duration = (completed_at - started_at).total_seconds() * 1000

        return EvalSuiteResult(
            suite_name=suite_name,
            tasks_total=len(results),
            tasks_passed=tasks_passed,
            tasks_failed=tasks_failed,
            overall_pass_rate=overall_pass_rate,
            results=results,
            by_category=by_category,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            total_duration_ms=total_duration
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="独立评估运行器")
    parser.add_argument("--suite", default="all", choices=["all", "regression", "capability", "edge_case"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--report", default="all", choices=["none", "json", "markdown", "html", "all"])

    args = parser.parse_args()

    # 运行评估
    runner = StandaloneRunner()
    result = runner.run_suite(args.suite, args.trials)

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"📊 评估摘要: {result.suite_name}")
    print("=" * 60)
    print(f"总任务数: {result.tasks_total}")
    print(f"通过: {result.tasks_passed}")
    print(f"失败: {result.tasks_failed}")
    print(f"通过率: {result.overall_pass_rate:.1%}")
    print(f"总耗时: {result.total_duration_ms/1000:.2f}s")

    if result.by_category:
        print("\n按类别:")
        for cat, stats in result.by_category.items():
            print(f"  {cat}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.0%})")

    # 生成报告
    if args.report != "none":
        print("\n📄 生成报告...")

        if args.report in ["json", "all"]:
            reporter = EvalReporter()
            json_path = reporter.generate_json_report(result)
            print(f"   JSON: {json_path}")

        if args.report in ["markdown", "all"]:
            reporter = EvalReporter()
            md_path = reporter.generate_markdown_report(result)
            print(f"   Markdown: {md_path}")

        if args.report in ["html", "all"]:
            html_reporter = HTMLReporter()
            html_path = html_reporter.generate_suite_report(result)
            print(f"   HTML: {html_path}")

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
