"""
槽位提取评分器

评估槽位提取的准确性，支持:
- Precision / Recall / F1
- 各槽位单独评估
- 标准化检查
"""

from collections import defaultdict
from typing import Dict, List, Any, Optional, Set

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


class SlotGrader(BaseGrader):
    """槽位提取评分器"""

    grader_type = GraderType.SLOT_F1

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        min_precision: float = 0.90,
        min_recall: float = 0.90,
        check_fields: Optional[List[str]] = None
    ):
        super().__init__(config)
        self.min_precision = config.min_precision if config and config.min_precision else min_precision
        self.min_recall = config.min_recall if config and config.min_recall else min_recall
        self.check_fields = config.check_fields if config and config.check_fields else check_fields

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估槽位提取准确性

        Args:
            predictions: 模型预测结果，每项包含 slots dict
            test_cases: 测试用例，每项包含 expected_slots

        Returns:
            GraderResult
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        per_slot_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        failures = []

        for pred, test_case in zip(predictions, test_cases):
            pred_slots = pred.get("slots", {})
            exp_slots = test_case.expected_slots or {}

            # 确定要检查的字段
            if self.check_fields:
                fields_to_check = set(self.check_fields)
            else:
                fields_to_check = set(pred_slots.keys()) | set(exp_slots.keys())

            for slot_name in fields_to_check:
                pred_val = pred_slots.get(slot_name)
                exp_val = exp_slots.get(slot_name)

                # 初始化统计
                if slot_name not in per_slot_stats:
                    per_slot_stats[slot_name] = {"tp": 0, "fp": 0, "fn": 0}

                # 比较值
                match = self._compare_slot_values(pred_val, exp_val)

                if match and pred_val is not None:
                    # True Positive
                    total_tp += 1
                    per_slot_stats[slot_name]["tp"] += 1
                elif pred_val is not None and exp_val is None:
                    # False Positive (预测了不应该有的槽位)
                    total_fp += 1
                    per_slot_stats[slot_name]["fp"] += 1
                    failures.append({
                        "type": "false_positive",
                        "input": test_case.input,
                        "slot": slot_name,
                        "predicted": pred_val,
                        "expected": None
                    })
                elif pred_val is None and exp_val is not None:
                    # False Negative (漏掉了应该有的槽位)
                    total_fn += 1
                    per_slot_stats[slot_name]["fn"] += 1
                    failures.append({
                        "type": "false_negative",
                        "input": test_case.input,
                        "slot": slot_name,
                        "predicted": None,
                        "expected": exp_val
                    })
                elif not match:
                    # 值不匹配
                    total_fp += 1
                    total_fn += 1
                    per_slot_stats[slot_name]["fp"] += 1
                    per_slot_stats[slot_name]["fn"] += 1
                    failures.append({
                        "type": "value_mismatch",
                        "input": test_case.input,
                        "slot": slot_name,
                        "predicted": pred_val,
                        "expected": exp_val
                    })

        # 计算 Precision / Recall / F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 计算各槽位指标
        per_slot_metrics = {}
        for slot_name, stats in per_slot_stats.items():
            p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
            r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            per_slot_metrics[slot_name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f, 4)
            }

        # 检查是否通过
        passed = precision >= self.min_precision and recall >= self.min_recall

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=f1,
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "per_slot_metrics": per_slot_metrics,
                "thresholds": {
                    "min_precision": self.min_precision,
                    "min_recall": self.min_recall
                }
            },
            failures=failures
        )

    def _compare_slot_values(self, pred: Any, exp: Any) -> bool:
        """比较槽位值是否匹配"""
        if pred is None and exp is None:
            return True
        if pred is None or exp is None:
            return False

        # 处理列表类型 (如 extras)
        if isinstance(exp, list) and isinstance(pred, list):
            return set(pred) == set(exp)

        # 处理数值类型
        if isinstance(exp, (int, float)) and isinstance(pred, (int, float)):
            return abs(pred - exp) < 0.001

        # 字符串比较 (忽略大小写和空格)
        if isinstance(exp, str) and isinstance(pred, str):
            return pred.strip().lower() == exp.strip().lower()

        return pred == exp
