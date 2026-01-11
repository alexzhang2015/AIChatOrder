"""
模糊表达匹配评分器

评估 FuzzyExpressionMatcher 对口语化表达的理解能力
"""

from typing import Dict, List, Any, Optional

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


class FuzzyMatchGrader(BaseGrader):
    """模糊表达匹配评分器"""

    grader_type = GraderType.FUZZY_MATCH

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        min_accuracy: float = 0.85,
        min_confidence: float = 0.7
    ):
        super().__init__(config)
        self.min_accuracy = config.min_match_accuracy if config and config.min_match_accuracy else min_accuracy
        self.min_confidence = min_confidence

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估模糊表达匹配准确性

        Args:
            predictions: 匹配结果列表，每项包含:
                - matched: bool
                - slot_name: str
                - value: Any
                - confidence: float
            test_cases: 测试用例，metadata 中包含:
                - expected_slot: str
                - expected_value: Any
                - min_confidence: float (可选)

        Returns:
            GraderResult
        """
        correct = 0
        total = len(test_cases)
        failures = []

        for pred, test_case in zip(predictions, test_cases):
            metadata = test_case.metadata or {}
            expected_slot = metadata.get("expected_slot")
            expected_value = metadata.get("expected_value")
            expected_action = metadata.get("expected_action")
            expected_mappings = metadata.get("expected_mappings")
            case_min_conf = metadata.get("min_confidence", self.min_confidence)

            # 检查匹配结果
            matched = pred.get("matched", False)
            pred_slot = pred.get("slot_name")
            pred_value = pred.get("value")
            pred_confidence = pred.get("confidence", 0)
            pred_action = pred.get("action")
            pred_mappings = pred.get("extra_mappings", {})

            is_correct = True
            failure_reasons = []

            # 检查是否应该匹配
            if expected_slot or expected_value or expected_action or expected_mappings:
                if not matched:
                    is_correct = False
                    failure_reasons.append("Expected match but got no match")
                else:
                    # 检查槽位名
                    if expected_slot and pred_slot != expected_slot:
                        is_correct = False
                        failure_reasons.append(f"Slot mismatch: expected {expected_slot}, got {pred_slot}")

                    # 检查值
                    if expected_value and pred_value != expected_value:
                        is_correct = False
                        failure_reasons.append(f"Value mismatch: expected {expected_value}, got {pred_value}")

                    # 检查动作
                    if expected_action and pred_action != expected_action:
                        is_correct = False
                        failure_reasons.append(f"Action mismatch: expected {expected_action}, got {pred_action}")

                    # 检查额外映射
                    if expected_mappings:
                        for k, v in expected_mappings.items():
                            if pred_mappings.get(k) != v:
                                is_correct = False
                                failure_reasons.append(f"Mapping mismatch for {k}: expected {v}, got {pred_mappings.get(k)}")

                    # 检查置信度
                    if pred_confidence < case_min_conf:
                        is_correct = False
                        failure_reasons.append(f"Low confidence: {pred_confidence} < {case_min_conf}")

            if is_correct:
                correct += 1
            else:
                failures.append({
                    "type": "fuzzy_match_failure",
                    "input": test_case.input,
                    "expected": {
                        "slot": expected_slot,
                        "value": expected_value,
                        "action": expected_action,
                        "mappings": expected_mappings
                    },
                    "actual": {
                        "matched": matched,
                        "slot": pred_slot,
                        "value": pred_value,
                        "confidence": pred_confidence,
                        "action": pred_action,
                        "mappings": pred_mappings
                    },
                    "reasons": failure_reasons
                })

        accuracy = correct / total if total > 0 else 0
        passed = accuracy >= self.min_accuracy

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=accuracy,
            details={
                "accuracy": round(accuracy, 4),
                "correct": correct,
                "total": total,
                "min_accuracy_required": self.min_accuracy
            },
            failures=failures
        )
