"""
约束验证评分器

评估 CustomizationRulesEngine 对产品约束的验证和自动修正能力
"""

from typing import Dict, List, Any, Optional

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


class ConstraintGrader(BaseGrader):
    """约束验证评分器"""

    grader_type = GraderType.CONSTRAINT_VALIDATION

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        require_correct_auto_fix: bool = True
    ):
        super().__init__(config)
        self.require_correct_auto_fix = require_correct_auto_fix

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估约束验证结果

        Args:
            predictions: 验证结果列表，每项包含:
                - valid: bool
                - adjusted_slots: Dict
                - warnings: List[str]
                - auto_corrections: List[Dict]
            test_cases: 测试用例，expected_behavior 中包含:
                - valid: bool
                - auto_correct_to: Dict (可选)
                - warning_contains: str (可选)

        Returns:
            GraderResult
        """
        correct = 0
        total = len(test_cases)
        failures = []

        for pred, test_case in zip(predictions, test_cases):
            expected = test_case.expected_behavior or {}
            is_correct = True
            failure_reasons = []

            pred_valid = pred.get("valid", True)
            pred_adjusted = pred.get("adjusted_slots", {})
            pred_warnings = pred.get("warnings", [])
            pred_corrections = pred.get("auto_corrections", [])

            # 检查验证结果
            expected_valid = expected.get("valid")
            if expected_valid is not None:
                if pred_valid != expected_valid:
                    is_correct = False
                    failure_reasons.append(f"Validity mismatch: expected {expected_valid}, got {pred_valid}")

            # 检查自动修正
            expected_auto_correct = expected.get("auto_correct_to")
            if expected_auto_correct and self.require_correct_auto_fix:
                for slot_name, expected_value in expected_auto_correct.items():
                    actual_value = pred_adjusted.get(slot_name)
                    if actual_value != expected_value:
                        is_correct = False
                        failure_reasons.append(
                            f"Auto-correct mismatch for {slot_name}: "
                            f"expected {expected_value}, got {actual_value}"
                        )

            # 检查警告信息
            expected_warning = expected.get("warning_contains")
            if expected_warning:
                warning_found = any(expected_warning in w for w in pred_warnings)
                if not warning_found:
                    # 也检查 auto_corrections 中的消息
                    correction_msgs = [c.get("message", "") for c in pred_corrections]
                    warning_found = any(expected_warning in m for m in correction_msgs)

                if not warning_found:
                    is_correct = False
                    failure_reasons.append(f"Expected warning containing '{expected_warning}' not found")

            if is_correct:
                correct += 1
            else:
                failures.append({
                    "type": "constraint_validation_failure",
                    "input": test_case.input,
                    "expected": expected,
                    "actual": {
                        "valid": pred_valid,
                        "adjusted_slots": pred_adjusted,
                        "warnings": pred_warnings,
                        "auto_corrections": pred_corrections
                    },
                    "reasons": failure_reasons
                })

        accuracy = correct / total if total > 0 else 0
        passed = accuracy >= 1.0  # 约束验证要求 100% 正确

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=accuracy,
            details={
                "accuracy": round(accuracy, 4),
                "correct": correct,
                "total": total,
                "require_correct_auto_fix": self.require_correct_auto_fix
            },
            failures=failures
        )
