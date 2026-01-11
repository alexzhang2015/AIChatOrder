"""
意图分类评分器

评估意图识别的准确率，支持:
- 总体准确率
- 各意图准确率
- 混淆矩阵
- 置信度校准
"""

from collections import defaultdict
from typing import Dict, List, Any, Optional

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


class IntentGrader(BaseGrader):
    """意图分类评分器"""

    grader_type = GraderType.INTENT_ACCURACY

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        min_accuracy: float = 0.95,
        per_intent_min: float = 0.90
    ):
        super().__init__(config)
        self.min_accuracy = config.min_accuracy if config and config.min_accuracy else min_accuracy
        self.per_intent_min = config.per_intent_min if config and config.per_intent_min else per_intent_min

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估意图分类准确率

        Args:
            predictions: 模型预测结果，每项包含 intent, confidence
            test_cases: 测试用例，每项包含 expected_intent

        Returns:
            GraderResult
        """
        if len(predictions) != len(test_cases):
            return GraderResult(
                grader_type=self.grader_type,
                passed=False,
                score=0.0,
                details={"error": "predictions and test_cases length mismatch"},
                failures=[]
            )

        correct = 0
        total = len(predictions)
        per_intent_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        failures = []

        for pred, test_case in zip(predictions, test_cases):
            expected_intent = test_case.expected_intent
            if not expected_intent:
                continue

            predicted_intent = pred.get("intent", "UNKNOWN")
            confidence = pred.get("confidence", 0.0)

            per_intent_stats[expected_intent]["total"] += 1
            confusion_matrix[expected_intent][predicted_intent] += 1

            if predicted_intent == expected_intent:
                correct += 1
                per_intent_stats[expected_intent]["correct"] += 1

                # 检查置信度是否符合预期
                expected_conf = test_case.expected_confidence
                if expected_conf and confidence < expected_conf:
                    failures.append({
                        "type": "low_confidence",
                        "input": test_case.input,
                        "expected_intent": expected_intent,
                        "predicted_intent": predicted_intent,
                        "expected_confidence": expected_conf,
                        "actual_confidence": confidence
                    })
            else:
                failures.append({
                    "type": "wrong_intent",
                    "input": test_case.input,
                    "expected_intent": expected_intent,
                    "predicted_intent": predicted_intent,
                    "confidence": confidence
                })

        # 计算总体准确率
        accuracy = self._compute_score(correct, total)

        # 计算各意图准确率
        per_intent_accuracy = {}
        for intent, stats in per_intent_stats.items():
            if stats["total"] > 0:
                per_intent_accuracy[intent] = stats["correct"] / stats["total"]

        # 检查是否通过
        passed = accuracy >= self.min_accuracy
        for intent, acc in per_intent_accuracy.items():
            if acc < self.per_intent_min:
                passed = False
                break

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=accuracy,
            details={
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "per_intent_accuracy": per_intent_accuracy,
                "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
                "thresholds": {
                    "min_accuracy": self.min_accuracy,
                    "per_intent_min": self.per_intent_min
                }
            },
            failures=failures
        )
