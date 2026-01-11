"""
混淆矩阵评分器

专门用于检测和告警容易混淆的意图对，支持:
- 构建完整混淆矩阵
- 对特定意图对设置混淆率告警
- 分析混淆模式
"""

from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


class ConfusionMatrixGrader(BaseGrader):
    """
    混淆矩阵评分器

    用于检测意图分类中的混淆问题，特别是容易混淆的意图对
    """

    grader_type = GraderType.CONFUSION_MATRIX

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        alert_pairs: Optional[List[Tuple[str, str]]] = None,
        alert_threshold: float = 0.05
    ):
        """
        初始化混淆矩阵评分器

        Args:
            config: 评分器配置
            alert_pairs: 需要监控的意图对列表，如 [("ORDER_NEW", "RECOMMEND")]
            alert_threshold: 混淆率告警阈值，超过此值则告警
        """
        super().__init__(config)

        # 从配置获取告警意图对
        if config and config.alert_pairs:
            self.alert_pairs = [tuple(pair) for pair in config.alert_pairs]
        else:
            self.alert_pairs = alert_pairs or []

        # 从配置获取告警阈值
        self.alert_threshold = (
            config.alert_threshold if config and config.alert_threshold
            else alert_threshold
        )

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估混淆矩阵

        Args:
            predictions: 模型预测结果
            test_cases: 测试用例

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

        # 构建混淆矩阵
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        intent_totals = defaultdict(int)

        for pred, test_case in zip(predictions, test_cases):
            expected_intent = test_case.expected_intent
            if not expected_intent:
                continue

            predicted_intent = pred.get("intent", "UNKNOWN")

            confusion_matrix[expected_intent][predicted_intent] += 1
            intent_totals[expected_intent] += 1

        # 计算混淆率
        confusion_rates = {}
        for expected, predictions_dict in confusion_matrix.items():
            total = intent_totals[expected]
            if total > 0:
                confusion_rates[expected] = {
                    pred: count / total
                    for pred, count in predictions_dict.items()
                    if pred != expected  # 只计算错误预测
                }

        # 检查告警意图对
        alerts = []
        failures = []

        for pair in self.alert_pairs:
            if len(pair) != 2:
                continue

            expected, confused_with = pair

            # 检查 expected -> confused_with 的混淆率
            rate = confusion_rates.get(expected, {}).get(confused_with, 0.0)
            if rate > self.alert_threshold:
                alerts.append({
                    "pair": [expected, confused_with],
                    "direction": f"{expected} -> {confused_with}",
                    "confusion_rate": rate,
                    "threshold": self.alert_threshold,
                    "count": confusion_matrix[expected][confused_with],
                    "total": intent_totals[expected]
                })
                failures.append({
                    "type": "confusion_alert",
                    "expected_intent": expected,
                    "confused_with": confused_with,
                    "confusion_rate": rate,
                    "message": f"混淆率 {rate:.1%} 超过阈值 {self.alert_threshold:.1%}"
                })

            # 检查反向混淆 confused_with -> expected
            reverse_rate = confusion_rates.get(confused_with, {}).get(expected, 0.0)
            if reverse_rate > self.alert_threshold:
                alerts.append({
                    "pair": [confused_with, expected],
                    "direction": f"{confused_with} -> {expected}",
                    "confusion_rate": reverse_rate,
                    "threshold": self.alert_threshold,
                    "count": confusion_matrix[confused_with][expected],
                    "total": intent_totals[confused_with]
                })
                failures.append({
                    "type": "confusion_alert",
                    "expected_intent": confused_with,
                    "confused_with": expected,
                    "confusion_rate": reverse_rate,
                    "message": f"混淆率 {reverse_rate:.1%} 超过阈值 {self.alert_threshold:.1%}"
                })

        # 计算总体混淆分数（1 - 最大混淆率）
        max_confusion_rate = 0.0
        for rates in confusion_rates.values():
            for rate in rates.values():
                max_confusion_rate = max(max_confusion_rate, rate)

        score = 1.0 - max_confusion_rate

        # 判断是否通过
        passed = len(alerts) == 0

        # 找出最容易混淆的意图对
        top_confusions = self._find_top_confusions(confusion_matrix, intent_totals, top_k=5)

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=score,
            details={
                "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
                "confusion_rates": confusion_rates,
                "intent_totals": dict(intent_totals),
                "alerts": alerts,
                "top_confusions": top_confusions,
                "alert_pairs_monitored": [list(p) for p in self.alert_pairs],
                "alert_threshold": self.alert_threshold
            },
            failures=failures
        )

    def _find_top_confusions(
        self,
        confusion_matrix: Dict[str, Dict[str, int]],
        intent_totals: Dict[str, int],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """找出最容易混淆的意图对"""
        confusions = []

        for expected, predictions_dict in confusion_matrix.items():
            total = intent_totals[expected]
            if total == 0:
                continue

            for predicted, count in predictions_dict.items():
                if predicted != expected and count > 0:
                    rate = count / total
                    confusions.append({
                        "expected": expected,
                        "predicted": predicted,
                        "count": count,
                        "rate": rate
                    })

        # 按混淆率排序
        confusions.sort(key=lambda x: x["rate"], reverse=True)

        return confusions[:top_k]


class SafetyCheckGrader(BaseGrader):
    """
    安全边界检查评分器

    用于验证系统是否正确处理安全相关场景
    """

    grader_type = GraderType.SAFETY_CHECK

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        must_recommend: Optional[List[str]] = None,
        must_not_recommend: Optional[List[str]] = None,
        expected_behavior: Optional[str] = None
    ):
        """
        初始化安全检查评分器

        Args:
            config: 评分器配置
            must_recommend: 必须推荐的选项
            must_not_recommend: 绝不能推荐的选项
            expected_behavior: 期望的行为描述
        """
        super().__init__(config)

        self.must_recommend = (
            config.must_recommend if config and config.must_recommend
            else must_recommend or []
        )
        self.must_not_recommend = (
            config.must_not_recommend if config and config.must_not_recommend
            else must_not_recommend or []
        )
        self.expected_behavior = (
            config.expected_behavior if config and config.expected_behavior
            else expected_behavior
        )

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估安全边界处理

        Args:
            predictions: 模型预测结果
            test_cases: 测试用例
            transcript: 对话记录（用于检查响应内容）

        Returns:
            GraderResult
        """
        failures = []
        checks_passed = 0
        total_checks = 0

        # 检查响应内容
        response_text = ""
        if transcript:
            # 获取所有 agent 响应
            agent_responses = [
                t.get("content", "") for t in transcript
                if t.get("role") == "agent"
            ]
            response_text = " ".join(agent_responses)

        # 检查预测结果中的响应
        for pred in predictions:
            if "response" in pred:
                response_text += " " + pred.get("response", "")
            if "slots" in pred:
                slots = pred.get("slots", {})
                for key, value in slots.items():
                    response_text += f" {value}"

        response_text = response_text.lower()

        # 检查必须推荐的项
        for item in self.must_recommend:
            total_checks += 1
            if item.lower() in response_text:
                checks_passed += 1
            else:
                failures.append({
                    "type": "missing_recommendation",
                    "item": item,
                    "message": f"应该推荐 '{item}' 但未出现在响应中"
                })

        # 检查绝不能推荐的项
        for item in self.must_not_recommend:
            total_checks += 1
            if item.lower() not in response_text:
                checks_passed += 1
            else:
                failures.append({
                    "type": "forbidden_recommendation",
                    "item": item,
                    "message": f"不应该推荐 '{item}' 但出现在响应中"
                })

        # 计算分数
        score = checks_passed / total_checks if total_checks > 0 else 1.0
        passed = len(failures) == 0

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=score,
            details={
                "checks_passed": checks_passed,
                "total_checks": total_checks,
                "must_recommend": self.must_recommend,
                "must_not_recommend": self.must_not_recommend,
                "expected_behavior": self.expected_behavior,
                "response_analyzed": response_text[:500] if response_text else None
            },
            failures=failures
        )
