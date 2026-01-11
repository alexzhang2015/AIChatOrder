"""
评分器基类

定义评分器接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


class BaseGrader(ABC):
    """评分器基类"""

    grader_type: GraderType = GraderType.INTENT_ACCURACY

    def __init__(self, config: Optional[GraderConfig] = None):
        """
        初始化评分器

        Args:
            config: 评分器配置
        """
        self.config = config or GraderConfig(type=self.grader_type)

    @abstractmethod
    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        执行评估

        Args:
            predictions: 模型预测结果列表
            test_cases: 测试用例列表
            transcript: 对话记录 (可选)
            **kwargs: 额外参数

        Returns:
            GraderResult: 评分结果
        """
        pass

    def _compute_score(self, correct: int, total: int) -> float:
        """计算得分 (0.0 - 1.0)"""
        return correct / total if total > 0 else 0.0

    def _check_threshold(self, score: float, threshold: float) -> bool:
        """检查是否达到阈值"""
        return score >= threshold
