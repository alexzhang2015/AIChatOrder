"""
对话记录评分器

评估对话效率和质量指标:
- 对话轮数
- 完成时间
- 工具调用次数
"""

from typing import Dict, List, Any, Optional

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


class TranscriptGrader(BaseGrader):
    """对话记录评分器"""

    grader_type = GraderType.TRANSCRIPT

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        max_turns: int = 10,
        max_tool_calls: Optional[int] = None,
        max_duration_seconds: Optional[float] = None
    ):
        super().__init__(config)
        self.max_turns = config.max_turns if config and config.max_turns else max_turns
        self.max_tool_calls = max_tool_calls
        self.max_duration_seconds = max_duration_seconds

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        评估对话效率

        Args:
            predictions: 未使用
            test_cases: 未使用
            transcript: 对话记录

        Returns:
            GraderResult
        """
        if not transcript:
            return GraderResult(
                grader_type=self.grader_type,
                passed=True,
                score=1.0,
                details={"message": "No transcript to evaluate"},
                failures=[]
            )

        # 统计对话轮数
        user_turns = sum(1 for msg in transcript if msg.get("role") == "user")
        assistant_turns = sum(1 for msg in transcript if msg.get("role") == "assistant")
        total_turns = len(transcript)

        # 统计工具调用
        tool_calls = sum(
            len(msg.get("tool_calls", []))
            for msg in transcript
            if msg.get("tool_calls")
        )

        # 检查是否通过
        failures = []
        passed = True

        if user_turns > self.max_turns:
            passed = False
            failures.append({
                "type": "too_many_turns",
                "max_allowed": self.max_turns,
                "actual": user_turns
            })

        if self.max_tool_calls and tool_calls > self.max_tool_calls:
            passed = False
            failures.append({
                "type": "too_many_tool_calls",
                "max_allowed": self.max_tool_calls,
                "actual": tool_calls
            })

        # 计算效率分数 (轮数越少越好)
        efficiency_score = max(0, 1 - (user_turns - 1) / self.max_turns) if self.max_turns > 1 else 1.0

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=efficiency_score,
            details={
                "user_turns": user_turns,
                "assistant_turns": assistant_turns,
                "total_messages": total_turns,
                "tool_calls": tool_calls,
                "max_turns_allowed": self.max_turns,
                "efficiency_score": round(efficiency_score, 4)
            },
            failures=failures
        )
