"""
LLM Rubric 评分器

使用 LLM 根据评分标准 (Rubric) 评估对话质量
支持多维度评分和详细理由
"""

import json
import logging
from typing import Dict, List, Any, Optional

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType

logger = logging.getLogger(__name__)

# 默认评分 Rubric
DEFAULT_RUBRIC = """
评估对话质量（各项 1-5 分）：
1. 意图理解：是否准确捕捉用户需求
2. 引导自然：推荐是否自然、不强推
3. 确认清晰：下单前是否清晰复述
4. 响应效率：是否高效完成，避免冗余轮次
"""


class LLMRubricGrader(BaseGrader):
    """LLM Rubric 评分器"""

    grader_type = GraderType.LLM_RUBRIC

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        model: str = "gpt-4o-mini",
        rubric: Optional[str] = None,
        min_scores: Optional[Dict[str, float]] = None
    ):
        super().__init__(config)
        self.model = config.model if config and config.model else model
        self.rubric = config.rubric if config and config.rubric else (rubric or DEFAULT_RUBRIC)
        self.min_scores = config.min_scores if config and config.min_scores else (min_scores or {})
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                from config import get_openai_settings
                settings = get_openai_settings()
                self._client = OpenAI(
                    api_key=settings.api_key,
                    base_url=settings.base_url or None
                )
            except Exception as e:
                logger.warning(f"无法初始化 OpenAI 客户端: {e}")
                self._client = None
        return self._client

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> GraderResult:
        """
        使用 LLM 评估对话质量

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
                passed=False,
                score=0.0,
                details={"error": "No transcript provided"},
                failures=[]
            )

        if not self.client:
            return GraderResult(
                grader_type=self.grader_type,
                passed=False,
                score=0.0,
                details={"error": "OpenAI client not available"},
                failures=[]
            )

        # 构建评估 prompt
        prompt = self._build_prompt(transcript, kwargs.get("context"))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content)
            scores = result.get("scores", {})
            reasoning = result.get("reasoning", "")

            # 计算平均分
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            normalized_score = avg_score / 5.0  # 归一化到 0-1

            # 检查是否通过
            passed = True
            failures = []
            for criterion, min_score in self.min_scores.items():
                actual_score = scores.get(criterion, 0)
                if actual_score < min_score:
                    passed = False
                    failures.append({
                        "type": "below_threshold",
                        "criterion": criterion,
                        "expected_min": min_score,
                        "actual": actual_score
                    })

            return GraderResult(
                grader_type=self.grader_type,
                passed=passed,
                score=normalized_score,
                details={
                    "scores": scores,
                    "reasoning": reasoning,
                    "average_score": round(avg_score, 2),
                    "min_scores_required": self.min_scores,
                    "model_used": self.model
                },
                failures=failures
            )

        except Exception as e:
            logger.error(f"LLM 评分失败: {e}")
            return GraderResult(
                grader_type=self.grader_type,
                passed=False,
                score=0.0,
                details={"error": str(e)},
                failures=[{"type": "llm_error", "message": str(e)}]
            )

    def _build_prompt(
        self,
        transcript: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """构建评估 prompt"""
        # 格式化对话记录
        transcript_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in transcript
        ])

        context_text = ""
        if context:
            context_text = f"\n## 额外上下文\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        return f"""请根据以下评分标准评估对话质量。

## 对话记录
{transcript_text}

## 评分标准
{self.rubric}
{context_text}

请返回 JSON 格式的评分结果：
{{
  "scores": {{
    "意图理解": 1-5分,
    "引导自然": 1-5分,
    "确认清晰": 1-5分,
    "响应效率": 1-5分
  }},
  "reasoning": "详细评分理由，说明每项得分的依据"
}}

注意：
1. 每项评分必须是 1-5 的整数
2. 评分理由要具体，引用对话中的内容作为依据
3. 评分要客观公正，既指出优点也指出不足
"""
