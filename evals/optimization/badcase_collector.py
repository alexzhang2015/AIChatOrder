"""
Bad Case 收集器

从评估结果、生产日志、用户反馈中收集 Bad Case
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum


class BadCaseCategory(str, Enum):
    """Bad Case 类别"""
    INTENT_CONFUSION = "intent_confusion"      # 意图混淆
    SLOT_EXTRACTION = "slot_extraction"        # 槽位提取错误
    FUZZY_EXPRESSION = "fuzzy_expression"      # 模糊表达未识别
    CONSTRAINT_VIOLATION = "constraint_violation"  # 约束违反
    DIALOGUE_CONTEXT = "dialogue_context"      # 对话上下文问题
    MULTI_TURN = "multi_turn"                  # 多轮对话问题
    BUSINESS_RULE = "business_rule"            # 业务规则问题
    RESPONSE_QUALITY = "response_quality"      # 回复质量问题
    OTHER = "other"


class Severity(str, Enum):
    """严重程度"""
    CRITICAL = "critical"  # 影响核心功能
    MAJOR = "major"        # 影响用户体验
    MINOR = "minor"        # 边缘场景


class FixStatus(str, Enum):
    """修复状态"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    IN_PROGRESS = "in_progress"
    FIXED = "fixed"
    VERIFIED = "verified"
    WONT_FIX = "wont_fix"


class FixStrategy(str, Enum):
    """修复策略"""
    PROMPT_TUNING = "prompt_tuning"      # Prompt 调优
    ADD_FEW_SHOT = "add_few_shot"        # 增加 Few-shot 示例
    ADD_RULE = "add_rule"                # 增加规则
    ADD_FUZZY_EXPR = "add_fuzzy_expr"    # 增加模糊表达
    FIX_CONSTRAINT = "fix_constraint"    # 修复约束配置
    CODE_FIX = "code_fix"                # 代码修复
    DATA_UPDATE = "data_update"          # 数据更新
    MANUAL_REVIEW = "manual_review"      # 需人工评审


@dataclass
class BadCase:
    """Bad Case 记录"""
    id: str
    timestamp: str
    source: str  # eval, production, user_feedback, manual

    # 输入信息
    user_input: str
    conversation_history: List[Dict] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # 实际输出
    actual_intent: str = ""
    actual_confidence: float = 0.0
    actual_slots: Dict[str, Any] = field(default_factory=dict)
    actual_response: str = ""

    # 期望输出
    expected_intent: str = ""
    expected_slots: Dict[str, Any] = field(default_factory=dict)
    expected_response: str = ""

    # 分析信息
    category: str = BadCaseCategory.OTHER.value
    root_cause: str = ""
    severity: str = Severity.MINOR.value
    tags: List[str] = field(default_factory=list)

    # 修复信息
    fix_strategy: str = ""
    fix_status: str = FixStatus.PENDING.value
    fix_description: str = ""
    fixed_in_version: str = ""
    fixed_by: str = ""
    fixed_at: str = ""

    # 验证信息
    verified_at: str = ""
    verification_result: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BadCase":
        return cls(**data)


class BadCaseCollector:
    """Bad Case 收集器"""

    def __init__(self, storage_dir: str = "evals/optimization/badcases"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_eval_result(
        self,
        eval_result: Any,
        auto_categorize: bool = True
    ) -> List[BadCase]:
        """
        从评估结果中收集 Bad Case

        Args:
            eval_result: EvalResult 对象
            auto_categorize: 是否自动分类

        Returns:
            收集到的 Bad Case 列表
        """
        badcases = []

        for trial in eval_result.trials:
            for grader_type, grader_result in trial.grader_results.items():
                # 处理 GraderResult 对象或字典
                if hasattr(grader_result, 'failures'):
                    failures = grader_result.failures
                elif isinstance(grader_result, dict):
                    failures = grader_result.get('failures', [])
                else:
                    continue

                for failure in failures:
                    badcase = self._create_badcase_from_failure(
                        failure=failure,
                        grader_type=grader_type,
                        task_id=eval_result.task_id,
                        auto_categorize=auto_categorize
                    )
                    badcases.append(badcase)

        return badcases

    def collect_from_production_log(
        self,
        log_entry: Dict[str, Any],
        user_feedback: Optional[str] = None
    ) -> BadCase:
        """
        从生产日志中收集 Bad Case

        Args:
            log_entry: 日志条目
            user_feedback: 用户反馈

        Returns:
            BadCase 对象
        """
        return BadCase(
            id=self._generate_id(log_entry.get("user_input", "")),
            timestamp=datetime.now().isoformat(),
            source="production",
            user_input=log_entry.get("user_input", ""),
            conversation_history=log_entry.get("history", []),
            context=log_entry.get("context", {}),
            actual_intent=log_entry.get("intent", ""),
            actual_confidence=log_entry.get("confidence", 0.0),
            actual_slots=log_entry.get("slots", {}),
            actual_response=log_entry.get("response", ""),
            root_cause=user_feedback or "",
            severity=Severity.MAJOR.value,
        )

    def collect_manual(
        self,
        user_input: str,
        expected_intent: str,
        expected_slots: Optional[Dict] = None,
        actual_intent: str = "",
        actual_slots: Optional[Dict] = None,
        category: str = "",
        root_cause: str = "",
        severity: str = Severity.MINOR.value
    ) -> BadCase:
        """
        手动添加 Bad Case

        Args:
            user_input: 用户输入
            expected_intent: 期望意图
            expected_slots: 期望槽位
            actual_intent: 实际意图
            actual_slots: 实际槽位
            category: 类别
            root_cause: 根因
            severity: 严重程度

        Returns:
            BadCase 对象
        """
        return BadCase(
            id=self._generate_id(user_input),
            timestamp=datetime.now().isoformat(),
            source="manual",
            user_input=user_input,
            expected_intent=expected_intent,
            expected_slots=expected_slots or {},
            actual_intent=actual_intent,
            actual_slots=actual_slots or {},
            category=category or self._auto_categorize_by_diff(
                expected_intent, actual_intent, expected_slots, actual_slots
            ),
            root_cause=root_cause,
            severity=severity,
        )

    def add(self, badcase: BadCase, filename: Optional[str] = None):
        """添加单个 Bad Case"""
        self.save([badcase], filename)

    def save(self, badcases: List[BadCase], filename: Optional[str] = None):
        """保存 Bad Case 列表"""
        if not filename:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"badcases_{date_str}.json"

        filepath = self.storage_dir / filename

        # 加载已有数据
        existing = []
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # 合并去重
        existing_ids = {bc["id"] for bc in existing}
        for bc in badcases:
            if bc.id not in existing_ids:
                existing.append(bc.to_dict())

        # 保存
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        return filepath

    def load(self, filename: Optional[str] = None) -> List[BadCase]:
        """加载 Bad Case 列表"""
        if filename:
            filepath = self.storage_dir / filename
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [BadCase.from_dict(d) for d in data]
            return []

        # 加载所有文件
        all_badcases = []
        for filepath in self.storage_dir.glob("badcases_*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_badcases.extend([BadCase.from_dict(d) for d in data])

        return all_badcases

    def load_pending(self) -> List[BadCase]:
        """加载待处理的 Bad Case"""
        all_cases = self.load()
        return [bc for bc in all_cases if bc.fix_status == FixStatus.PENDING.value]

    def update_status(
        self,
        badcase_id: str,
        status: str,
        fix_description: str = "",
        fixed_in_version: str = ""
    ):
        """更新 Bad Case 状态"""
        # 支持字符串或枚举
        status_value = status.value if isinstance(status, FixStatus) else status

        all_cases = self.load()
        for bc in all_cases:
            if bc.id == badcase_id:
                bc.fix_status = status_value
                bc.fix_description = fix_description
                bc.fixed_in_version = fixed_in_version
                if status_value == FixStatus.FIXED.value:
                    bc.fixed_at = datetime.now().isoformat()
                break

        # 按日期分组保存
        by_date = {}
        for bc in all_cases:
            date_str = bc.timestamp[:10].replace("-", "")
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(bc)

        for date_str, cases in by_date.items():
            filename = f"badcases_{date_str}.json"
            filepath = self.storage_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([bc.to_dict() for bc in cases], f, ensure_ascii=False, indent=2)

    def _create_badcase_from_failure(
        self,
        failure: Dict,
        grader_type: str,
        task_id: str,
        auto_categorize: bool
    ) -> BadCase:
        """从失败记录创建 Bad Case"""
        user_input = str(failure.get("input", ""))

        badcase = BadCase(
            id=self._generate_id(f"{task_id}_{user_input}"),
            timestamp=datetime.now().isoformat(),
            source="eval",
            user_input=user_input,
            actual_intent=failure.get("predicted_intent", failure.get("predicted", "")),
            actual_slots=failure.get("predicted_slots", {}),
            expected_intent=failure.get("expected_intent", failure.get("expected", "")),
            expected_slots=failure.get("expected_slots", {}),
            tags=[task_id, grader_type],
        )

        if auto_categorize:
            badcase.category = self._auto_categorize(failure, grader_type)
            badcase.severity = self._auto_severity(failure, grader_type)

        return badcase

    def _auto_categorize(self, failure: Dict, grader_type: str) -> str:
        """自动分类"""
        failure_type = failure.get("type", "")

        if "intent" in grader_type.lower() or failure_type == "wrong_intent":
            return BadCaseCategory.INTENT_CONFUSION.value
        elif "slot" in grader_type.lower():
            return BadCaseCategory.SLOT_EXTRACTION.value
        elif "fuzzy" in grader_type.lower():
            return BadCaseCategory.FUZZY_EXPRESSION.value
        elif "constraint" in grader_type.lower():
            return BadCaseCategory.CONSTRAINT_VIOLATION.value
        else:
            return BadCaseCategory.OTHER.value

    def _auto_categorize_by_diff(
        self,
        expected_intent: str,
        actual_intent: str,
        expected_slots: Optional[Dict],
        actual_slots: Optional[Dict]
    ) -> str:
        """根据差异自动分类"""
        if expected_intent != actual_intent:
            return BadCaseCategory.INTENT_CONFUSION.value
        if expected_slots != actual_slots:
            return BadCaseCategory.SLOT_EXTRACTION.value
        return BadCaseCategory.OTHER.value

    def _auto_severity(self, failure: Dict, grader_type: str) -> str:
        """自动评估严重程度"""
        # 意图完全错误视为严重
        if failure.get("type") == "wrong_intent":
            expected = failure.get("expected_intent", "")
            actual = failure.get("predicted_intent", "")
            # 核心意图错误更严重
            critical_intents = ["ORDER_NEW", "ORDER_CANCEL", "PAYMENT"]
            if expected in critical_intents or actual in critical_intents:
                return Severity.CRITICAL.value
            return Severity.MAJOR.value

        return Severity.MINOR.value

    def _generate_id(self, content: str) -> str:
        """生成唯一 ID"""
        hash_input = f"{content}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
