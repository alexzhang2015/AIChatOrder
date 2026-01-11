"""
Bad Case 分析器

分析 Bad Case 模式，识别共性问题，推荐修复策略
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from evals.optimization.badcase_collector import (
    BadCase, BadCaseCategory, Severity, FixStrategy, FixStatus
)


@dataclass
class Pattern:
    """问题模式"""
    pattern_type: str
    description: str
    count: int
    examples: List[str]
    suggested_fix: str
    fix_strategy: FixStrategy
    priority: int  # 1-5, 1最高


@dataclass
class FixRecommendation:
    """修复建议"""
    strategy: FixStrategy
    description: str
    affected_cases: List[str]  # Bad Case IDs
    effort: str  # low, medium, high
    impact: str  # low, medium, high
    action_items: List[Dict[str, Any]]
    priority_score: float


@dataclass
class AnalysisReport:
    """分析报告"""
    total_cases: int
    by_category: Dict[str, int]
    by_severity: Dict[str, int]
    by_status: Dict[str, int]
    top_patterns: List[Pattern]
    recommendations: List[FixRecommendation]
    priority_queue: List[BadCase]
    insights: List[str]


class BadCaseAnalyzer:
    """Bad Case 分析器"""

    def __init__(self):
        # 意图混淆模式配置
        self.confusion_fixes = {
            ("RECOMMEND", "ORDER_NEW"): {
                "strategy": FixStrategy.ADD_FEW_SHOT,
                "description": "增加'浏览而非下单'的 few-shot 示例",
            },
            ("ORDER_CANCEL", "CHITCHAT"): {
                "strategy": FixStrategy.ADD_RULE,
                "description": "添加隐式取消规则（算了、不要了）",
            },
            ("ORDER_MODIFY", "ORDER_NEW"): {
                "strategy": FixStrategy.PROMPT_TUNING,
                "description": "优化 Prompt 中新增 vs 修改的边界说明",
            },
        }

    def analyze(self, badcases: List[BadCase]) -> AnalysisReport:
        """
        全面分析 Bad Case

        Args:
            badcases: Bad Case 列表

        Returns:
            分析报告
        """
        if not badcases:
            return AnalysisReport(
                total_cases=0,
                by_category={},
                by_severity={},
                by_status={},
                top_patterns=[],
                recommendations=[],
                priority_queue=[],
                insights=["暂无 Bad Case"]
            )

        # 基础统计
        by_category = dict(Counter(bc.category for bc in badcases))
        by_severity = dict(Counter(bc.severity for bc in badcases))
        by_status = dict(Counter(bc.fix_status for bc in badcases))

        # 模式识别
        patterns = self._find_patterns(badcases)

        # 生成修复建议
        recommendations = self._generate_recommendations(badcases, patterns)

        # 优先级排序
        priority_queue = self._prioritize(badcases)

        # 生成洞察
        insights = self._generate_insights(badcases, patterns)

        return AnalysisReport(
            total_cases=len(badcases),
            by_category=by_category,
            by_severity=by_severity,
            by_status=by_status,
            top_patterns=patterns,
            recommendations=recommendations,
            priority_queue=priority_queue,
            insights=insights
        )

    def _find_patterns(self, badcases: List[BadCase]) -> List[Pattern]:
        """识别共性模式"""
        patterns = []

        # 1. 意图混淆模式
        intent_patterns = self._find_intent_confusion_patterns(badcases)
        patterns.extend(intent_patterns)

        # 2. 槽位提取模式
        slot_patterns = self._find_slot_patterns(badcases)
        patterns.extend(slot_patterns)

        # 3. 表达模式
        expression_patterns = self._find_expression_patterns(badcases)
        patterns.extend(expression_patterns)

        # 按优先级排序
        patterns.sort(key=lambda p: (p.priority, -p.count))

        return patterns[:10]  # 返回前10个模式

    def _find_intent_confusion_patterns(self, badcases: List[BadCase]) -> List[Pattern]:
        """识别意图混淆模式"""
        patterns = []

        # 按 (expected, actual) 分组
        confusion_groups = defaultdict(list)
        for bc in badcases:
            if bc.category == BadCaseCategory.INTENT_CONFUSION.value:
                key = (bc.expected_intent, bc.actual_intent)
                confusion_groups[key].append(bc)

        for (expected, actual), cases in confusion_groups.items():
            if len(cases) >= 2:  # 至少2个相似 case 才算模式
                fix_info = self.confusion_fixes.get(
                    (expected, actual),
                    {"strategy": FixStrategy.PROMPT_TUNING, "description": "需要分析"}
                )

                patterns.append(Pattern(
                    pattern_type="intent_confusion",
                    description=f"意图混淆: {expected} → {actual}",
                    count=len(cases),
                    examples=[bc.user_input for bc in cases[:5]],
                    suggested_fix=fix_info["description"],
                    fix_strategy=fix_info["strategy"],
                    priority=1 if any(bc.severity == Severity.CRITICAL.value for bc in cases) else 2
                ))

        return patterns

    def _find_slot_patterns(self, badcases: List[BadCase]) -> List[Pattern]:
        """识别槽位提取模式"""
        patterns = []

        # 按缺失的槽位分组
        missing_slots = defaultdict(list)
        wrong_slots = defaultdict(list)

        for bc in badcases:
            if bc.category == BadCaseCategory.SLOT_EXTRACTION.value:
                expected = bc.expected_slots or {}
                actual = bc.actual_slots or {}

                for slot, value in expected.items():
                    if slot not in actual:
                        missing_slots[slot].append(bc)
                    elif actual.get(slot) != value:
                        wrong_slots[slot].append(bc)

        # 生成缺失槽位模式
        for slot, cases in missing_slots.items():
            if len(cases) >= 2:
                patterns.append(Pattern(
                    pattern_type="missing_slot",
                    description=f"槽位未提取: {slot}",
                    count=len(cases),
                    examples=[bc.user_input for bc in cases[:5]],
                    suggested_fix=f"检查 {slot} 的提取规则和别名配置",
                    fix_strategy=FixStrategy.DATA_UPDATE,
                    priority=2
                ))

        # 生成错误槽位模式
        for slot, cases in wrong_slots.items():
            if len(cases) >= 2:
                patterns.append(Pattern(
                    pattern_type="wrong_slot_value",
                    description=f"槽位值错误: {slot}",
                    count=len(cases),
                    examples=[f"{bc.user_input} → {bc.actual_slots.get(slot)}" for bc in cases[:5]],
                    suggested_fix=f"检查 {slot} 的标准化和别名映射",
                    fix_strategy=FixStrategy.DATA_UPDATE,
                    priority=2
                ))

        return patterns

    def _find_expression_patterns(self, badcases: List[BadCase]) -> List[Pattern]:
        """识别表达模式"""
        patterns = []

        # 提取常见的失败表达
        fuzzy_cases = [bc for bc in badcases
                       if bc.category == BadCaseCategory.FUZZY_EXPRESSION.value]

        if len(fuzzy_cases) >= 2:
            # 尝试找出共同的表达模式
            expressions = [bc.user_input for bc in fuzzy_cases]
            common_words = self._find_common_words(expressions)

            if common_words:
                patterns.append(Pattern(
                    pattern_type="unrecognized_expression",
                    description=f"未识别的模糊表达，常见词: {', '.join(common_words)}",
                    count=len(fuzzy_cases),
                    examples=expressions[:5],
                    suggested_fix="添加模糊表达配置到 slots_v2.yaml",
                    fix_strategy=FixStrategy.ADD_FUZZY_EXPR,
                    priority=2
                ))

        return patterns

    def _find_common_words(self, texts: List[str]) -> List[str]:
        """找出文本中的共同词"""
        if not texts:
            return []

        # 分词统计
        word_counts = Counter()
        for text in texts:
            # 简单分词
            words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text)
            word_counts.update(words)

        # 过滤停用词，返回出现频率高的词
        stopwords = {"的", "了", "吗", "呢", "啊", "我", "要", "想", "来", "个", "杯"}
        common = [(word, count) for word, count in word_counts.most_common(10)
                  if word not in stopwords and count >= 2]

        return [word for word, _ in common[:5]]

    def _generate_recommendations(
        self,
        badcases: List[BadCase],
        patterns: List[Pattern]
    ) -> List[FixRecommendation]:
        """生成修复建议"""
        recommendations = []

        # 按修复策略分组
        by_strategy = defaultdict(list)
        for pattern in patterns:
            by_strategy[pattern.fix_strategy].append(pattern)

        for strategy, strategy_patterns in by_strategy.items():
            # 收集受影响的 case
            affected_ids = []
            action_items = []

            for pattern in strategy_patterns:
                # 找到匹配该模式的 bad case
                for bc in badcases:
                    if bc.id not in affected_ids:
                        if self._matches_pattern(bc, pattern):
                            affected_ids.append(bc.id)

                # 生成具体的修复动作
                action_items.append({
                    "pattern": pattern.description,
                    "action": pattern.suggested_fix,
                    "examples": pattern.examples[:3]
                })

            # 评估工作量和影响
            effort = self._estimate_effort(strategy, len(action_items))
            impact = self._estimate_impact(affected_ids, badcases)

            recommendations.append(FixRecommendation(
                strategy=strategy,
                description=self._get_strategy_description(strategy),
                affected_cases=affected_ids,
                effort=effort,
                impact=impact,
                action_items=action_items,
                priority_score=self._compute_priority_score(
                    len(affected_ids), effort, impact
                )
            ))

        # 按优先级排序
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations

    def _matches_pattern(self, badcase: BadCase, pattern: Pattern) -> bool:
        """检查 bad case 是否匹配模式"""
        if pattern.pattern_type == "intent_confusion":
            # 从描述中提取期望和实际意图
            match = re.search(r"(\w+) → (\w+)", pattern.description)
            if match:
                expected, actual = match.groups()
                return (badcase.expected_intent == expected and
                        badcase.actual_intent == actual)

        elif pattern.pattern_type in ["missing_slot", "wrong_slot_value"]:
            return badcase.category == BadCaseCategory.SLOT_EXTRACTION.value

        elif pattern.pattern_type == "unrecognized_expression":
            return badcase.category == BadCaseCategory.FUZZY_EXPRESSION.value

        return False

    def _estimate_effort(self, strategy: FixStrategy, num_items: int) -> str:
        """估算工作量"""
        effort_map = {
            FixStrategy.ADD_FEW_SHOT: "low",
            FixStrategy.ADD_RULE: "low",
            FixStrategy.ADD_FUZZY_EXPR: "low",
            FixStrategy.DATA_UPDATE: "low",
            FixStrategy.PROMPT_TUNING: "medium",
            FixStrategy.FIX_CONSTRAINT: "medium",
            FixStrategy.CODE_FIX: "high",
            FixStrategy.MANUAL_REVIEW: "high",
        }
        base_effort = effort_map.get(strategy, "medium")

        # 数量多会增加工作量
        if num_items > 5:
            return "high" if base_effort == "medium" else base_effort

        return base_effort

    def _estimate_impact(self, affected_ids: List[str], badcases: List[BadCase]) -> str:
        """估算影响"""
        if not affected_ids:
            return "low"

        # 计算受影响 case 中的严重程度分布
        affected_cases = [bc for bc in badcases if bc.id in affected_ids]
        critical_count = sum(1 for bc in affected_cases
                            if bc.severity == Severity.CRITICAL.value)
        major_count = sum(1 for bc in affected_cases
                         if bc.severity == Severity.MAJOR.value)

        if critical_count > 0:
            return "high"
        elif major_count >= 3:
            return "high"
        elif major_count >= 1:
            return "medium"
        else:
            return "low"

    def _compute_priority_score(
        self,
        affected_count: int,
        effort: str,
        impact: str
    ) -> float:
        """计算优先级分数"""
        # 影响越大、工作量越小、受影响数量越多，优先级越高
        impact_score = {"high": 3, "medium": 2, "low": 1}.get(impact, 1)
        effort_score = {"low": 3, "medium": 2, "high": 1}.get(effort, 1)

        return (impact_score * 2 + effort_score + min(affected_count / 10, 1)) / 4

    def _get_strategy_description(self, strategy: FixStrategy) -> str:
        """获取策略描述"""
        descriptions = {
            FixStrategy.PROMPT_TUNING: "优化 Prompt 模板，调整指令和边界说明",
            FixStrategy.ADD_FEW_SHOT: "增加 Few-shot 示例，覆盖更多场景",
            FixStrategy.ADD_RULE: "添加前置规则，减少 LLM 调用",
            FixStrategy.ADD_FUZZY_EXPR: "添加模糊表达配置，支持更多口语化表达",
            FixStrategy.FIX_CONSTRAINT: "修复约束配置，完善业务规则",
            FixStrategy.CODE_FIX: "修改代码逻辑，修复 bug",
            FixStrategy.DATA_UPDATE: "更新数据配置，如别名、标准化映射",
            FixStrategy.MANUAL_REVIEW: "需要人工评审，确定修复方案",
        }
        return descriptions.get(strategy, "")

    def _prioritize(self, badcases: List[BadCase]) -> List[BadCase]:
        """按优先级排序 Bad Case"""
        # 只返回待处理的 case
        pending = [bc for bc in badcases if bc.fix_status == FixStatus.PENDING.value]

        # 排序规则: 严重程度 > 类别重要性 > 时间
        severity_order = {
            Severity.CRITICAL.value: 0,
            Severity.MAJOR.value: 1,
            Severity.MINOR.value: 2
        }
        category_order = {
            BadCaseCategory.INTENT_CONFUSION.value: 0,
            BadCaseCategory.SLOT_EXTRACTION.value: 1,
            BadCaseCategory.CONSTRAINT_VIOLATION.value: 2,
            BadCaseCategory.FUZZY_EXPRESSION.value: 3,
            BadCaseCategory.DIALOGUE_CONTEXT.value: 4,
            BadCaseCategory.OTHER.value: 5,
        }

        pending.sort(key=lambda bc: (
            severity_order.get(bc.severity, 2),
            category_order.get(bc.category, 5),
            bc.timestamp
        ))

        return pending[:20]  # 返回前20个

    def _generate_insights(
        self,
        badcases: List[BadCase],
        patterns: List[Pattern]
    ) -> List[str]:
        """生成洞察"""
        insights = []

        total = len(badcases)
        if total == 0:
            return ["暂无 Bad Case，系统表现良好"]

        # 类别分布洞察
        by_category = Counter(bc.category for bc in badcases)
        top_category = by_category.most_common(1)[0]
        insights.append(
            f"最常见的问题类型是「{top_category[0]}」，占 {top_category[1]/total:.0%}"
        )

        # 严重程度洞察
        critical_count = sum(1 for bc in badcases
                            if bc.severity == Severity.CRITICAL.value)
        if critical_count > 0:
            insights.append(f"⚠️ 有 {critical_count} 个严重问题需要优先处理")

        # 模式洞察
        if patterns:
            top_pattern = patterns[0]
            insights.append(
                f"发现重要模式: {top_pattern.description}，"
                f"影响 {top_pattern.count} 个 case，"
                f"建议: {top_pattern.suggested_fix}"
            )

        # 修复进度洞察
        fixed_count = sum(1 for bc in badcases
                         if bc.fix_status in [FixStatus.FIXED.value, FixStatus.VERIFIED.value])
        if fixed_count > 0:
            insights.append(f"已修复 {fixed_count}/{total} ({fixed_count/total:.0%})")

        return insights
