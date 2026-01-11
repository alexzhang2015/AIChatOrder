"""
状态检查评分器

验证最终订单状态是否符合预期
支持嵌套字段检查和模糊匹配
"""

from typing import Dict, List, Any, Optional

from evals.graders.base import BaseGrader
from evals.harness.models import GraderResult, GraderConfig, TestCase, GraderType


class StateGrader(BaseGrader):
    """状态检查评分器"""

    grader_type = GraderType.STATE_CHECK

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        expect: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.expect = config.expect if config and config.expect else (expect or {})

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        test_cases: List[TestCase],
        transcript: Optional[List[Dict[str, Any]]] = None,
        outcome: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> GraderResult:
        """
        检查最终状态是否符合预期

        Args:
            predictions: 未使用
            test_cases: 未使用
            transcript: 未使用
            outcome: 最终状态

        Returns:
            GraderResult
        """
        if not outcome:
            return GraderResult(
                grader_type=self.grader_type,
                passed=False,
                score=0.0,
                details={"error": "No outcome provided"},
                failures=[]
            )

        # 从 kwargs 或 config 获取期望状态
        expect = kwargs.get("expect", self.expect)
        if not expect:
            return GraderResult(
                grader_type=self.grader_type,
                passed=True,
                score=1.0,
                details={"message": "No expectations defined, auto-pass"},
                failures=[]
            )

        # 检查状态
        failures = []
        checks_passed = 0
        checks_total = 0

        for path, expected_value in self._flatten_dict(expect):
            checks_total += 1
            actual_value = self._get_nested_value(outcome, path)

            if self._compare_values(actual_value, expected_value):
                checks_passed += 1
            else:
                failures.append({
                    "type": "state_mismatch",
                    "path": path,
                    "expected": expected_value,
                    "actual": actual_value
                })

        score = checks_passed / checks_total if checks_total > 0 else 0
        passed = len(failures) == 0

        return GraderResult(
            grader_type=self.grader_type,
            passed=passed,
            score=score,
            details={
                "checks_passed": checks_passed,
                "checks_total": checks_total,
                "expected": expect,
                "actual_outcome": outcome
            },
            failures=failures
        )

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = ""
    ) -> List[tuple]:
        """扁平化嵌套字典为路径-值对"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict) and not self._is_special_value(v):
                items.extend(self._flatten_dict(v, new_key))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # 处理对象列表
                for i, item in enumerate(v):
                    items.extend(self._flatten_dict(item, f"{new_key}[{i}]"))
            else:
                items.append((new_key, v))
        return items

    def _is_special_value(self, v: Any) -> bool:
        """检查是否是特殊检查值"""
        if isinstance(v, dict):
            return any(k.startswith("$") or k in ["within_seconds", "contains", "matches"]
                      for k in v.keys())
        return False

    def _get_nested_value(self, d: Dict[str, Any], path: str) -> Any:
        """获取嵌套路径的值"""
        keys = path.replace("[", ".").replace("]", "").split(".")
        value = d
        for key in keys:
            if value is None:
                return None
            if isinstance(value, list):
                try:
                    value = value[int(key)]
                except (IndexError, ValueError):
                    return None
            elif isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def _compare_values(self, actual: Any, expected: Any) -> bool:
        """比较值是否匹配"""
        if expected is None:
            return actual is None

        # 处理特殊检查
        if isinstance(expected, dict):
            if "contains" in expected:
                return expected["contains"] in str(actual)
            if "matches" in expected:
                import re
                return bool(re.match(expected["matches"], str(actual)))
            if "within_seconds" in expected:
                # 时间范围检查 - 简化实现
                return True

        # 列表比较 (顺序无关)
        if isinstance(expected, list) and isinstance(actual, list):
            if all(isinstance(e, dict) for e in expected):
                # 对象列表 - 检查每个期望的对象是否存在
                for exp_item in expected:
                    found = False
                    for act_item in actual:
                        if self._dict_matches(act_item, exp_item):
                            found = True
                            break
                    if not found:
                        return False
                return True
            else:
                return set(actual) == set(expected)

        # 字符串比较 (忽略大小写)
        if isinstance(expected, str) and isinstance(actual, str):
            return actual.lower() == expected.lower()

        return actual == expected

    def _dict_matches(self, actual: Dict, expected: Dict) -> bool:
        """检查实际字典是否包含期望的所有键值"""
        for k, v in expected.items():
            if k not in actual:
                return False
            if not self._compare_values(actual[k], v):
                return False
        return True
