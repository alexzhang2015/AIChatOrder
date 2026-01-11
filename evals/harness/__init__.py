"""评估执行框架"""

# 导入基础模型 (无外部依赖)
from evals.harness.models import (
    TaskConfig, GraderConfig, TestCase, GraderType,
    Trial, EvalResult, EvalSuiteResult, GraderResult, TaskCategory
)

# 延迟导入有外部依赖的模块
def get_runner():
    from evals.harness.runner import EvalRunner
    return EvalRunner

def get_environment():
    from evals.harness.environment import EvalEnvironment
    return EvalEnvironment

def get_reporter():
    from evals.harness.reporter import EvalReporter
    return EvalReporter

__all__ = [
    # 基础模型
    "TaskConfig",
    "GraderConfig",
    "TestCase",
    "GraderType",
    "Trial",
    "EvalResult",
    "EvalSuiteResult",
    "GraderResult",
    "TaskCategory",
    # 延迟导入函数
    "get_runner",
    "get_environment",
    "get_reporter",
]
