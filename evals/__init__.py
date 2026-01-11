"""
AI 点单系统评估框架 (Evals)

基于 Anthropic《Demystifying Evals for AI Agents》核心理念设计
用于评估意图分类、槽位提取、对话质量等能力

目录结构:
- harness/: 评估执行框架
- graders/: 评分器实现
- tasks/: 任务定义 (YAML)
- fixtures/: 测试数据
- results/: 评估结果与 Transcripts
"""

# 延迟导入以避免循环依赖
def get_runner():
    from evals.harness.runner import EvalRunner
    return EvalRunner

def get_result_classes():
    from evals.harness.models import EvalResult, Trial, TaskConfig, GraderConfig
    return EvalResult, Trial, TaskConfig, GraderConfig

__all__ = [
    "get_runner",
    "get_result_classes",
]
