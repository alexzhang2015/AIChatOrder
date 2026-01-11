"""
AI 点单 Agent 持续优化模块

提供:
- Bad Case 收集与管理
- 根因分析与模式识别
- 自动化修复建议
- 优化效果追踪
- 数据增强
- Embedding 知识库构建
- 生产数据收集

使用方法:
    # CLI 工具
    python -m evals.optimization.cli --help

    # 编程接口
    from evals.optimization import BadCaseCollector, DataAugmenter
"""

from evals.optimization.badcase_collector import BadCase, BadCaseCollector
from evals.optimization.badcase_analyzer import BadCaseAnalyzer
from evals.optimization.fix_tracker import FixTracker, OptimizationRound
from evals.optimization.data_augmenter import DataAugmenter, ProductCorpus, AugmentedExample
from evals.optimization.embedding_builder import EmbeddingKnowledgeBase, EmbeddingDocument
from evals.optimization.production_collector import ProductionDataCollector, ProductionSample

__all__ = [
    # Bad Case 管理
    "BadCase",
    "BadCaseCollector",
    "BadCaseAnalyzer",
    # 优化追踪
    "FixTracker",
    "OptimizationRound",
    # 数据增强
    "DataAugmenter",
    "ProductCorpus",
    "AugmentedExample",
    # Embedding 构建
    "EmbeddingKnowledgeBase",
    "EmbeddingDocument",
    # 生产数据收集
    "ProductionDataCollector",
    "ProductionSample",
]
