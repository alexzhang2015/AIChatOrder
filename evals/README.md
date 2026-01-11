# AI 点单 Agent 评估与优化系统

本模块提供完整的评估、优化和监控能力，帮助持续改进 AI 点单 Agent 的性能。

## 目录结构

```
evals/
├── README.md                 # 本文档
├── suites/                   # 评估套件定义
│   ├── intent_suite.py       # 意图识别评估
│   ├── slot_suite.py         # 槽位提取评估
│   └── dialogue_suite.py     # 多轮对话评估
├── results/                  # 评估结果存储
├── optimization/             # 持续优化模块
│   ├── OPTIMIZATION_STRATEGY.md  # 优化策略文档
│   ├── badcase_collector.py  # Bad Case 收集
│   ├── badcase_analyzer.py   # Bad Case 分析
│   ├── fix_tracker.py        # 优化追踪
│   ├── data_augmenter.py     # 数据增强
│   ├── embedding_builder.py  # Embedding 构建
│   ├── production_collector.py # 生产数据收集
│   └── cli.py                # CLI 工具
└── portal/                   # 运营监控 Portal
    ├── README.md             # Portal 文档
    └── api.py                # Portal API
```

## 快速开始

### 1. 运行评估

```bash
# 完整评估
python -m evals.run_standalone --suite full

# 仅意图评估
python -m evals.run_standalone --suite intent

# 仅槽位评估
python -m evals.run_standalone --suite slot
```

### 2. 使用优化 CLI

```bash
# 查看帮助
python -m evals.optimization.cli --help

# 数据增强
python -m evals.optimization.cli augment generate --num 30 --output data/

# 构建 Embedding
python -m evals.optimization.cli embedding build --output embeddings/

# Bad Case 分析
python -m evals.optimization.cli analyze --output report.json

# 优化追踪
python -m evals.optimization.cli track dashboard
```

### 3. 访问监控 Portal

启动服务后访问:
```
http://localhost:8000/portal
```

## 核心功能

### 评估框架 (Evaluation)

基于 Anthropic 评估方法论构建的评估系统:

- **意图识别评估**: 11 种意图类型的分类准确率
- **槽位提取评估**: 商品、规格、定制选项的 F1 分数
- **多轮对话评估**: 对话流程、状态管理、上下文理解

### 优化模块 (Optimization)

持续改进 Agent 性能的工具集:

| 模块 | 功能 |
|------|------|
| BadCaseCollector | 收集失败案例，按严重程度分类 |
| BadCaseAnalyzer | 分析失败模式，生成修复建议 |
| FixTracker | 追踪优化进度，记录指标变化 |
| DataAugmenter | 扩展商品语料，生成训练样本 |
| EmbeddingKnowledgeBase | 构建多层知识库 |
| ProductionDataCollector | 收集生产数据，管理标注队列 |

### 监控 Portal

Web 界面提供:

- 系统概览与趋势
- 评估执行与管理
- Bad Case 分析
- 优化追踪
- 数据管理
- 实时监控

## 优化工作流

推荐的优化流程:

```
1. 运行评估 → 获取基线指标
2. 分析 Bad Case → 识别问题模式
3. 数据增强 → 扩展训练数据
4. 更新 Embedding → 改进检索
5. 重新评估 → 验证改进效果
6. 追踪进度 → 记录优化历史
```

## 指标说明

### 意图识别
- **准确率 (Accuracy)**: 正确分类的比例
- **召回率 (Recall)**: 各意图类型的识别率
- **混淆矩阵**: 意图间的误分类分布

### 槽位提取
- **精确率 (Precision)**: 提取槽位的正确率
- **召回率 (Recall)**: 应提取槽位的覆盖率
- **F1 分数**: 精确率和召回率的调和平均

### 多轮对话
- **任务完成率**: 成功完成点单的比例
- **平均轮次**: 完成任务所需对话轮数
- **上下文保持**: 正确维护对话上下文的能力

## 配置

### 评估配置

编辑 `evals/config.yaml`:

```yaml
suites:
  intent:
    golden_file: data/golden_intents.yaml
    methods: [zero_shot, few_shot, rag_enhanced, function_calling]
  slot:
    schema_file: schema/slots_v2.yaml
```

### 优化配置

优化模块使用默认配置，可通过 CLI 参数调整:

```bash
# 增加每意图样本数
python -m evals.optimization.cli augment generate --num 50

# 设置 Embedding 输出目录
python -m evals.optimization.cli embedding build --output my_embeddings/
```

## 扩展指南

### 添加新评估套件

1. 在 `suites/` 下创建新文件
2. 实现 `EvaluationSuite` 接口
3. 在配置中注册套件

### 添加新优化策略

1. 在 `optimization/` 下创建模块
2. 在 `__init__.py` 中导出
3. 在 CLI 中添加命令

## 相关链接

- [优化策略详细文档](optimization/OPTIMIZATION_STRATEGY.md)
- [Portal 使用指南](portal/README.md)
- [项目主文档](../CLAUDE.md)
