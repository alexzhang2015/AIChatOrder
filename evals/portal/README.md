# AI 点单 Agent 运营监控 Portal

运营监控 Portal 是一个综合性的管理界面，用于监控、评估和优化 AI 点单 Agent 的性能。

## 功能概览

### 1. 系统概览 (Overview)
- **核心指标展示**: 意图准确率、槽位 F1、Bad Case 数量、日均请求量
- **趋势图表**: 7天性能趋势可视化
- **系统警告**: 实时显示需要关注的问题
- **最近活动**: 评估运行、Bad Case、优化操作的时间线

### 2. 评估管理 (Evaluation)
- **运行评估**: 支持多种评估套件（full、intent、slot、dialogue）
- **查看状态**: 实时显示评估运行状态
- **历史记录**: 浏览历史评估结果和详细报告

### 3. Bad Case 管理 (Bad Cases)
- **列表视图**: 分页展示所有 Bad Case
- **过滤功能**: 按严重程度、类别、状态筛选
- **详情查看**: 查看 Bad Case 的完整信息
- **导出功能**: 导出 Bad Case 进行离线分析

### 4. 优化追踪 (Optimization)
- **仪表盘**: 显示优化进度和关键指标
- **优化历史**: 查看所有优化轮次的详细记录
- **开始新轮次**: 创建新的优化周期
- **完成轮次**: 记录优化结果和改进

### 5. 数据管理 (Data Management)
- **数据增强**: 生成训练样本、边界案例
- **Embedding 构建**: 构建和更新知识库
- **生产数据**: 查看待标注样本、导出训练数据

### 6. 实时监控 (Real-time Monitor)
- **健康检查**: 系统各组件的健康状态
- **性能指标**: 请求延迟、吞吐量、错误率
- **意图分布**: 实时意图识别分布

## 访问方式

启动服务后访问:
```
http://localhost:8000/portal
```

## API 端点

Portal 提供以下 API 端点 (前缀 `/api/portal`):

### 概览
- `GET /overview` - 获取系统概览数据
- `GET /trends` - 获取趋势数据

### 评估
- `GET /eval/status` - 获取评估状态
- `POST /eval/run` - 运行评估
- `GET /eval/history` - 获取评估历史
- `GET /eval/report/{eval_id}` - 获取评估报告

### Bad Case
- `GET /badcases` - 获取 Bad Case 列表
- `GET /badcases/{case_id}` - 获取 Bad Case 详情
- `POST /badcases/export` - 导出 Bad Case

### 优化
- `GET /optimization/dashboard` - 获取优化仪表盘
- `GET /optimization/history` - 获取优化历史
- `POST /optimization/start` - 开始新优化轮次
- `POST /optimization/complete` - 完成优化轮次

### 数据增强
- `POST /augment/generate` - 生成增强数据
- `GET /augment/stats` - 获取增强统计

### Embedding
- `POST /embedding/build` - 构建 Embedding
- `GET /embedding/stats` - 获取 Embedding 统计
- `POST /embedding/search` - 搜索 Embedding

### 生产数据
- `GET /production/pending` - 获取待标注样本
- `POST /production/export` - 导出训练数据
- `GET /production/stats` - 获取生产数据统计

### 监控
- `GET /monitor/health` - 获取健康状态
- `GET /monitor/metrics` - 获取性能指标
- `GET /monitor/intents` - 获取意图分布

## 技术架构

```
Portal 架构
├── 前端 (static/portal.html)
│   ├── 单页应用 (SPA)
│   ├── Chart.js 图表
│   └── 响应式设计
│
└── 后端 (evals/portal/api.py)
    ├── FastAPI Router
    ├── 集成 BadCaseCollector
    ├── 集成 FixTracker
    ├── 集成 DataAugmenter
    ├── 集成 EmbeddingKnowledgeBase
    └── 集成 ProductionDataCollector
```

## 使用示例

### 运行完整评估流程

1. 打开 Portal: `http://localhost:8000/portal`
2. 点击「评估管理」
3. 选择评估套件「full」
4. 点击「运行评估」
5. 等待评估完成，查看结果

### 处理 Bad Case

1. 进入「Bad Case 管理」
2. 使用过滤器定位严重问题
3. 点击具体案例查看详情
4. 记录问题原因，规划修复

### 优化跟踪

1. 进入「优化追踪」
2. 点击「开始新轮次」创建优化周期
3. 完成优化后，点击「完成轮次」
4. 查看历史趋势了解改进效果

## 配置说明

Portal 集成到主应用中，无需额外配置。确保以下模块可用:

```python
from evals.optimization import (
    BadCaseCollector,
    BadCaseAnalyzer,
    FixTracker,
    DataAugmenter,
    EmbeddingKnowledgeBase,
    ProductionDataCollector
)
```

## 故障排除

### Portal 页面无法加载
- 确保 `static/portal.html` 文件存在
- 检查服务器日志是否有错误

### API 返回错误
- 检查 `evals/portal/api.py` 是否正确导入
- 确认依赖模块已正确安装

### 评估无法运行
- 确保 OpenAI API 密钥已配置
- 检查 `evals/` 目录权限

## 相关文档

- [优化策略文档](../optimization/OPTIMIZATION_STRATEGY.md)
- [评估框架文档](../README.md)
- [CLI 工具使用](../optimization/cli.py)
