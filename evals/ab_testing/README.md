# A/B 测试框架

用于 AI 点单 Agent 的 A/B 测试框架，支持实验定义、流量分配、统计分析和报告生成。

## 核心组件

| 模块 | 说明 |
|-----|------|
| `experiment.py` | 实验定义、变体管理、流量分配 |
| `analyzer.py` | 统计检验、置信区间、效应量分析 |
| `runner.py` | 实验执行、数据收集、报告生成 |

## 快速开始

### 1. 创建实验

```python
from evals.ab_testing import ABExperiment, MetricDefinition

# 使用便捷方法创建
experiment = ABExperiment.create(
    name="意图分类模型对比",
    control_config={
        "model": "gpt-3.5-turbo",
        "method": "zero_shot"
    },
    treatment_config={
        "model": "gpt-4o-mini",
        "method": "few_shot"
    },
    primary_metric="intent_accuracy",
    description="对比不同模型在意图分类上的表现",
    hypothesis="GPT-4o-mini 的准确率将提高 5% 以上"
)

# 添加次要指标
experiment.secondary_metrics = [
    MetricDefinition(name="slot_f1", description="槽位提取 F1"),
    MetricDefinition(name="response_time", higher_is_better=False)
]

# 添加护栏指标
experiment.guardrail_metrics = [
    MetricDefinition(
        name="error_rate",
        higher_is_better=False,
        is_guardrail=True
    )
]
```

### 2. 启动实验并分配流量

```python
# 启动实验
experiment.start()

# 为用户分配变体
variant = experiment.allocate(user_id="user_123")

print(f"用户分配到: {variant.name}")
print(f"配置: {variant.config}")
```

### 3. 运行实验

```python
from evals.ab_testing import ABTestRunner

runner = ABTestRunner()

# 定义 Agent 函数
def my_agent(input_data, variant_config):
    # 根据 variant_config 执行不同逻辑
    model = variant_config.get("model")
    method = variant_config.get("method")

    # ... 实际执行逻辑 ...

    return {
        "intent": "ORDER_NEW",
        "confidence": 0.95,
        "slots": {"product": "美式咖啡"}
    }

# 批量运行测试
test_cases = [
    {"input": "来杯美式", "expected_intent": "ORDER_NEW"},
    {"input": "有什么推荐", "expected_intent": "RECOMMEND"},
    # ...
]

records = runner.run_batch(experiment, my_agent, test_cases)
```

### 4. 分析结果

```python
# 分析实验
analysis = runner.analyze_experiment(experiment)

# 查看结果
print(f"建议: {analysis.recommendation}")
print(f"是否得出结论: {analysis.is_conclusive}")
print(f"推荐采用: {analysis.winner}")

# 生成报告
report = runner.generate_report(analysis, format="markdown")
print(report)
```

## 统计分析功能

### 支持的检验方法

| 方法 | 适用场景 | 函数 |
|-----|---------|------|
| Welch's t-test | 连续指标（均值比较） | `analyzer.two_sample_ttest()` |
| z-test | 比例指标（转化率） | `analyzer.proportion_test()` |

### 多重比较校正

```python
from evals.ab_testing import ABTestAnalyzer

analyzer = ABTestAnalyzer()

# 多个指标的 p 值
p_values = [0.01, 0.03, 0.04, 0.08]

# Bonferroni 校正（保守）
corrected_bonf = analyzer.apply_multiple_testing_correction(
    p_values, method="bonferroni"
)

# Benjamini-Hochberg FDR 控制（相对宽松）
corrected_bh = analyzer.apply_multiple_testing_correction(
    p_values, method="fdr_bh"
)
```

### 样本量计算

```python
# 计算所需样本量
required_n = analyzer.calculate_sample_size(
    baseline_rate=0.85,           # 基线转化率
    minimum_detectable_effect=0.05,  # 5% 相对提升
    power=0.8,                    # 80% 统计功效
    significance_level=0.05       # 5% 显著性水平
)

# 估算运行时间
runtime_days = analyzer.estimate_runtime(
    required_sample_size=required_n,
    daily_traffic=1000,
    traffic_percentage=0.5,  # 50% 流量参与实验
    num_variants=2
)
```

## 流量分配策略

| 策略 | 说明 | 使用场景 |
|-----|------|---------|
| `RANDOM` | 完全随机分配 | 简单测试 |
| `USER_ID_HASH` | 基于用户ID哈希 | 确保同一用户始终分到同一组 |
| `SESSION_HASH` | 基于会话哈希 | 允许同一用户在不同会话中体验不同版本 |
| `DETERMINISTIC` | 确定性分配 | 测试/调试 |

```python
from evals.ab_testing import AllocationStrategy

experiment.allocation_strategy = AllocationStrategy.USER_ID_HASH
```

## 分层实验

支持按特定维度分层，确保各层内流量均匀分配：

```python
from evals.ab_testing import StratificationRule

experiment.stratification = [
    StratificationRule(
        field="intent_type",
        values=["ORDER_NEW", "ORDER_MODIFY", "RECOMMEND"],
        description="按意图类型分层"
    ),
    StratificationRule(
        field="user_type",
        values=["new", "returning"],
        description="按用户类型分层"
    )
]
```

## 护栏指标

护栏指标用于保护关键业务指标不受负面影响：

```python
experiment.guardrail_metrics = [
    MetricDefinition(
        name="error_rate",
        description="错误率",
        higher_is_better=False,
        is_guardrail=True
    ),
    MetricDefinition(
        name="latency_p95",
        description="P95 延迟",
        higher_is_better=False,
        is_guardrail=True
    )
]
```

当护栏指标出现显著负面变化时，系统会自动建议停止实验。

## 实验生命周期

```
DRAFT → SCHEDULED → RUNNING → COMPLETED
                  ↘         ↗
                   PAUSED
                      ↓
                   ABORTED
```

```python
# 启动实验
experiment.start()

# 暂停实验
experiment.pause()

# 恢复实验
experiment.resume()

# 完成实验
experiment.complete()

# 中止实验
experiment.abort(reason="护栏指标违规")
```

## 实验注册表

使用 `ExperimentRegistry` 管理多个实验：

```python
from evals.ab_testing import ExperimentRegistry

registry = ExperimentRegistry()

# 注册实验
registry.register(experiment)

# 获取实验
exp = registry.get("exp_12345678")

# 列出所有运行中的实验
running = registry.get_running_experiments()

# 按标签筛选
model_tests = registry.list_experiments(tags=["model_comparison"])
```

## 配置文件格式

实验可以保存为 YAML 文件：

```yaml
id: exp_model_compare
name: 意图分类模型对比
description: 对比 GPT-3.5 和 GPT-4o-mini
hypothesis: GPT-4o-mini 准确率提升 5% 以上

variants:
  - id: control
    name: 对照组
    weight: 0.5
    is_control: true
    config:
      model: gpt-3.5-turbo
      method: zero_shot

  - id: treatment
    name: 实验组
    weight: 0.5
    is_control: false
    config:
      model: gpt-4o-mini
      method: few_shot

primary_metrics:
  - name: intent_accuracy
    description: 意图分类准确率
    is_primary: true
    higher_is_better: true
    minimum_detectable_effect: 0.05

guardrail_metrics:
  - name: error_rate
    higher_is_better: false
    is_guardrail: true

allocation_strategy: user_hash
traffic_percentage: 0.5
min_runtime_days: 7
min_sample_per_variant: 500
```

## 最佳实践

### 1. 实验设计

- **明确假设**: 在实验开始前明确定义假设和预期效果
- **选择合适的指标**: 主要指标应直接反映业务目标
- **设置护栏指标**: 保护关键业务指标不受损害

### 2. 样本量

- **提前计算**: 使用 `calculate_sample_size()` 提前规划
- **考虑最小可检测效应**: 5% 相对变化通常需要较大样本量
- **预留安全边际**: 实际运行时间可能比预期长

### 3. 运行时间

- **最小运行时间**: 至少 7 天，覆盖周期性变化
- **达到目标样本量**: 确保统计功效足够
- **不要过早停止**: 避免"偷看"数据导致的假阳性

### 4. 分析结果

- **多重比较校正**: 多个指标时使用 FDR 控制
- **检查护栏指标**: 即使主要指标提升，护栏违规也应停止
- **考虑实际意义**: 统计显著不等于业务有意义

## 示例用例

### 对比不同分类方法

```python
experiment = ABExperiment.create(
    name="分类方法对比",
    control_config={"method": "zero_shot"},
    treatment_config={"method": "few_shot"},
    primary_metric="intent_accuracy"
)
```

### 对比不同模型

```python
experiment = ABExperiment.create(
    name="模型对比",
    control_config={"model": "gpt-3.5-turbo"},
    treatment_config={"model": "gpt-4o-mini"},
    primary_metric="task_completion_rate"
)
```

### 对比不同 Prompt 策略

```python
experiment = ABExperiment.create(
    name="Prompt 策略对比",
    control_config={"prompt_version": "v1"},
    treatment_config={"prompt_version": "v2_with_examples"},
    primary_metric="slot_f1"
)
```
