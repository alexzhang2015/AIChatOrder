# 评估体系增强计划

基于 `voice-ordering-agent-deep-eval-supplement.md` 文档分析，制定以下增强计划。

---

## 一、当前差距分析

### 已有能力 ✅
- 意图识别评估 (IntentGrader)
- 槽位提取评估 (SlotGrader, F1)
- 模糊表达评估 (FuzzyMatchGrader)
- 约束验证评估 (ConstraintGrader)
- 对话状态检查 (StateGrader)
- LLM 评分 (LLMRubricGrader)
- 优化工具链 (BadCase 收集/分析/追踪)
- 监控 Portal API

### 需补充能力 ❌
- L4/L5 业务层指标
- 技术指标 → 业务影响映射
- 用户模拟器 (当前 TODO 状态)
- A/B 测试框架
- 业务视角看板
- 延迟性能监控
- 行业基准对标

---

## 二、实施计划

### Phase 1: 业务指标体系 (P0) - ✅ 已完成

#### 交付物

| 文件 | 说明 |
|-----|------|
| `evals/metrics/business_metrics.yaml` | 分层指标体系定义 (L2-L5) |
| `evals/metrics/business_impact.py` | 技术-业务影响计算器 |
| `evals/metrics/__init__.py` | 模块导出 |
| `evals/harness/business_reporter.py` | 业务报告生成器 |
| Portal API 新增接口 | `/api/portal/business/*` |

#### 新增 API 接口

- `GET /api/portal/business/weekly-report` - 获取业务周报 (JSON)
- `GET /api/portal/business/weekly-report/markdown` - 获取业务周报 (Markdown)
- `POST /api/portal/business/impact-analysis` - 分析业务影响
- `GET /api/portal/business/metrics-config` - 获取业务指标配置

#### 使用示例

```python
from evals.metrics.business_impact import calculate_business_impact
from evals.harness.business_reporter import BusinessReporter

# 计算单指标影响
impact = calculate_business_impact("intent_accuracy", 0.85, 0.95)
print(impact.financial_impact)  # "月 GMV 影响: +¥120,000"

# 生成业务周报
reporter = BusinessReporter()
report = reporter.generate_weekly_report(
    current_metrics={"intent_accuracy": 0.92, ...},
    previous_metrics={"intent_accuracy": 0.89, ...}
)
print(reporter.format_report_markdown(report))
```

---

#### 1.1 定义端到端业务指标 (已实现)

```yaml
# evals/metrics/business_metrics.yaml

business_metrics:
  # L4: 端到端业务指标
  end_to_end:
    - name: order_completion_rate
      description: "成功完成订单 / 总尝试"
      formula: "completed_orders / total_attempts"
      target: "> 85%"

    - name: first_call_resolution
      description: "无需人工介入的订单比例"
      formula: "auto_resolved / total_orders"
      target: "> 90%"

    - name: order_accuracy
      description: "订单与用户意图完全一致"
      formula: "correct_orders / total_orders"
      target: "> 95%"

    - name: average_handling_time
      description: "平均完成订单时间"
      unit: "seconds"
      target: "< 90"
      benchmark_human: 120

  # L5: 用户体验指标 (需要生产数据)
  user_experience:
    - name: escalation_rate
      description: "转人工比例"
      target: "< 10%"
```

#### 1.2 技术-业务映射表

```python
# evals/metrics/business_impact.py

BUSINESS_IMPACT_MAPPING = {
    "intent_accuracy": {
        "from": 0.85,
        "to": 0.95,
        "business_metric": "order_correct_rate",
        "impact": "+10%",
        "financial_impact": "假设日均1000单，客单价40元，月增GMV约12万"
    },
    "first_call_resolution": {
        "from": 0.80,
        "to": 0.95,
        "business_metric": "人工介入成本",
        "impact": "-15%",
        "financial_impact": "假设人工处理成本5元/单，月省约2.25万"
    },
    "slot_f1": {
        "from": 0.80,
        "to": 0.92,
        "business_metric": "订单修改率",
        "impact": "-12%",
        "note": "槽位提取准确减少用户纠正次数"
    }
}
```

#### 1.3 交付物
- [ ] `evals/metrics/business_metrics.yaml` - 业务指标定义
- [ ] `evals/metrics/business_impact.py` - 影响映射计算
- [ ] `evals/harness/business_reporter.py` - 业务报告生成器

---

### Phase 2: 用户模拟器 (P1) - ✅ 已完成

#### 交付物

| 文件 | 说明 |
|-----|------|
| `evals/fixtures/personas.yaml` | 用户画像库 (11 个画像) |
| `evals/harness/user_simulator.py` | LLM 用户模拟器 |
| `evals/harness/environment.py` | 新增 agent_respond 方法 |
| `evals/harness/runner.py` | 实现 _run_simulation 方法 |
| `evals/tasks/conversation/*.yaml` | 6 个对话模拟测试用例 |

#### 用户画像类型

| 画像 ID | 名称 | 优先级 | 描述 |
|--------|------|--------|------|
| rushed_worker | 赶时间的上班族 | 高 | 快速点单，说话简洁 |
| coffee_novice | 咖啡小白 | 高 | 需要引导和推荐 |
| health_conscious | 健康顾虑者 | 高 | 乳糖不耐受，关注热量 |
| slang_user | 网络用语达人 | 高 | 使用网络流行语 |
| implicit_canceler | 隐式取消者 | 高 | 不直接说取消 |
| error_recovery_tester | 错误恢复测试者 | 高 | 测试系统纠错能力 |
| complex_customizer | 复杂定制用户 | 中 | 多个定制要求 |
| order_modifier | 频繁修改者 | 中 | 经常改变主意 |
| multi_item_orderer | 多杯订购者 | 中 | 一次点多杯 |
| dialect_speaker | 方言口音用户 | 中 | 带口音或方言 |
| out_of_scope_requester | 超范围请求者 | 中 | 测试边界处理 |

#### 使用示例

```python
from evals.harness.user_simulator import LLMUserSimulator, SimulationEvaluator
from evals.harness.environment import EvalEnvironment

# 创建模拟器
simulator = LLMUserSimulator()

# 获取画像
persona = simulator.get_persona("rushed_worker")

# 创建 Agent 响应函数
env = EvalEnvironment()
env.reset()

# 运行模拟
result = simulator.simulate_conversation(
    agent_respond_func=env.agent_respond,
    persona=persona,
    max_turns=10
)

# 评估结果
evaluator = SimulationEvaluator(simulator)
evaluation = evaluator.evaluate_result(result, persona)
print(f"评分: {evaluation['scores']['overall']:.2f}")
```

---

#### 2.1 完善对话模拟器（已实现）

`runner.py` 中 `_run_simulation` 方法已完整实现：

```python
# evals/harness/user_simulator.py

@dataclass
class UserPersona:
    """用户画像"""
    name: str
    description: str
    goal: str
    traits: List[str]
    constraints: List[str]

class LLMUserSimulator:
    """LLM 驱动的用户模拟器"""

    PERSONAS = [
        UserPersona(
            name="赶时间的上班族",
            description="早上赶着上班，希望快速点完",
            goal="点一杯大杯冰美式，最好30秒内搞定",
            traits=["impatient", "knows_what_they_want"],
            constraints=[]
        ),
        UserPersona(
            name="咖啡小白",
            description="不太懂咖啡，需要引导和推荐",
            goal="点一杯不太苦的咖啡",
            traits=["coffee_novice", "needs_guidance"],
            constraints=[]
        ),
        UserPersona(
            name="健康顾虑者",
            description="乳糖不耐受，关注热量",
            goal="点一杯拿铁，要燕麦奶，少糖",
            traits=["health_conscious"],
            constraints=["乳糖不耐受"]
        ),
        UserPersona(
            name="网络用语达人",
            description="年轻人，喜欢用网络流行语",
            goal="点一杯'续命水'",
            traits=["uses_slang", "casual"],
            constraints=[]
        ),
        UserPersona(
            name="复杂定制用户",
            description="对咖啡很挑剔，有很多自定义要求",
            goal="大杯冰燕麦拿铁，少糖，加浓缩，不要奶油",
            traits=["picky", "many_customizations"],
            constraints=[]
        ),
    ]

    async def simulate_conversation(
        self,
        agent,
        persona: UserPersona,
        max_turns: int = 10
    ) -> SimulationResult:
        """运行模拟对话"""
        pass
```

#### 2.2 交付物
- [x] `evals/harness/user_simulator.py` - 用户模拟器
- [x] `evals/fixtures/personas.yaml` - 用户画像库
- [x] 更新 `runner.py` 的 `_run_simulation` 方法
- [x] 新增 6 个对话模拟测试用例

---

### Phase 3: 增强评估任务 (P1) - ✅ 已完成

#### 交付物

| 文件 | 说明 |
|-----|------|
| `evals/tasks/intent/intent_boundary.yaml` | 意图边界测试 (37 个用例) |
| `evals/tasks/edge_cases/safety_boundary.yaml` | 安全边界测试 (18 个用例) |
| `evals/graders/confusion_grader.py` | 混淆矩阵 + 安全检查评分器 |
| `evals/harness/models.py` | 新增 CONFUSION_MATRIX, SAFETY_CHECK 类型 |

#### 新增评分器

| 评分器 | 功能 |
|-------|------|
| ConfusionMatrixGrader | 监控特定意图对的混淆率，超过阈值告警 |
| SafetyCheckGrader | 检查健康约束处理，验证推荐/排除项 |

#### 使用示例

```yaml
# 混淆矩阵评分器
graders:
  - type: confusion_matrix
    alert_pairs:
      - ["ORDER_NEW", "RECOMMEND"]
      - ["ORDER_MODIFY", "ORDER_NEW"]
    alert_threshold: 0.10

# 安全检查评分器
graders:
  - type: safety_check
    must_recommend: ["燕麦奶", "豆奶"]
    must_not_recommend: ["牛奶"]
```

---

#### 3.1 意图边界测试（已实现）

```yaml
# evals/tasks/intent/intent_boundary.yaml

task:
  id: "intent-boundary-001"
  name: "意图边界测试"
  description: "测试容易混淆的意图对"

  test_cases:
    # ORDER_NEW vs RECOMMEND
    - input: "有什么推荐的"
      expected_intent: RECOMMEND
    - input: "有拿铁吗"
      expected_intent: ORDER_NEW  # 隐式点单
    - input: "想喝点提神的"
      expected_intent: RECOMMEND
    - input: "来杯提神的"
      expected_intent: ORDER_NEW

    # ORDER_MODIFY vs ORDER_NEW (需要上下文)
    - input: "换成大杯"
      context: {has_order: true}
      expected_intent: ORDER_MODIFY
    - input: "要大杯"
      context: {has_order: false}
      expected_intent: ORDER_NEW

    # ORDER_CANCEL 隐式表达
    - input: "算了太贵了"
      expected_intent: ORDER_CANCEL
    - input: "还是不喝了"
      expected_intent: ORDER_CANCEL

  graders:
    - type: intent_accuracy
      min_accuracy: 0.95
    - type: confusion_matrix
      alert_pairs:
        - [ORDER_NEW, RECOMMEND]
        - [ORDER_MODIFY, ORDER_NEW]
      alert_threshold: 0.05
```

#### 3.2 安全边界测试

```yaml
# evals/tasks/edge_cases/safety_boundary.yaml

task:
  id: "safety-001"
  name: "安全边界测试"

  test_cases:
    # 过敏信息处理
    - input: "我乳糖不耐受，有什么推荐"
      expected_behavior:
        must_recommend: ["燕麦奶", "豆奶", "椰奶"]
        must_not_recommend: ["拿铁", "卡布奇诺"]  # 除非说明换奶

    # 超范围请求
    - input: "来一瓶茅台"
      expected_behavior: "礼貌说明没有该商品"

    - input: "帮我订个机票"
      expected_behavior: "礼貌引导回点单"
```

#### 3.3 交付物
- [x] `evals/tasks/intent/intent_boundary.yaml`
- [x] `evals/tasks/edge_cases/safety_boundary.yaml`
- [x] `evals/graders/confusion_grader.py` - 混淆率告警评分器

---

### Phase 4: 业务看板增强 (P1) - ✅ 已完成

#### 交付物

| 文件 | 说明 |
|-----|------|
| `evals/portal/business_dashboard.py` | 业务看板核心类 |
| `evals/portal/templates/business_report.html` | 可视化看板 HTML 模板 |
| `evals/portal/api.py` | 新增 4 个看板 API 接口 |

#### 新增 API 接口

| 接口 | 方法 | 说明 |
|-----|------|------|
| `/api/portal/business/dashboard` | GET | 获取完整看板数据 |
| `/api/portal/business/dashboard/html` | GET | 获取看板 HTML 页面 |
| `/api/portal/business/dashboard/custom` | POST | 自定义指标看板 |
| `/api/portal/business/health-score` | GET | 获取健康度评分 |

#### 看板组件

| 组件类型 | 功能 |
|---------|------|
| metric_card | 核心指标卡片（5 个） |
| trend_chart | 趋势图表数据 |
| pie_chart | 失败原因分布 |
| alert_list | 告警提醒 |
| action_list | 改进建议 |
| table | 业务影响预估 |

#### 使用示例

```python
from evals.portal.business_dashboard import BusinessDashboard

dashboard = BusinessDashboard()
view = dashboard.get_dashboard()

print(f"健康度: {view.summary['health_score']}")
print(f"状态: {view.summary['health_emoji']} {view.summary['health_text']}")
```

访问 HTML 看板:
```
GET http://localhost:8000/api/portal/business/dashboard/html
```

---

#### 4.1 业务视角报告（已实现）

```python
# evals/portal/business_dashboard.py

class BusinessDashboard:
    """面向业务的看板"""

    def generate_weekly_report(self) -> dict:
        return {
            "business_metrics": {
                "订单成功率": {
                    "current": "92%",
                    "last_week": "89%",
                    "trend": "↑ +3%",
                    "target": "95%",
                    "status": "🟡 接近目标"
                },
                # ...
            },
            "failure_analysis": {
                "理解错用户意图": {
                    "count": 85,
                    "percentage": "25%",
                    "typical_cases": [...],
                    "action": "增加意图边界训练数据"
                },
                # ...
            },
            "improvement_impact": {
                "本周上线优化": "增加 50 个网络用语映射",
                "直接效果": "'续命水'识别率从 60% → 95%",
                "业务影响": "预计每日减少 20 单识别失败"
            }
        }
```

#### 4.2 交付物
- [x] `evals/portal/business_dashboard.py`
- [x] `evals/portal/templates/business_report.html`
- [x] Portal API 新增看板接口

---

### Phase 5: 性能与基准 (P2) - ✅ 已完成

#### 交付物

| 文件 | 说明 |
|-----|------|
| `evals/metrics/latency_requirements.yaml` | 延迟性能要求配置 |
| `evals/metrics/industry_benchmarks.yaml` | 行业基准对标配置 |
| `evals/harness/latency_collector.py` | 延迟收集器 |
| `evals/graders/performance_grader.py` | 性能评分器 (3 个) |
| `evals/tasks/performance/*.yaml` | 性能测试任务 (2 个) |

#### 新增评分器

| 评分器 | 功能 |
|-------|------|
| LatencyGrader | 评估延迟是否满足 SLA (P50/P95/P99) |
| BenchmarkGrader | 与行业基准对比准确率 |
| PerformanceProfileGrader | 综合延迟+准确率生成性能画像 |

#### 延迟收集器功能

- 多组件延迟测量 (端到端、意图分类、槽位提取、LLM 生成等)
- 百分位数统计 (P50, P90, P95, P99)
- SLA 违规检测和告警
- 健康度评分 (0-100)
- 行业基准对比
- 优化建议生成

#### 使用示例

```yaml
# 延迟性能评分
graders:
  - type: latency
    p50_target: 500
    p95_target: 1000
    p99_target: 2000
    critical: 3000

# 行业基准对比
graders:
  - type: benchmark
    metric: "intent_accuracy"
    benchmark_source: "industry"

# 性能画像
graders:
  - type: performance_profile
    latency_weight: 0.3
    accuracy_weight: 0.7
```

```python
from evals.harness.latency_collector import LatencyCollector, LatencyComponent

# 创建收集器
collector = LatencyCollector()

# 记录延迟
collector.record(LatencyComponent.END_TO_END, 450.0, intent="ORDER_NEW")

# 获取统计
stats = collector.get_stats()
print(f"P95: {stats.p95}ms")

# 生成报告
report = collector.generate_report()
print(f"健康度: {report.health_score}/100 (等级: {report.grade})")
```

#### 5.1 延迟监控（已实现）

```yaml
# evals/metrics/latency_requirements.yaml

end_to_end:
  targets:
    p50: 500    # 50% 请求应在 500ms 内完成
    p95: 1000   # 95% 请求应在 1000ms 内完成
    p99: 2000   # 99% 请求应在 2000ms 内完成
  sla:
    critical: 3000  # 超过此值视为严重问题
    warning: 1500   # 超过此值发出警告

component_breakdown:
  intent_classification:
    targets: {p50: 150, p95: 300, p99: 500}
  slot_extraction:
    targets: {p50: 100, p95: 200, p99: 400}
  llm_generation:
    targets: {p50: 200, p95: 400, p99: 800}
```

#### 5.2 行业基准（已实现）

```yaml
# evals/metrics/industry_benchmarks.yaml

intent_recognition:
  overall_accuracy:
    excellent: 0.98
    good: 0.95
    acceptable: 0.90
    industry_average: 0.92

slot_extraction:
  overall_f1:
    excellent: 0.95
    good: 0.90
    acceptable: 0.85
    industry_average: 0.88

dialogue_management:
  task_completion_rate:
    excellent: 0.92
    good: 0.85
    acceptable: 0.80
    industry_average: 0.83
```

#### 5.3 交付物
- [x] `evals/metrics/latency_requirements.yaml`
- [x] `evals/metrics/industry_benchmarks.yaml`
- [x] `evals/harness/latency_collector.py`
- [x] `evals/graders/performance_grader.py`
- [x] `evals/tasks/performance/latency_benchmark.yaml`
- [x] `evals/tasks/performance/accuracy_benchmark.yaml`

---

### Phase 6: A/B 测试框架 (P2) - ✅ 已完成

#### 交付物

| 文件 | 说明 |
|-----|------|
| `evals/ab_testing/experiment.py` | 实验定义、变体管理、流量分配 |
| `evals/ab_testing/analyzer.py` | 统计检验、置信区间、效应量分析 |
| `evals/ab_testing/runner.py` | 实验执行、数据收集、报告生成 |
| `evals/ab_testing/README.md` | 完整使用文档 |

#### 核心功能

| 功能 | 说明 |
|-----|------|
| ABExperiment | 实验定义类，支持变体、指标、分层配置 |
| ABTestAnalyzer | 统计分析器，支持 t-test、比例检验、多重比较校正 |
| ABTestRunner | 实验运行器，协调执行、收集数据、生成报告 |
| ExperimentRegistry | 实验注册表，管理多个实验生命周期 |

#### 统计分析功能

- Welch's t-test (均值比较)
- z-test for proportions (比例检验)
- 置信区间计算
- 效应量计算 (Cohen's d)
- 统计功效分析
- 多重比较校正 (Bonferroni, BH FDR)
- 样本量计算
- 运行时间估算

#### 流量分配策略

| 策略 | 说明 |
|-----|------|
| RANDOM | 完全随机分配 |
| USER_ID_HASH | 基于用户ID哈希（确保一致性） |
| SESSION_HASH | 基于会话哈希 |
| DETERMINISTIC | 确定性分配（测试用） |

#### 使用示例

```python
from evals.ab_testing import ABExperiment, ABTestRunner, MetricDefinition

# 创建实验
experiment = ABExperiment.create(
    name="分类方法对比",
    control_config={"method": "zero_shot"},
    treatment_config={"method": "few_shot"},
    primary_metric="intent_accuracy"
)

# 添加护栏指标
experiment.guardrail_metrics = [
    MetricDefinition(name="error_rate", higher_is_better=False, is_guardrail=True)
]

# 启动实验
experiment.start()

# 运行并分析
runner = ABTestRunner()
records = runner.run_batch(experiment, agent_func, test_cases)
analysis = runner.analyze_experiment(experiment)

# 生成报告
report = runner.generate_report(analysis)
print(f"建议: {analysis.recommendation}")
print(f"推荐采用: {analysis.winner}")
```

#### 6.1 实验框架（已实现）

```python
# evals/ab_testing/experiment.py

@dataclass
class ABExperiment:
    id: str
    name: str
    description: str
    hypothesis: str

    variants: List[Variant]
    primary_metrics: List[MetricDefinition]
    secondary_metrics: List[MetricDefinition]
    guardrail_metrics: List[MetricDefinition]

    allocation_strategy: AllocationStrategy
    traffic_percentage: float
    stratification: List[StratificationRule]

    status: ExperimentStatus
    # ...
```

#### 6.2 交付物
- [x] `evals/ab_testing/experiment.py`
- [x] `evals/ab_testing/analyzer.py`
- [x] `evals/ab_testing/runner.py`
- [x] `evals/ab_testing/README.md`
- [x] `evals/ab_testing/__init__.py`

---

## 三、优先级排序

| 阶段 | 内容 | 优先级 | 预计工时 |
|-----|------|--------|---------|
| Phase 1 | 业务指标体系 | P0 | 3-5 天 |
| Phase 2 | 用户模拟器 | P1 | 5-7 天 |
| Phase 3 | 增强评估任务 | P1 | 3-5 天 |
| Phase 4 | 业务看板增强 | P1 | 3-5 天 |
| Phase 5 | 性能与基准 | P2 | 3-5 天 |
| Phase 6 | A/B 测试框架 | P2 | 5-7 天 |

**总计：22-34 天**

---

## 四、立即可执行项

### 今天就能开始：

1. **创建业务指标定义文件**
   ```bash
   mkdir -p evals/metrics
   touch evals/metrics/business_metrics.yaml
   ```

2. **添加意图边界测试用例**
   - 扩展 `evals/tasks/intent/` 下的测试

3. **更新现有评估报告增加业务视角**
   - 在 `reporter.py` 中增加业务指标计算

### 需要团队讨论：

1. 业务指标的目标值设定
2. 用户画像库的完善
3. A/B 测试的分流策略
