# AI 点单 Agent 优化策略详解

## 一、优化架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            持续优化闭环                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│    │ 数据收集  │────▶│ 数据分析  │────▶│ 模型优化  │────▶│ 效果验证  │        │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘        │
│         │                                                    │              │
│         └────────────────────────────────────────────────────┘              │
│                              反馈循环                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                            四大优化策略                                       │
├───────────────┬───────────────┬───────────────┬────────────────────────────┤
│  1. 数据增强   │  2. Embedding │  3. Prompt    │  4. 规则引擎增强            │
│     策略      │     优化       │     调优       │                            │
└───────────────┴───────────────┴───────────────┴────────────────────────────┘
```

---

## 二、策略一：数据增强 (Data Augmentation)

### 2.1 商品语料扩充

#### 问题分析
当前商品数据有限，用户的表达方式多样化，导致：
- 商品别名覆盖不全（如 "续命咖啡" = 美式咖啡）
- 口语化表达识别率低
- 方言/网络用语支持不足

#### 解决方案

```yaml
# config/schema/product_corpus.yaml - 扩充商品语料

product_corpus:
  美式咖啡:
    # 标准名称
    canonical: "美式咖啡"

    # 正式名称变体
    formal_aliases:
      - "美式"
      - "Americano"
      - "冰美式"
      - "热美式"

    # 口语化表达
    colloquial:
      - "续命水"
      - "续命咖啡"
      - "续命液"
      - "黑咖啡"
      - "清咖"
      - "苦咖啡"

    # 网络用语
    internet_slang:
      - "命续"
      - "提神神器"
      - "打工人标配"
      - "肝帝必备"

    # 方言表达 (示例)
    dialect:
      粤语: ["黑咖", "齋啡"]
      上海话: ["黑咖啡"]

    # 常见错别字/拼音
    typos:
      - "没事咖啡"
      - "美事"
      - "meishi"
      - "americano"

    # 组合表达模式
    expression_patterns:
      - "{size}美式"           # 大杯美式
      - "{temperature}美式"    # 冰美式、热美式
      - "一杯{adj}美式"        # 一杯浓一点的美式
      - "{modifier}美式咖啡"   # 加浓美式咖啡

    # 语义关联词
    semantic_related:
      提神: 0.9
      简单: 0.8
      黑咖啡: 0.95
      无糖: 0.7
```

### 2.2 意图表达增强

```yaml
# config/schema/intent_expressions.yaml - 增强意图表达

intent_expressions:
  ORDER_NEW:
    # 直接表达
    direct:
      - "我要一杯{product}"
      - "来一杯{product}"
      - "给我一杯{product}"
      - "帮我点一杯{product}"

    # 间接表达
    indirect:
      - "有{product}吗"              # 可能是询问，也可能是点单
      - "{product}还有吗"
      - "想喝{product}"
      - "{product}来一个"

    # 省略表达
    elliptic:
      - "{product}"                  # 只说产品名
      - "一个{product}"
      - "要{product}"

    # 多商品表达
    multi_item:
      - "{product1}和{product2}"
      - "一杯{product1}，一杯{product2}"
      - "{num1}杯{product1}，{num2}杯{product2}"

    # 条件表达
    conditional:
      - "如果有{product}的话来一杯"
      - "{product}有的话要一个"

    # 口语化/网络化
    colloquial:
      - "整一杯{product}"
      - "搞一个{product}"
      - "来个{product}呗"
      - "rua一杯{product}"

  ORDER_CANCEL:
    # 直接取消
    direct:
      - "取消订单"
      - "我不要了"
      - "退掉"

    # 隐式取消 (重点优化)
    implicit:
      - "算了"
      - "不点了"
      - "还是不要了"
      - "先不喝了"
      - "下次再说"
      - "等会儿再点"
      - "改天吧"
      - "突然不想喝了"
```

### 2.3 数据增强工具实现

见 `data_augmenter.py`

---

## 三、策略二：Embedding 数据优化

### 3.1 构建多层次 Embedding 知识库

```
┌─────────────────────────────────────────────────────────────────┐
│                    Embedding 知识库架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Layer 1: 商品知识库 (Product Knowledge)                  │   │
│  │  - 商品名称 + 别名 + 描述                                 │   │
│  │  - 商品特征向量                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Layer 2: 意图示例库 (Intent Examples)                    │   │
│  │  - 每个意图的典型表达                                     │   │
│  │  - Few-shot 示例                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Layer 3: 对话历史库 (Dialogue History)                   │   │
│  │  - 成功对话案例                                           │   │
│  │  - 多轮对话上下文                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Layer 4: Bad Case 修复库 (Fixed Cases)                   │   │
│  │  - 已修复的 Bad Case                                      │   │
│  │  - 边界案例                                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Embedding 数据构建策略

```python
# 示例: Embedding 数据格式

embedding_data = {
    # 商品知识
    "products": [
        {
            "id": "latte_001",
            "text": "拿铁咖啡，经典意式浓缩与蒸奶的完美结合，口感丝滑细腻",
            "metadata": {
                "type": "product",
                "product_name": "拿铁",
                "category": "espresso_based",
                "keywords": ["拿铁", "latte", "奶咖", "丝滑"]
            }
        },
        {
            "id": "latte_alias_001",
            "text": "奶咖就是拿铁，加奶的咖啡",
            "metadata": {
                "type": "product_alias",
                "product_name": "拿铁",
                "alias": "奶咖"
            }
        }
    ],

    # 意图示例
    "intents": [
        {
            "id": "order_new_001",
            "text": "我想点一杯拿铁",
            "metadata": {
                "type": "intent_example",
                "intent": "ORDER_NEW",
                "slots": {"product_name": "拿铁"},
                "confidence": 1.0
            }
        },
        {
            "id": "order_cancel_implicit_001",
            "text": "算了还是不喝了",
            "metadata": {
                "type": "intent_example",
                "intent": "ORDER_CANCEL",
                "is_implicit": True,
                "difficulty": "hard"
            }
        }
    ],

    # 对话历史
    "dialogues": [
        {
            "id": "dialogue_001",
            "text": "用户: 来杯拿铁 | 助手: 好的，请问要什么杯型？| 用户: 大杯 | 助手: 好的，大杯拿铁",
            "metadata": {
                "type": "dialogue",
                "turns": 4,
                "success": True,
                "products": ["拿铁"]
            }
        }
    ],

    # Bad Case 修复
    "fixed_cases": [
        {
            "id": "fix_001",
            "text": "有什么推荐的咖啡吗？我想喝点提神的",
            "metadata": {
                "type": "fixed_case",
                "original_intent": "ORDER_NEW",  # 错误识别
                "correct_intent": "RECOMMEND",   # 正确意图
                "fix_date": "2024-01-15",
                "root_cause": "RECOMMEND vs ORDER_NEW 边界"
            }
        }
    ]
}
```

---

## 四、策略三：生产数据微调

### 4.1 用户输入收集与标注流程

```
┌──────────────────────────────────────────────────────────────────┐
│                     生产数据收集流程                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   用户输入 ──▶ 意图识别 ──▶ 置信度检查                             │
│                              │                                   │
│                    ┌─────────┴─────────┐                         │
│                    ▼                   ▼                         │
│              置信度 >= 0.9       置信度 < 0.9                     │
│                    │                   │                         │
│                    ▼                   ▼                         │
│              自动入库           人工标注队列                        │
│           (高质量样本)          (低置信度样本)                      │
│                                        │                         │
│                                        ▼                         │
│                                  标注完成后入库                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 微调数据格式

```json
// fine_tuning_data.jsonl - OpenAI Fine-tuning 格式
{"messages": [
  {"role": "system", "content": "你是咖啡店点单助手，识别用户意图和槽位"},
  {"role": "user", "content": "来杯续命水"},
  {"role": "assistant", "content": "{\"intent\": \"ORDER_NEW\", \"confidence\": 0.95, \"slots\": {\"product_name\": \"美式咖啡\"}, \"reasoning\": \"续命水是美式咖啡的网络用语\"}"}
]}

{"messages": [
  {"role": "system", "content": "你是咖啡店点单助手，识别用户意图和槽位"},
  {"role": "user", "content": "算了不喝了"},
  {"role": "assistant", "content": "{\"intent\": \"ORDER_CANCEL\", \"confidence\": 0.92, \"slots\": {}, \"reasoning\": \"算了表达取消意愿\"}"}
]}
```

### 4.3 Few-shot 动态选择策略

```python
# 根据用户输入动态选择最相关的 few-shot 示例

def select_few_shots(user_input: str, top_k: int = 5) -> List[Example]:
    """
    动态选择 Few-shot 示例策略:
    1. 语义相似度检索
    2. 意图平衡 (确保覆盖多个意图)
    3. 难度加权 (优先选择边界案例)
    """
    # 1. 语义检索
    similar = embedding_retriever.search(user_input, top_k * 2)

    # 2. 意图平衡
    intent_counts = Counter()
    balanced = []
    for example in similar:
        intent = example.metadata["intent"]
        if intent_counts[intent] < 2:  # 每个意图最多2个
            balanced.append(example)
            intent_counts[intent] += 1
        if len(balanced) >= top_k:
            break

    # 3. 加入边界案例
    if has_ambiguous_pattern(user_input):
        edge_cases = get_edge_cases_for_pattern(user_input)
        balanced = edge_cases[:2] + balanced[:-2]

    return balanced
```

---

## 五、策略四：规则引擎增强

### 5.1 多层规则架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        规则引擎架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: 前置规则 (Pre-LLM Rules)                              │
│  ├── 精确匹配规则: "取消订单" → ORDER_CANCEL (100%)              │
│  ├── 正则规则: /续命(水|咖啡)?/ → 美式咖啡                        │
│  └── 关键词规则: 包含"投诉" → COMPLAINT                          │
│                                                                 │
│  Layer 2: LLM 识别 (主流程)                                      │
│  └── OpenAI + RAG + Function Calling                            │
│                                                                 │
│  Layer 3: 后置规则 (Post-LLM Rules)                              │
│  ├── 置信度校验: < 0.7 触发人工复核                               │
│  ├── 约束检查: 星冰乐不能热 → 自动修正                            │
│  └── 业务规则: 晚上推荐低因咖啡                                   │
│                                                                 │
│  Layer 4: 回退规则 (Fallback Rules)                              │
│  └── LLM 失败时的降级策略                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 规则配置扩展

```yaml
# config/schema/rules_v2.yaml

# 前置精确规则 (跳过 LLM)
pre_rules:
  high_confidence:
    - pattern: "^取消(订单)?$"
      intent: ORDER_CANCEL
      confidence: 0.99
      skip_llm: true

    - pattern: "^(结账|买单|付款)$"
      intent: PAYMENT
      confidence: 0.99
      skip_llm: true

# 模糊表达增强
fuzzy_expressions_v2:
  # 产品识别
  product_slang:
    续命水: {product: "美式咖啡", temperature: "冰", shots: "双份"}
    肥宅快乐水: {product: "星冰乐"}
    dirty: {product: "馥芮白", shots: "双份"}
    澳瑞白: {product: "馥芮白"}
    脏咖啡: {product: "馥芮白"}

  # 修饰词映射
  modifiers:
    浓一点: {action: "add_shot"}
    提神: {action: "add_shot"}
    淡一点: {action: "reduce_shot"}
    不苦: {action: "add_sugar", sweetness: "半糖"}
    健康: {milk_type: "燕麦奶", sweetness: "无糖"}
    减脂: {milk_type: "脱脂奶", sweetness: "无糖"}

# 意图消歧规则
disambiguation:
  # RECOMMEND vs ORDER_NEW
  - condition:
      contains_any: ["推荐", "建议", "什么好"]
      not_contains: ["要", "来", "点"]
    prefer: RECOMMEND

  # ORDER_MODIFY vs ORDER_NEW
  - condition:
      has_context: true  # 已有订单
      contains_any: ["换", "改", "加", "减"]
    prefer: ORDER_MODIFY

# 上下文规则
context_rules:
  # 已有订单时的理解优化
  - when:
      has_active_order: true
    rules:
      - pattern: "再来一杯"
        intent: ORDER_NEW
        action: "duplicate_last_item"
      - pattern: "换成{product}"
        intent: ORDER_MODIFY
      - pattern: "不要了"
        intent: ORDER_CANCEL
        target: "last_item"  # 取消最后一项
```

---

## 六、实施路线图

### Phase 1: 数据基础建设 (1-2周)

1. **扩充商品语料库**
   - 收集 100+ 商品别名
   - 整理网络用语映射
   - 添加常见错别字

2. **构建 Embedding 知识库**
   - 商品知识 500+ 条
   - 意图示例 200+ 条
   - Bad Case 100+ 条

### Phase 2: 模型优化 (2-3周)

1. **Prompt 模板优化**
   - 基于 Bad Case 分析优化边界说明
   - 添加更多 Few-shot 示例

2. **RAG 检索优化**
   - 调整检索 top_k 参数
   - 实现动态 Few-shot 选择

### Phase 3: 规则增强 (1-2周)

1. **扩展前置规则**
   - 添加高置信度精确匹配
   - 优化模糊表达映射

2. **增强后置规则**
   - 完善约束检查
   - 添加业务规则

### Phase 4: 持续迭代 (持续)

1. **生产数据收集**
   - 自动收集低置信度样本
   - 用户反馈标注

2. **周期性评估**
   - 每周运行 Eval
   - 分析指标趋势

---

## 七、效果度量

### KPI 指标

| 指标 | 当前值 | 目标值 | 说明 |
|------|--------|--------|------|
| 意图准确率 | 85% | 95% | 核心指标 |
| 槽位 F1 | 80% | 92% | 提取准确性 |
| Bad Case 率 | 8% | 2% | 失败率 |
| 平均置信度 | 0.78 | 0.88 | 模型确定性 |

### 监控看板

```
┌────────────────────────────────────────────────────────────┐
│                    优化效果看板                              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  意图准确率趋势                 Bad Case 分布               │
│  ┌────────────────────┐       ┌────────────────────┐      │
│  │     ╱              │       │ ■ 意图混淆 45%      │      │
│  │   ╱                │       │ ■ 槽位错误 30%      │      │
│  │ ╱                  │       │ ■ 模糊表达 15%      │      │
│  │────────────────────│       │ ■ 其他 10%         │      │
│  └────────────────────┘       └────────────────────┘      │
│                                                            │
│  本周修复: 23 个 Bad Case                                   │
│  净减少: 18 个 (新增 5 个)                                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 八、工具使用

### 快速开始

```bash
# 1. 运行数据增强
python -m evals.optimization.data_augmenter --expand-products

# 2. 构建 Embedding 知识库
python -m evals.optimization.embedding_builder --rebuild

# 3. 运行评估
python -m evals.run_standalone --suite full --report html

# 4. 分析 Bad Case
python -m evals.optimization.cli analyze --input evals/results/

# 5. 生成修复建议
python -m evals.optimization.cli recommend --top 10
```

详细实现见各模块源码。
