# AI 点单 Agent 持续优化指南

基于 Eval 驱动的持续迭代优化方案

## 优化闭环流程

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│   │ 评估    │───▶│ 分析    │───▶│ 优化    │───▶│ 验证    │    │
│   │ Eval    │    │ BadCase │    │ 迭代    │    │ 回归    │    │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│        ▲                                            │          │
│        └────────────────────────────────────────────┘          │
│                      持续迭代                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 一、Bad Case 数据分析分类

针对 AI 点单场景的 Bad Case 分类体系：

### 1. 意图识别类问题

| 问题类型 | 示例 | 根因分析 |
|---------|------|---------|
| **边界意图混淆** | "看看有什么" → ORDER_NEW (应为 RECOMMEND) | Few-shot 示例不足 |
| **隐式意图** | "算了吧" → CHITCHAT (应为 ORDER_CANCEL) | 规则覆盖不全 |
| **复合意图** | "换成大杯的，再加一杯" → 只识别 MODIFY | 多意图拆分能力弱 |
| **否定表达** | "不要美式要拿铁" → 提取美式 | 否定词处理不当 |

### 2. 槽位提取类问题

| 问题类型 | 示例 | 根因分析 |
|---------|------|---------|
| **口语化表达** | "续命水" → 未识别为美式咖啡 | 模糊表达配置缺失 |
| **数量歧义** | "来两三杯" → quantity=2 或 3? | 模糊数量处理策略 |
| **指代消解** | "那个冰的" → 缺少上文关联 | 多轮上下文弱 |
| **同音/近义词** | "拿贴" → 未识别为拿铁 | 纠错能力不足 |

### 3. 对话管理类问题

| 问题类型 | 示例 | 根因分析 |
|---------|------|---------|
| **多轮追问弱** | 用户说"明天呢？" 没理解是问营业时间 | 上下文传递不完整 |
| **状态丢失** | 修改订单后忘记之前的选项 | 状态管理 bug |
| **过度确认** | 每个槽位都反复确认 | 对话策略过于保守 |
| **无关问题处理** | 问"今天天气" 回复点单信息 | 意图边界不清 |

### 4. 业务规则类问题

| 问题类型 | 示例 | 根因分析 |
|---------|------|---------|
| **约束未生效** | 星冰乐+热 没有自动修正 | 规则引擎配置漏 |
| **价格计算错** | 加料后总价不对 | 业务逻辑 bug |
| **库存未校验** | 推荐了已售罄产品 | 实时数据未接入 |

## 二、优化策略三板斧

### 策略 1: Prompt 调优

```yaml
# config/prompts/intent_classification_v2.yaml
optimization_history:
  - version: "1.0"
    date: "2024-01-01"
    changes: "初始版本"
    eval_score: 0.75

  - version: "1.1"
    date: "2024-01-15"
    changes: "增加隐式取消意图的 few-shot 示例"
    eval_score: 0.82
    bad_cases_fixed:
      - "算了吧 → ORDER_CANCEL"
      - "不要了 → ORDER_CANCEL"

  - version: "1.2"
    date: "2024-02-01"
    changes: "优化否定表达处理，增加边界说明"
    eval_score: 0.88

# 当前优化版本
current_prompt:
  system: |
    你是一个咖啡点单助手，负责识别用户意图。

    ## 意图边界说明（重要！）
    - ORDER_NEW: 用户明确表达要**点**某个产品
    - ORDER_MODIFY: 用户要**改**已有订单的属性
    - ORDER_CANCEL: 用户要**取消**，包括隐式表达如"算了"、"不要了"
    - RECOMMEND: 用户**询问**推荐，没有明确点单
    - CHITCHAT: 与点单**完全无关**的闲聊

    ## 易混淆场景处理
    1. "看看有什么" → RECOMMEND（不是 ORDER_NEW）
    2. "算了/不要了/不点了" → ORDER_CANCEL（不是 CHITCHAT）
    3. "再来一杯" → ORDER_NEW（新增，不是 MODIFY）

  few_shot_examples:
    # 针对 Bad Case 补充的示例
    - input: "算了吧"
      output: {intent: "ORDER_CANCEL", confidence: 0.9}
      note: "隐式取消"

    - input: "看看菜单"
      output: {intent: "RECOMMEND", confidence: 0.85}
      note: "浏览不是下单"

    - input: "不要美式，要拿铁"
      output: {intent: "ORDER_NEW", slots: {product_name: "拿铁"}}
      note: "否定+肯定，取肯定部分"
```

### 策略 2: 工程链路优化

```python
# services/preprocessing.py
"""
前置规则判断 + PE 前置拼装
在 LLM 调用前进行规则过滤，减少 LLM 压力
"""

class IntentPreprocessor:
    """意图预处理器 - 前置规则判断"""

    # 高置信度规则匹配（不需要 LLM）
    DETERMINISTIC_RULES = {
        "ORDER_CANCEL": [
            r"^(取消|不要了|算了|不点了|不想要了)$",
            r"(取消|撤销)(订单|点单)",
        ],
        "ORDER_QUERY": [
            r"(订单|咖啡).*(到哪|状态|好了没|准备)",
            r"查(一下|询)订单",
        ],
        "CHITCHAT": [
            r"^(你好|谢谢|好的|嗯|哈哈|再见)$",
            r"今天天气",
        ],
    }

    def preprocess(self, text: str) -> Optional[Dict]:
        """
        前置规则匹配
        返回 None 表示需要 LLM 处理
        """
        text = text.strip()

        for intent, patterns in self.DETERMINISTIC_RULES.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return {
                        "intent": intent,
                        "confidence": 0.95,
                        "method": "rule_based",
                        "matched_pattern": pattern
                    }

        return None  # 需要 LLM


class ContextEnhancer:
    """上下文增强器 - PE 前置拼装"""

    def enhance_prompt(
        self,
        user_input: str,
        conversation_history: List[Dict],
        current_order: Optional[Dict]
    ) -> str:
        """
        拼装增强后的 Prompt
        """
        context_parts = []

        # 1. 当前订单状态
        if current_order:
            context_parts.append(f"[当前订单] {self._format_order(current_order)}")

        # 2. 最近对话历史（保留关键信息）
        if conversation_history:
            recent = conversation_history[-3:]  # 最近3轮
            history_text = self._summarize_history(recent)
            context_parts.append(f"[对话历史] {history_text}")

        # 3. 用户输入
        context_parts.append(f"[用户说] {user_input}")

        # 4. 指代消解提示
        if self._has_reference(user_input):
            context_parts.append("[注意] 用户可能使用了指代词，请结合上下文理解")

        return "\n".join(context_parts)

    def _has_reference(self, text: str) -> bool:
        """检测是否有指代词"""
        references = ["那个", "这个", "它", "刚才", "上一个", "一样的"]
        return any(ref in text for ref in references)
```

### 策略 3: 数据精调

```yaml
# config/schema/slots_v2_optimized.yaml
# 基于 Bad Case 分析的数据迭代

fuzzy_expressions:
  # v1.1: 新增口语化表达（来自 Bad Case）
  product_name:
    - pattern: "续命水|续命咖啡"
      maps_to: "美式咖啡"
      confidence: 0.9
      added_in: "v1.1"
      source: "bad_case_20240115"

    - pattern: "dirty|脏脏"
      maps_to: "Dirty咖啡"
      confidence: 0.85
      added_in: "v1.1"

  sweetness:
    # v1.2: 优化模糊甜度表达
    - pattern: "正常甜|标准就行"
      maps_to: "标准"
      confidence: 0.9

    - pattern: "不要太甜|微甜|淡一点"
      maps_to: "少糖"
      confidence: 0.85
      note: "合并多个相似表达"

# FAQ 迭代 - 来自无法处理的问题
faq_additions:
  - question_patterns:
      - "你们几点开门"
      - "营业时间"
      - "明天几点"
    answer: "我们的营业时间是早上7:00到晚上10:00，全年无休哦~"
    intent: "BUSINESS_INFO"
    added_in: "v1.2"

  - question_patterns:
      - "可以外送吗"
      - "送不送外卖"
    answer: "支持外送哦，您可以通过美团或饿了么下单，也可以在这里点单后选择配送~"
    intent: "DELIVERY_INFO"
```

## 三、Bad Case 管理工作流

### 3.1 Bad Case 收集

```python
# evals/optimization/badcase_collector.py

@dataclass
class BadCase:
    """Bad Case 记录"""
    id: str
    timestamp: datetime
    source: str  # eval, production, manual

    # 输入
    user_input: str
    conversation_history: List[Dict]

    # 实际输出
    actual_intent: str
    actual_slots: Dict
    actual_response: str

    # 期望输出
    expected_intent: str
    expected_slots: Dict
    expected_response: str

    # 分析
    category: str  # intent_confusion, slot_extraction, dialogue, business
    root_cause: str
    severity: str  # critical, major, minor

    # 修复
    fix_strategy: str  # prompt, rule, data, code
    fix_status: str  # pending, in_progress, fixed, verified
    fixed_in_version: Optional[str]


class BadCaseCollector:
    """Bad Case 收集器"""

    def __init__(self, storage_path: str = "evals/optimization/badcases"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def collect_from_eval(self, eval_result: EvalResult) -> List[BadCase]:
        """从评估结果中收集 Bad Case"""
        badcases = []

        for trial in eval_result.trials:
            for grader_type, grader_result in trial.grader_results.items():
                for failure in grader_result.failures:
                    badcase = BadCase(
                        id=f"eval_{eval_result.task_id}_{len(badcases)}",
                        timestamp=datetime.now(),
                        source="eval",
                        user_input=str(failure.get("input", "")),
                        conversation_history=[],
                        actual_intent=failure.get("predicted_intent", ""),
                        actual_slots=failure.get("predicted_slots", {}),
                        actual_response="",
                        expected_intent=failure.get("expected_intent", ""),
                        expected_slots=failure.get("expected_slots", {}),
                        expected_response="",
                        category=self._categorize(failure),
                        root_cause="",
                        severity=self._assess_severity(failure),
                        fix_strategy="",
                        fix_status="pending",
                        fixed_in_version=None
                    )
                    badcases.append(badcase)

        return badcases

    def collect_from_production(self, log_entry: Dict) -> BadCase:
        """从生产日志中收集 Bad Case"""
        # 通常通过用户反馈或人工标注触发
        pass

    def save(self, badcases: List[BadCase]):
        """保存 Bad Case"""
        date_str = datetime.now().strftime("%Y%m%d")
        filepath = self.storage_path / f"badcases_{date_str}.json"

        existing = []
        if filepath.exists():
            with open(filepath, "r") as f:
                existing = json.load(f)

        existing.extend([asdict(bc) for bc in badcases])

        with open(filepath, "w") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
```

### 3.2 Bad Case 分析报告

```python
# evals/optimization/badcase_analyzer.py

class BadCaseAnalyzer:
    """Bad Case 分析器"""

    def analyze(self, badcases: List[BadCase]) -> Dict:
        """生成分析报告"""
        report = {
            "summary": {
                "total": len(badcases),
                "by_category": Counter(bc.category for bc in badcases),
                "by_severity": Counter(bc.severity for bc in badcases),
                "by_status": Counter(bc.fix_status for bc in badcases),
            },
            "top_patterns": self._find_patterns(badcases),
            "recommended_fixes": self._recommend_fixes(badcases),
            "priority_queue": self._prioritize(badcases),
        }
        return report

    def _find_patterns(self, badcases: List[BadCase]) -> List[Dict]:
        """发现共性模式"""
        patterns = []

        # 按实际意图分组，找出误分类模式
        intent_confusion = defaultdict(list)
        for bc in badcases:
            if bc.category == "intent_confusion":
                key = f"{bc.expected_intent} → {bc.actual_intent}"
                intent_confusion[key].append(bc.user_input)

        for confusion, examples in intent_confusion.items():
            if len(examples) >= 3:  # 至少3个相似 case
                patterns.append({
                    "type": "intent_confusion",
                    "pattern": confusion,
                    "count": len(examples),
                    "examples": examples[:5],
                    "suggested_fix": self._suggest_fix_for_confusion(confusion, examples)
                })

        return patterns

    def _recommend_fixes(self, badcases: List[BadCase]) -> List[Dict]:
        """推荐修复方案"""
        recommendations = []

        # 按修复策略分组
        by_strategy = defaultdict(list)
        for bc in badcases:
            if bc.fix_status == "pending":
                strategy = self._infer_fix_strategy(bc)
                by_strategy[strategy].append(bc)

        for strategy, cases in by_strategy.items():
            recommendations.append({
                "strategy": strategy,
                "count": len(cases),
                "effort": self._estimate_effort(strategy, len(cases)),
                "impact": self._estimate_impact(cases),
                "action_items": self._generate_action_items(strategy, cases)
            })

        return sorted(recommendations, key=lambda x: x["impact"], reverse=True)
```

### 3.3 自动化修复流程

```python
# evals/optimization/auto_fixer.py

class AutoFixer:
    """自动化修复器"""

    def __init__(self):
        self.prompt_updater = PromptUpdater()
        self.rule_updater = RuleUpdater()
        self.data_updater = DataUpdater()

    def apply_fixes(self, recommendations: List[Dict]) -> Dict:
        """应用推荐的修复"""
        results = {"applied": [], "skipped": [], "failed": []}

        for rec in recommendations:
            strategy = rec["strategy"]

            try:
                if strategy == "add_few_shot":
                    self.prompt_updater.add_few_shot_examples(rec["action_items"])
                    results["applied"].append(rec)

                elif strategy == "add_rule":
                    self.rule_updater.add_preprocessing_rules(rec["action_items"])
                    results["applied"].append(rec)

                elif strategy == "add_fuzzy_expr":
                    self.data_updater.add_fuzzy_expressions(rec["action_items"])
                    results["applied"].append(rec)

                else:
                    # 需要人工处理
                    results["skipped"].append(rec)

            except Exception as e:
                results["failed"].append({"recommendation": rec, "error": str(e)})

        return results


class PromptUpdater:
    """Prompt 更新器"""

    def add_few_shot_examples(self, examples: List[Dict]):
        """添加 few-shot 示例"""
        # 加载当前 prompt 配置
        config_path = "config/prompts/intent_classification.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 添加新示例
        existing = config.get("few_shot_examples", [])
        for ex in examples:
            if not self._is_duplicate(ex, existing):
                existing.append({
                    "input": ex["input"],
                    "output": ex["expected_output"],
                    "note": f"Added from bad_case: {ex.get('bad_case_id', 'unknown')}",
                    "added_date": datetime.now().isoformat()
                })

        config["few_shot_examples"] = existing
        config["version"] = self._increment_version(config.get("version", "1.0"))

        # 保存
        with open(config_path, "w") as f:
            yaml.dump(config, f, allow_unicode=True)
```

## 四、持续优化仪表盘

### 4.1 优化进度追踪

```
┌────────────────────────────────────────────────────────────────┐
│                    AI 点单 Agent 优化仪表盘                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Eval 通过率趋势                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  100% ┤                                          ●───●   │ │
│  │   90% ┤                              ●───●───●           │ │
│  │   80% ┤              ●───●───●───●                       │ │
│  │   70% ┤  ●───●───●                                       │ │
│  │   60% ┼──┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴─▶ │ │
│  │       W1  W2  W3  W4  W5  W6  W7  W8  W9  W10 W11 W12    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Bad Case 分布                      修复进度                    │
│  ┌────────────────────────┐        ┌────────────────────────┐ │
│  │ 意图混淆    ████████ 45%│        │ 已修复  ██████████ 62% │ │
│  │ 槽位提取    █████ 28%   │        │ 进行中  ████ 23%       │ │
│  │ 对话管理    ███ 15%     │        │ 待处理  ██ 15%         │ │
│  │ 业务规则    ██ 12%      │        │                        │ │
│  └────────────────────────┘        └────────────────────────┘ │
│                                                                │
│  本周重点修复项                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 1. [P0] 隐式取消意图 (算了/不要了) - 已修复 ✅              │ │
│  │ 2. [P0] 否定表达槽位提取 - 进行中 🔄                       │ │
│  │ 3. [P1] 多轮对话指代消解 - 待处理 ⏳                       │ │
│  │ 4. [P1] 口语化产品名映射 - 进行中 🔄                       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 每周优化 Checklist

```markdown
## 每周优化 Checklist

### 周一：数据收集
- [ ] 跑一轮完整 Eval Suite
- [ ] 收集上周生产环境 Bad Case
- [ ] 整理用户反馈

### 周二：分析归因
- [ ] Bad Case 分类打标
- [ ] 根因分析
- [ ] 识别共性 Pattern

### 周三：方案设计
- [ ] 确定修复策略（Prompt/Rule/Data/Code）
- [ ] 编写修复方案
- [ ] 评审优先级

### 周四：实施修复
- [ ] 应用修复
- [ ] 本地验证
- [ ] 提交 PR

### 周五：验证发布
- [ ] 跑回归 Eval，确认不降级
- [ ] 验证 Bad Case 修复效果
- [ ] 发布上线
- [ ] 更新优化文档
```

## 五、实施建议

### 阶段一：建立基线（Week 1-2）
1. 完善现有 Eval 任务覆盖
2. 建立 Bad Case 收集机制
3. 设定优化目标（如意图准确率 95%+）

### 阶段二：快速迭代（Week 3-6）
1. 每周一轮优化循环
2. 优先解决高频 Bad Case
3. 积累 Prompt 和数据资产

### 阶段三：精细打磨（Week 7+）
1. 处理长尾 Case
2. 优化对话体验
3. 建立自动化监控告警
