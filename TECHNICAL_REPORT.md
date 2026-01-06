# AI 点单意图识别系统 - 技术汇报文档

> 基于大语言模型的智能咖啡店点单意图识别与多轮对话系统

---

## 一、项目概述

### 1.1 项目背景

随着大语言模型（LLM）技术的快速发展，自然语言理解能力得到了显著提升。本项目将 LLM 应用于咖啡店点单场景，构建了一个完整的**意图识别 + 槽位填充 + 多轮对话 + 技能执行**系统，实现智能化的点单交互体验。

### 1.2 核心能力

| 能力 | 描述 |
|-----|------|
| **意图识别** | 准确识别用户的11种点单相关意图 |
| **槽位提取** | 自动提取饮品名称、杯型、温度、甜度等7类槽位信息 |
| **多轮对话** | 基于LangGraph的状态机工作流，支持完整对话流程 |
| **配置化槽位** | YAML配置的槽位定义，支持热更新 |
| **技能执行** | Claude Skill风格的可配置执行层，8种内置技能 |
| **多方法对比** | 支持 Zero-shot、Few-shot、RAG、Function Calling 四种分类方法 |

### 1.3 技术栈

```
后端: Python 3.10+ / FastAPI / uvicorn
工作流: LangGraph (状态机对话管理)
大模型: OpenAI GPT-4o-mini（支持自定义 Base URL）
配置化: YAML Schema (槽位 + 技能)
前端: 原生 HTML/CSS/JavaScript（无框架依赖）
包管理: uv（现代化 Python 包管理器）
```

---

## 二、系统架构

### 2.1 整体架构图 (v3.0)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户界面层                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   意图分析演示页面   │    │        多轮对话点单页面              │ │
│  │   (index.html)      │    │        (chat.html)                  │ │
│  └─────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           API 服务层 (FastAPI)                       │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────────┐  │
│  │ /api/classify │ │ /api/compare  │ │ /api/chat (LangGraph)     │  │
│  │ 单次意图识别  │ │ 方法对比      │ │ 多轮对话+技能执行          │  │
│  └───────────────┘ └───────────────┘ └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    对话管理层 - LangGraph Workflow                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    OrderingWorkflow                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │    │
│  │  │ StateGraph  │  │ Checkpointer│  │  Conditional Edges  │  │    │
│  │  │ 状态图      │  │  状态持久化 │  │    条件路由         │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │    │
│  │                                                               │    │
│  │  节点: intent_recognition → route_by_intent → handle_* →    │    │
│  │        record_message → END                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   NLU层 - Slots      │ │  意图识别层           │ │  执行层 - Skills     │
│  (slots.yaml)        │ │  OpenAIClassifier    │ │  (skills.yaml)       │
│  ┌────────────────┐  │ │  ┌────────────────┐  │ │  ┌────────────────┐  │
│  │ SlotSchemaReg  │  │ │  │ Function Call  │  │ │  │ SkillRegistry  │  │
│  │ - 7个槽位定义  │  │ │  │ Zero/Few-shot  │  │ │  │ - 8个技能定义  │  │
│  │ - 值规范化    │  │ │  │ RAG Enhanced   │  │ │  │ - Handler执行  │  │
│  │ - 价格计算    │  │ │  └────────────────┘  │ │  │ - 测试框架     │  │
│  └────────────────┘  │ └──────────────────────┘ │  └────────────────┘  │
└──────────────────────┘                          └──────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           外部服务层                                  │
│                    ┌─────────────────────┐                          │
│                    │   OpenAI API        │                          │
│                    │   (GPT-4o-mini)     │                          │
│                    └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
用户输入 ──→ 意图识别 ──→ 槽位提取 ──→ 技能匹配 ──→ 状态更新 ──→ 回复生成
   │            │            │            │            │            │
   │            ▼            ▼            ▼            ▼            ▼
   │       ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
   │       │ LLM    │   │ Schema │   │ Skills │   │ State  │   │ Order  │
   │       │ 推理   │   │ 规范化 │   │ 执行   │   │ 持久化 │   │ 信息   │
   │       └────────┘   └────────┘   └────────┘   └────────┘   └────────┘
   │            │                          │
   │            ▼                          ▼
   │    ┌──────────────────┐    ┌──────────────────────────────┐
   │    │  意图识别结果     │    │  技能执行结果                 │
   │    │  - intent        │    │  - skill_id: nutrition_info  │
   │    │  - confidence    │    │  - data: {calories: 190}     │
   │    │  - slots         │    │  - message: "拿铁190大卡"    │
   │    └──────────────────┘    └──────────────────────────────┘
   │
   └──────────────────────────────────────────────────────→ 展示给用户
```

---

## 三、LangGraph 工作流架构

### 3.1 LangGraph 简介

LangGraph 是 LangChain 团队开发的状态机工作流框架，专为构建复杂的 AI Agent 对话系统设计。相比传统的链式调用，LangGraph 提供：

- **显式状态管理**：TypedDict 定义的强类型状态
- **条件路由**：基于状态的动态流程控制
- **状态持久化**：支持会话恢复和历史追溯
- **可视化**：Mermaid 图表自动生成

### 3.2 工作流状态定义

```python
class OrderState(TypedDict, total=False):
    # 会话信息
    session_id: str
    conversation_state: str  # greeting, taking_order, confirming, modifying, payment, completed

    # 当前轮次信息
    user_message: str
    intent: str
    confidence: float
    slots: Dict[str, Any]
    intent_result: Dict[str, Any]

    # 订单信息
    current_order: Optional[OrderDict]

    # 响应信息
    response: str
    suggestions: List[str]
    actions: List[str]

    # 技能执行结果
    skill_result: Optional[Dict[str, Any]]

    # 历史消息 (使用累加方式)
    messages: Annotated[List[MessageDict], operator.add]

    # 控制流
    next_node: str
    should_end: bool
```

### 3.3 工作流节点

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph 工作流节点                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [入口] intent_recognition                                     │
│       │                                                         │
│       ▼                                                         │
│   route_by_intent (条件路由)                                     │
│       │                                                         │
│       ├──→ handle_chitchat      (闲聊处理)                      │
│       ├──→ handle_new_order     (新订单)                        │
│       ├──→ handle_modify_order  (修改订单)                      │
│       ├──→ handle_cancel_order  (取消订单)                      │
│       ├──→ handle_query_order   (查询订单) ←── estimate_time    │
│       ├──→ handle_product_info  (商品信息) ←── nutrition_info   │
│       ├──→ handle_recommend     (推荐) ←── smart_recommend      │
│       ├──→ handle_payment       (支付确认)                      │
│       ├──→ handle_complaint     (投诉处理)                      │
│       └──→ handle_unknown       (未知意图)                      │
│               │                                                 │
│               ▼                                                 │
│       record_message (记录消息)                                  │
│               │                                                 │
│               ▼                                                 │
│           [END]                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 智能路由逻辑

```python
def route_by_intent(state: OrderState) -> str:
    """根据意图路由到对应的处理节点"""
    intent = state.get("intent", "UNKNOWN")
    current_order = state.get("current_order")
    slots = state.get("slots", {})
    user_message = state.get("user_message", "").lower()

    # 智能路由：基于上下文修正意图

    # 情况1: "确认下单" 被误识别为 ORDER_NEW
    if intent == "ORDER_NEW" and any(kw in user_message for kw in ["确认", "下单", "结账"]):
        if current_order and current_order.get("items"):
            if not slots.get("product_name"):
                return "handle_payment"

    # 情况2: "加一份浓缩" 被误识别为 ORDER_NEW
    if intent == "ORDER_NEW" and any(kw in user_message for kw in ["加一", "加份"]):
        if current_order and current_order.get("items"):
            extras_from_text = registry.extract_extras_from_text(user_message)
            if extras_from_text:
                return "handle_modify_order"

    routing = {
        "CHITCHAT": "handle_chitchat",
        "ORDER_NEW": "handle_new_order",
        "ORDER_MODIFY": "handle_modify_order",
        ...
    }

    return routing.get(intent, "handle_unknown")
```

---

## 四、配置化槽位系统

### 4.1 YAML Schema 设计

系统采用 YAML 配置文件定义槽位，支持热更新和动态 Function Calling Schema 生成。

**文件位置**: `schema/slots.yaml`

```yaml
version: "1.0.0"
description: "AI点单系统槽位配置"

# ==================== 槽位定义 ====================
slots:
  product_name:
    type: string
    required: true
    description: "饮品名称"
    examples: ["拿铁", "美式咖啡", "卡布奇诺"]

  size:
    type: enum
    default: "中杯"
    description: "杯型大小"
    values:
      - value: "中杯"
        aliases: ["中", "标准", "regular", "M"]
        price_delta: 0
      - value: "大杯"
        aliases: ["大", "large", "L"]
        price_delta: 4
      - value: "超大杯"
        aliases: ["特大", "venti", "XL"]
        price_delta: 7

  temperature:
    type: enum
    default: "热"
    values:
      - value: "热"
        aliases: ["热的", "hot", "常温"]
      - value: "冰"
        aliases: ["冰的", "iced", "加冰", "冻"]
      - value: "温"
        aliases: ["温的", "微热", "不烫"]

  milk_type:
    type: enum
    default: "全脂奶"
    values:
      - value: "全脂奶"
        aliases: ["普通奶", "牛奶"]
        price_delta: 0
      - value: "燕麦奶"
        aliases: ["燕麦", "oat"]
        price_delta: 6
      - value: "椰奶"
        aliases: ["椰子奶", "coconut"]
        price_delta: 6

  extras:
    type: array
    item_type: string
    description: "额外配料"
    allowed_values:
      - value: "浓缩shot"
        aliases: ["浓缩", "加浓", "extra shot"]
        price_delta: 6
      - value: "香草糖浆"
        aliases: ["香草", "vanilla"]
        price_delta: 4
```

### 4.2 SlotSchemaRegistry 核心功能

```python
class SlotSchemaRegistry:
    """槽位配置注册中心"""

    def normalize_slots(self, raw_slots: Dict) -> Dict:
        """规范化槽位值"""
        # 示例: "燕麦" → "燕麦奶", "大" → "大杯"
        normalized = {}
        for slot_name, value in raw_slots.items():
            slot_def = self.slots.get(slot_name)
            if slot_def and slot_def.type == "enum":
                normalized[slot_name] = self._normalize_enum_value(slot_def, value)
            else:
                normalized[slot_name] = value
        return normalized

    def generate_function_schema(self) -> Dict:
        """动态生成 OpenAI Function Calling Schema"""
        properties = {}
        for slot_id, slot in self.slots.items():
            if slot.type == "enum":
                properties[slot_id] = {
                    "type": "string",
                    "enum": [v.value for v in slot.values],
                    "description": slot.description
                }
            elif slot.type == "array":
                properties[slot_id] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": slot.description
                }
        return {"type": "object", "properties": properties}

    def get_price_deltas(self) -> Dict:
        """获取价格增量配置"""
        # 用于订单价格计算
```

### 4.3 动态价格计算

```
┌────────────────────────────────────────────────────────┐
│                    价格计算公式                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  单价 = 基础价格 + 升杯费 + 换奶费 + 配料费             │
│  总价 = 单价 × 数量                                    │
│                                                        │
│  示例：大杯燕麦奶拿铁 + 浓缩shot × 2                   │
│  = (32 + 4[大杯] + 6[燕麦奶] + 6[浓缩]) × 2           │
│  = 48 × 2 = ¥96                                       │
│                                                        │
│  * 所有价格增量从 slots.yaml 配置读取                  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 五、Skills 技能执行层

### 5.1 Skills 设计理念

借鉴 Claude Skill 的设计模式，实现了一个**可配置的技能执行层**：

| 概念 | Slot Schema | Skills |
|------|------------|--------|
| 作用 | 信息提取规则 | 能力执行定义 |
| 类比 | Claude Tool Parameter Schema | Claude Skill |
| 配置 | slots.yaml | skills.yaml |
| 触发 | LLM 输出 | 意图+关键词匹配 |

### 5.2 Skills YAML 配置

**文件位置**: `schema/skills.yaml`

```yaml
version: "1.0.0"
description: "AI点单系统技能定义"

# ==================== 技能分类 ====================
categories:
  order:
    name: "订单管理"
    icon: "shopping-cart"
  inventory:
    name: "库存管理"
    icon: "warehouse"
  payment:
    name: "支付相关"
    icon: "credit-card"
  recommendation:
    name: "推荐系统"
    icon: "star"

# ==================== 技能定义 ====================
skills:
  # ---------- 营养查询技能 ----------
  nutrition_info:
    name: "营养查询"
    description: "查询商品的营养成分信息"
    category: "inventory"
    enabled: true

    triggers:
      keywords: ["卡路里", "热量", "营养", "成分", "咖啡因"]
      intents: ["PRODUCT_INFO"]

    parameters:
      product_name:
        type: string
        required: true
      size:
        type: string
        default: "中杯"
      milk_type:
        type: string
        default: "全脂奶"

    returns:
      type: object
      properties:
        calories: {type: integer, description: "卡路里"}
        caffeine: {type: integer, description: "咖啡因(mg)"}
        sugar: {type: integer, description: "糖分(g)"}

    handler: "inventory.nutrition_info"

    test_cases:
      - name: "测试拿铁营养"
        input: {product_name: "拿铁", size: "中杯"}
        expected: {calories: 190}

  # ---------- 智能推荐技能 ----------
  smart_recommend:
    name: "智能推荐"
    description: "基于用户偏好和上下文进行个性化推荐"
    category: "recommendation"

    triggers:
      keywords: ["推荐", "什么好", "喝什么"]
      intents: ["RECOMMEND"]

    parameters:
      weather:
        type: string
        enum: ["hot", "cold", "normal"]
      preference:
        type: string
        examples: ["不太甜", "提神", "适合减肥"]

    handler: "recommendation.smart_recommend"
```

### 5.3 内置技能列表

| 技能ID | 名称 | 触发条件 | 功能 |
|--------|------|----------|------|
| `check_inventory` | 库存查询 | "还有吗"、"卖完了吗" | 查询商品库存状态 |
| `nutrition_info` | 营养查询 | "卡路里"、"热量" | 返回营养成分信息 |
| `smart_recommend` | 智能推荐 | "推荐"、"喝什么好" | 基于天气/偏好推荐 |
| `estimate_time` | 预估时间 | "多久"、"几分钟" | 预估订单制作时间 |
| `apply_coupon` | 应用优惠券 | "优惠券"、"折扣码" | 验证并应用优惠 |
| `check_points` | 积分查询 | "积分"、"会员" | 查询会员积分余额 |
| `store_info` | 门店查询 | "营业时间"、"地址" | 返回门店信息 |
| `modify_order` | 修改订单 | "修改"、"换成" | 修改已下订单 |

### 5.4 技能执行架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Skills 执行架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户输入: "拿铁有多少卡路里"                                    │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SkillRegistry.find_matching_skills()                     │  │
│  │  - 匹配意图: PRODUCT_INFO ✓                               │  │
│  │  - 匹配关键词: "卡路里" ✓                                  │  │
│  │  - 最佳匹配: nutrition_info (score=0.70)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SkillExecutor.execute("nutrition_info", params)          │  │
│  │  - 参数验证: product_name="拿铁" ✓                        │  │
│  │  - 查找Handler: NutritionInfoHandler                      │  │
│  │  - 执行计算: 中杯拿铁全脂奶 → 190大卡                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SkillResult                                              │  │
│  │  - success: true                                          │  │
│  │  - skill_id: "nutrition_info"                             │  │
│  │  - data: {calories: 190, caffeine: 150, sugar: 18}        │  │
│  │  - message: "中杯拿铁（全脂奶）营养信息：热量190大卡..."   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Handler 实现示例

```python
class NutritionInfoHandler(SkillHandler):
    """营养信息处理器"""

    NUTRITION = {
        "拿铁": {"calories": 190, "caffeine": 150, "sugar": 18, "protein": 10, "fat": 7},
        "美式咖啡": {"calories": 15, "caffeine": 225, "sugar": 0, "protein": 1, "fat": 0},
        ...
    }

    SIZE_MULTIPLIER = {"中杯": 1.0, "大杯": 1.3, "超大杯": 1.6}
    MILK_CALORIES = {"全脂奶": 0, "脱脂奶": -30, "燕麦奶": -20}

    def execute(self, params: Dict, context: Dict = None) -> SkillResult:
        product_name = params.get("product_name", "")
        size = params.get("size", "中杯")
        milk_type = params.get("milk_type", "全脂奶")

        nutrition = self.NUTRITION.get(product_name).copy()

        # 调整杯型
        multiplier = self.SIZE_MULTIPLIER.get(size, 1.0)
        nutrition["calories"] = round(nutrition["calories"] * multiplier)

        # 调整奶类
        nutrition["calories"] += self.MILK_CALORIES.get(milk_type, 0)

        message = f"{size}{product_name}（{milk_type}）营养信息：\n"
        message += f"热量: {nutrition['calories']}大卡 | 咖啡因: {nutrition['caffeine']}mg"

        return SkillResult(
            success=True,
            data=nutrition,
            message=message,
            skill_id="nutrition_info"
        )
```

### 5.6 测试框架

```python
class SkillTester:
    """技能测试器"""

    def run_all_tests(self) -> Dict:
        """运行所有技能的测试用例"""
        results = {"total": 0, "passed": 0, "failed": 0, "skills": {}}

        for skill in self.registry.skills.values():
            skill_result = self.run_skill_tests(skill.id)
            results["skills"][skill.id] = skill_result
            results["total"] += skill_result["total"]
            results["passed"] += skill_result["passed"]
            results["failed"] += skill_result["failed"]

        return results

# 测试结果示例
"""
总计: 8 个技能, 13 个用例
通过: 13, 失败: 0

✅ 库存查询 (2/2)
✅ 应用优惠券 (2/2)
✅ 积分查询 (1/1)
✅ 智能推荐 (2/2)
✅ 预估时间 (2/2)
✅ 营养查询 (2/2)
✅ 门店查询 (1/1)
✅ 修改订单 (1/1)
"""
```

---

## 六、关键技术点

### 6.1 意图识别方法对比

本系统实现了**四种主流的意图识别方法**，支持实时切换和效果对比：

#### 6.1.1 Zero-shot Classification（零样本分类）

**核心思想**：直接通过 Prompt 引导 LLM 理解任务，无需示例。

```python
ZERO_SHOT_PROMPT = """你是一个智能咖啡店点单助手的意图识别模块。

## 可识别的意图类型
- ORDER_NEW: 新建订单（用户想点新饮品）
- ORDER_MODIFY: 修改订单
...

## 输出格式（JSON）
{user_input}
"""
```

**优点**：通用性强，快速适应新领域，无需标注数据
**缺点**：对复杂场景识别精度可能不足

#### 6.1.2 Few-shot Classification（少样本分类）

**核心思想**：通过少量示例教导模型，利用 In-Context Learning 能力。

**优点**：通过示例提高精确度，便于控制输出格式
**缺点**：需要精心设计示例，增加 Token 消耗

#### 6.1.3 RAG-Enhanced Classification（检索增强分类）

**核心思想**：动态检索与输入最相似的历史案例，作为上下文参考。

**优点**：利用历史数据，更好处理边界情况
**缺点**：依赖检索质量，需要维护案例库

#### 6.1.4 Function Calling（函数调用）

**核心思想**：利用 LLM 的结构化输出能力，直接映射到 API 调用。

**优点**：输出格式可靠，便于系统集成
**缺点**：需要 LLM 支持 Function Calling

#### 6.1.5 方法对比总结

| 方法 | 精度 | Token消耗 | 响应速度 | 适用场景 |
|------|------|----------|---------|---------|
| Zero-shot | ⭐⭐⭐ | 低 | 快 | 快速原型、通用场景 |
| Few-shot | ⭐⭐⭐⭐ | 中 | 中 | 需要控制输出格式 |
| RAG | ⭐⭐⭐⭐⭐ | 中高 | 中 | 有历史数据积累 |
| Function Calling | ⭐⭐⭐⭐ | 低 | 快 | 需要结构化输出 |

---

### 6.2 会话状态机

```
┌──────────────────────────────────────────────────────────────────┐
│                        会话状态转换图                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────┐    CHITCHAT     ┌──────────────┐                 │
│    │GREETING │ ──────────────→ │ TAKING_ORDER │                 │
│    └─────────┘                 └──────┬───────┘                 │
│                                       │                          │
│                              ORDER_NEW│                          │
│                                       ▼                          │
│                               ┌──────────────┐                   │
│                    ┌─────────│  CONFIRMING  │←─────────┐        │
│                    │         └──────┬───────┘          │        │
│                    │                │                   │        │
│           ORDER_MODIFY        PAYMENT/确认        ORDER_NEW      │
│                    │                │                   │        │
│                    ▼                ▼                   │        │
│             ┌──────────────┐ ┌──────────────┐          │        │
│             │  MODIFYING   │ │  COMPLETED   │──────────┘        │
│             └──────────────┘ └──────────────┘                   │
│                                     │                            │
│                              ORDER_CANCEL                        │
│                                     ▼                            │
│                           ┌──────────────────┐                  │
│                           │   (订单取消)      │                  │
│                           └──────────────────┘                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 七、API 接口文档

### 7.1 接口一览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 意图分析演示页面 |
| `/chat` | GET | 多轮对话页面 |
| `/api/status` | GET | 系统状态 |
| `/api/classify` | POST | 单次意图分类 |
| `/api/compare` | POST | 多方法对比 |
| `/api/chat/new` | GET | 创建新会话 |
| `/api/chat` | POST | 发送对话消息 (LangGraph) |
| `/api/chat/reset` | POST | 重置会话 |
| `/api/workflow/graph` | GET | 获取工作流图 (Mermaid) |
| `/api/menu` | GET | 获取菜单 |

### 7.2 核心接口详情

#### POST /api/chat

**请求体**:
```json
{
    "message": "拿铁有多少卡路里",
    "session_id": "abc12345",
    "use_langgraph": true
}
```

**响应体**:
```json
{
    "session_id": "abc12345",
    "state": "taking_order",
    "reply": "中杯拿铁（全脂奶）营养信息：\n热量: 190大卡 | 咖啡因: 150mg\n糖: 18g | 蛋白质: 10g | 脂肪: 7g",
    "intent_result": {
        "intent": "PRODUCT_INFO",
        "confidence": 0.92,
        "slots": {"product_name": "拿铁"}
    },
    "skill_result": {
        "success": true,
        "skill_id": "nutrition_info",
        "data": {
            "calories": 190,
            "caffeine": 150,
            "sugar": 18
        },
        "message": "中杯拿铁（全脂奶）营养信息：..."
    },
    "suggestions": ["来杯拿铁", "看看其他", "有什么推荐"],
    "engine": "langgraph"
}
```

---

## 八、文件结构

```
demo/
├── main.py                 # FastAPI 应用入口
├── workflow.py             # LangGraph 工作流实现
├── slot_schema.py          # 槽位配置注册中心
├── skills.py               # 技能执行层实现
├── schema/
│   ├── slots.yaml          # 槽位定义配置
│   └── skills.yaml         # 技能定义配置
├── static/
│   ├── index.html          # 意图分析页面
│   └── chat.html           # 多轮对话页面
├── docs/
│   ├── business/           # 商业文档
│   ├── product/            # 产品文档
│   └── design/             # 设计文档
├── pyproject.toml          # 项目配置
└── TECHNICAL_REPORT.md     # 本文档
```

---

## 九、性能与扩展

### 9.1 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 平均响应时间 | ~800ms | 包含 LLM 调用 |
| 技能执行时间 | <10ms | 纯内存计算 |
| 规则引擎响应 | <10ms | Fallback 模式 |
| 会话状态持久化 | 支持 | LangGraph MemorySaver |
| 配置热更新 | 支持 | on_change 回调 |

### 9.2 扩展方向

1. **接入真实向量数据库**：使用 Milvus/Pinecone 替代简单相似度计算
2. **多语言支持**：扩展英语、日语等语言的意图识别
3. **持久化存储**：Redis/PostgreSQL 存储会话和订单
4. **语音交互**：接入 ASR/TTS 实现语音点单
5. **推荐系统**：基于用户历史偏好的个性化推荐
6. **更多技能**：外卖配送、会员系统、库存管理等

---

## 十、部署与运行

### 10.1 环境配置

```bash
# .env 文件
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
```

### 10.2 启动命令

```bash
cd demo
uv sync           # 安装依赖
uv run python main.py  # 启动服务
```

### 10.3 访问地址

- 意图分析: http://localhost:8000
- 多轮对话: http://localhost:8000/chat

---

## 十一、总结

本系统成功实现了基于大语言模型的智能点单意图识别系统，核心技术亮点包括：

1. **LangGraph 工作流**：状态机驱动的对话管理，支持状态持久化
2. **配置化槽位系统**：YAML 定义的槽位规则，支持热更新和动态 Schema 生成
3. **Skills 执行层**：Claude Skill 风格的可配置技能系统，8种内置技能
4. **四种意图识别方法**：Zero-shot、Few-shot、RAG、Function Calling 全覆盖
5. **混合槽位提取**：LLM + 规则引擎双保险
6. **智能订单系统**：动态价格计算、订单状态追踪
7. **优雅降级机制**：API 失败时自动切换规则引擎

该系统不仅可作为技术演示，也为实际业务落地提供了完整的技术参考架构。

---

*文档版本: v3.0*
*更新日期: 2025年12月*
