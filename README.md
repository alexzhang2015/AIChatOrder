# AI 咖啡点单意图识别系统 (AIChatOrder)

基于大语言模型（LLM）的智能咖啡店点单系统。本项目演示了如何构建一个生产级质量的 AI Agent，涵盖了多轮对话管理、意图识别、复杂槽位提取、业务规则引擎、技能执行以及全链路监控。

## 🌟 核心特性

*   **🧠 智能对话管理 (LangGraph)**: 采用 LangGraph 构建状态机工作流，支持上下文保持、状态持久化 (SQLite) 和复杂的对话分支。
*   **🎯 多策略意图识别**: 支持 Zero-shot, Few-shot, RAG (检索增强), Function Calling 四种分类方法，可动态切换。
*   **🛠️ 强大的规则引擎 (Phase 1)**:
    *   **模糊表达匹配**: 支持"不要那么甜"（半糖）、"续命水"（美式）等口语化表达。
    *   **组合约束验证**: 自动校验产品规则（如"星冰乐不能做热的"）并自动修正。
    *   **增强槽位标准化**: 支持模糊匹配和别名映射（如"澳白" -> "馥芮白"）。
*   **🧩 可扩展技能系统 (Skills)**: 类似 Claude Skill 的插件化架构，内置库存查询、营养分析、智能推荐、优惠券等 8 种技能，通过 YAML 配置驱动。
*   **📊 全链路监控**: 内置结构化日志、性能指标收集 (Metrics) 和请求追踪 (Request ID)。
*   **⚙️ 现代化配置**: 基于 Pydantic Settings 和 YAML 的双层配置管理。

## 🏗️ 系统架构

```mermaid
graph TD
    User[用户输入] --> API[FastAPI 网关]
    API --> Monitor[监控中间件]
    Monitor --> Workflow[LangGraph 工作流]
    
    subgraph "Core Workflow (OrderingWorkflow)"
        Workflow --> IntentNode[意图识别节点]
        IntentNode --> Router{智能路由}
        
        Router -->|新订单| OrderNew[创建订单节点]
        Router -->|修改| OrderModify[修改订单节点]
        Router -->|查询| OrderQuery[查询/技能节点]
        Router -->|其他| OtherHandlers[其他业务节点]
    end
    
    subgraph "NLU & Logic Layer"
        IntentNode --> LLM[OpenAI / LLM]
        OrderNew --> Rules[规则引擎 (RulesEngine)]
        OrderModify --> Rules
        OrderQuery --> Skills[技能执行器 (SkillExecutor)]
    end
    
    subgraph "Configuration & Data"
        Rules --> SlotConfig[Slots Schema (YAML)]
        Skills --> SkillConfig[Skills Schema (YAML)]
        Workflow --> DB[(SQLite / VectorDB)]
    end
    
    Rules --校验/修正--> OrderNew
    Skills --执行结果--> OrderQuery
    
    Workflow --> Response[生成响应]
```

## 🚀 快速开始

### 1. 环境准备

本项目使用 [uv](https://github.com/astral-sh/uv) 进行极速包管理（Python 3.10+）。

```bash
# 安装依赖
uv sync
```

### 2. 配置

创建并编辑环境变量文件：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 OPENAI_API_KEY
```

### 3. 运行服务

```bash
# 启动 API 服务 (支持热重载)
uv run python main.py --reload
```

*   **意图分析演示**: [http://localhost:8000](http://localhost:8000)
*   **多轮对话演示**: [http://localhost:8000/chat](http://localhost:8000/chat)

## 📂 项目结构

```
AIChatOrder/
├── main.py              # 程序入口，FastAPI 应用定义
├── workflow.py          # LangGraph 工作流定义 (核心逻辑)
├── rules_engine.py      # 规则引擎 (模糊匹配、约束验证)
├── skills.py            # 技能执行系统 (Inventory, Nutrition, etc.)
├── monitoring.py        # 监控、日志和指标收集
├── config.py            # Pydantic 配置管理
├── database.py          # 数据库模型与操作
├── schema/              # YAML 配置文件
│   ├── slots.yaml       # 基础槽位定义
│   ├── slots_v2.yaml    # 增强规则配置 (模糊表达、约束)
│   ├── skills.yaml      # 技能定义与测试用例
│   └── intents.yaml     # 意图定义
├── static/              # 前端演示页面
├── test_phase1.py       # 功能验收测试脚本
└── pyproject.toml       # 项目依赖配置
```

## 🧪 测试与验证

本项目包含完善的测试脚本，用于验证规则引擎和业务逻辑。

```bash
# 1. 运行 Phase 1 核心功能测试 (规则、模糊匹配、约束)
uv run python test_phase1.py

# 2. 启动交互式命令行测试
uv run python test_phase1.py -i

# 3. 运行模块独立测试
uv run python skills.py      # 测试技能系统
uv run python workflow.py    # 测试工作流逻辑
```

## 🔧 核心组件说明

### 1. 规则引擎 (`rules_engine.py`)
负责处理复杂的业务逻辑，确保订单的有效性。
*   **输入**: "来杯热的星冰乐"
*   **处理**: 识别 "星冰乐" + "热"，触发 `product_constraints`。
*   **输出**: 自动修正为 "冰"，并提示用户。

### 2. 技能系统 (`skills.py`)
通过 `schema/skills.yaml` 定义工具，支持动态参数验证。
*   **示例**: 用户问 "拿铁多少热量？" -> 触发 `nutrition_info` 技能 -> 返回卡路里数据。

### 3. 监控系统 (`monitoring.py`)
*   **Request ID**: 全链路追踪。
*   **Metrics**: 记录 API 耗时、成功率、分类置信度等指标。
*   **Structured Log**: JSON 格式日志，自动脱敏。

## 📝 License

MIT