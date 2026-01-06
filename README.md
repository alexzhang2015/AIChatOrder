# AI 咖啡点单意图识别系统

基于大语言模型的智能咖啡店点单系统，支持多轮对话、意图识别、槽位提取和技能执行。

## 功能特性

- **多轮对话**: 基于 LangGraph 的状态机管理，支持上下文理解
- **意图识别**: 支持 4 种分类方法 (Zero-shot, Few-shot, RAG, Function Calling)
- **槽位提取**: YAML 配置化的槽位定义，支持别名和模糊匹配
- **规则引擎**: 产品约束验证、自动修正、模糊表达识别
- **技能框架**: 可扩展的技能执行系统，内置 8 个技能

## 支持的意图类型

| 意图 | 说明 | 示例 |
|------|------|------|
| ORDER_NEW | 新建订单 | "来杯大杯冰拿铁" |
| ORDER_MODIFY | 修改订单 | "换成燕麦奶" |
| ORDER_CANCEL | 取消订单 | "不要了" |
| ORDER_QUERY | 查询订单 | "我的订单到哪了" |
| PRODUCT_INFO | 商品咨询 | "拿铁多少钱" |
| RECOMMEND | 推荐请求 | "有什么推荐的" |
| PAYMENT | 支付相关 | "确认下单" |
| COMPLAINT | 投诉反馈 | "等太久了" |
| CHITCHAT | 闲聊 | "你好" |

## 快速开始

### 环境要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) 包管理器

### 安装

```bash
cd demo
uv sync
```

### 配置

创建 `.env` 文件：

```bash
cp .env.example .env
# 编辑 .env 添加你的 OpenAI API Key
```

### 运行

```bash
# 启动服务
uv run python main.py

# 开发模式（热重载）
uv run python main.py --reload
```

访问:
- 意图分析 Demo: http://localhost:8000
- 多轮对话 Demo: http://localhost:8000/chat

## 项目结构

```
demo/
├── main.py          # FastAPI 应用，意图分类器
├── workflow.py      # LangGraph 工作流
├── skills.py        # 技能执行框架
├── slot_schema.py   # 槽位配置加载器
├── rules_engine.py  # 规则引擎
├── schema/
│   ├── slots.yaml   # 槽位定义
│   ├── slots_v2.yaml# 扩展配置（模糊表达）
│   └── skills.yaml  # 技能定义
└── static/          # 前端页面
```

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/classify` | POST | 单次意图分类 |
| `/api/compare` | POST | 对比 4 种分类方法 |
| `/api/chat` | POST | 多轮对话 |
| `/api/chat/new` | GET | 创建新会话 |
| `/api/menu` | GET | 获取菜单 |
| `/api/status` | GET | 系统状态 |

## 运行测试

```bash
# 模块测试
uv run python workflow.py
uv run python skills.py
uv run python slot_schema.py
uv run python rules_engine.py

# Phase 1 功能测试
uv run python test_phase1.py

# 交互式测试
uv run python test_phase1.py -i
```

## 技术栈

- **后端**: FastAPI, LangGraph, OpenAI API
- **前端**: HTML/CSS/JavaScript
- **配置**: YAML
- **包管理**: uv

## 架构图

```
用户输入 → OpenAIClassifier (意图+槽位)
         → OrderingWorkflow.route_by_intent()
         → Handler Node (业务处理)
         → RulesEngine (约束验证)
         → SkillExecutor (技能匹配)
         → 响应 + 订单状态
```

## License

MIT
