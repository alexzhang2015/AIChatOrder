# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered coffee ordering system with intent recognition, slot extraction, and multi-turn dialogue. Uses LangGraph for conversation state management and supports 4 classification methods (zero-shot, few-shot, RAG, function calling).

## Development Commands

All commands run from `demo/` directory:

```bash
# Install dependencies
uv sync

# Start development server (port 8000)
uv run python -m app.main
uv run python -m app.main --reload  # with hot reload

# Run module tests (each module has if __name__ == "__main__" test code)
uv run python -m workflow.ordering      # LangGraph workflow tests
uv run python -m services.skills        # Skill execution tests
uv run python -m models.slot_schema     # Schema normalization tests
uv run python -m services.rules_engine  # Rules engine tests
uv run python -m services.classifier    # Classifier tests

# Run Phase 1 feature tests
uv run python -m tests.test_phase1      # Automated tests
uv run python -m tests.test_phase1 -i   # Interactive mode

# Run all tests with pytest
uv run pytest tests/
```

## Environment Setup

Create `demo/.env`:
```
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1  # optional
```

## Architecture

```
demo/
├── app/
│   ├── main.py              # FastAPI app, API endpoints
│   └── api/schemas.py       # Pydantic request/response models
├── core/
│   ├── types.py             # Enums: Intent, SessionState, OrderStatus
│   └── interfaces.py        # Abstract base classes
├── config/
│   ├── settings.py          # Pydantic Settings (env vars)
│   └── schema/              # YAML config files
│       ├── slots.yaml       # Slot definitions, aliases, menu
│       ├── slots_v2.yaml    # Extended: constraints, fuzzy expressions
│       └── skills.yaml      # Skill definitions with test cases
├── models/
│   ├── slot_schema.py       # SlotSchemaRegistry, YAML loader
│   ├── order.py             # Order & OrderItem models
│   └── session.py           # Session models
├── nlp/
│   ├── intent_registry.py   # Intent registry with descriptions
│   ├── extractor.py         # SlotExtractor
│   └── prompts.py           # LLM prompt templates
├── services/
│   ├── classifier.py        # OpenAIClassifier (4 methods)
│   ├── rules_engine.py      # CustomizationRulesEngine, FuzzyMatcher
│   ├── skills.py            # SkillRegistry, SkillExecutor, handlers
│   └── session_manager.py   # Session management
├── workflow/
│   └── ordering.py          # LangGraph StateGraph, OrderingWorkflow
├── infrastructure/
│   ├── cache.py             # LRU/TTL caching
│   ├── database.py          # SQLite ORM models
│   ├── resilience.py        # Circuit breaker pattern
│   └── monitoring.py        # Structured logging & metrics
├── evals/                   # Evaluation framework
│   ├── harness/             # Core engine, runner, models
│   ├── graders/             # 14 grader implementations
│   ├── metrics/             # Business metrics & benchmarks
│   ├── ab_testing/          # A/B testing framework
│   ├── portal/              # Operations Portal API
│   ├── tasks/               # Test case definitions
│   └── fixtures/            # User personas, test data
├── tests/                   # pytest test suite
└── static/                  # Frontend HTML/CSS/JS
```

## Data Flow

```
User Input → OpenAIClassifier (intent + slots)
           → OrderingWorkflow.route_by_intent()
           → Handler Node (handle_new_order, handle_modify, etc.)
           → RulesEngine.validate_and_adjust() (constraints)
           → SkillExecutor (optional skill matching)
           → Response + Order State
```

## Core Components

**OpenAIClassifier** (`services/classifier.py`): Intent classification with 4 methods (zero-shot, few-shot, RAG, function calling). Falls back to rule-based matching when OpenAI unavailable.

**OrderingWorkflow** (`workflow/ordering.py`): LangGraph StateGraph. Entry point `intent_recognition` → route to handler → `record_message` → END. Uses MemorySaver for session state.

**CustomizationRulesEngine** (`services/rules_engine.py`): Validates product+slot combinations (e.g., 星冰乐 can't be hot). Auto-corrects invalid combos.

**FuzzyExpressionMatcher** (`services/rules_engine.py`): Matches colloquial expressions like "不要那么甜" → 半糖, "续命水" → 美式咖啡.

**SkillExecutor** (`services/skills.py`): Executes skills by ID or auto-matches by text/intent. 8 built-in handlers for inventory, coupons, recommendations, etc.

## Intent Types

ORDER_NEW, ORDER_MODIFY, ORDER_CANCEL, ORDER_QUERY, PRODUCT_INFO, RECOMMEND, CUSTOMIZE, PAYMENT, COMPLAINT, CHITCHAT, UNKNOWN

## Key URLs

- `http://localhost:8000` - Operations Portal (default homepage)
- `http://localhost:8000/intent` - Intent analysis demo
- `http://localhost:8000/chat` - Multi-turn dialogue
- `http://localhost:8000/api/status` - System status
- `http://localhost:8000/api/workflow/graph` - Mermaid diagram

## Adding New Skills

1. Add to `config/schema/skills.yaml` with triggers, parameters, test_cases
2. Create handler class extending `SkillHandler` in `services/skills.py`
3. Register: `registry.register_handler("category.skill_id", MyHandler())`

## Adding Product Constraints

Edit `config/schema/slots_v2.yaml`:
```yaml
product_constraints:
  产品名:
    temperature:
      only: ["冰", "热"]
      auto_correct: "冰"
      error_message: "..."
```

## Adding Fuzzy Expressions

Edit `config/schema/slots_v2.yaml`:
```yaml
fuzzy_expressions:
  sweetness:
    - pattern: "不要那么甜|没那么甜"
      maps_to: "半糖"
      confidence: 0.85
```

## Language

Code comments and user-facing text are in Chinese (Simplified). Designed for Chinese coffee ordering scenarios.

## Evaluation System (evals/)

Comprehensive evaluation framework with 6 completed phases:

```
evals/
├── harness/                  # Core evaluation engine
│   ├── runner.py             # EvalRunner, test execution
│   ├── models.py             # GraderConfig, TestCase, GraderResult
│   ├── environment.py        # EvalEnvironment, agent integration
│   ├── user_simulator.py     # LLM-driven user simulation (11 personas)
│   ├── latency_collector.py  # Performance monitoring (P50/P95/P99)
│   └── business_reporter.py  # Business metrics reporting
├── graders/                  # 14 grader types
│   ├── base.py               # BaseGrader interface
│   ├── intent_grader.py      # Intent accuracy
│   ├── slot_grader.py        # Slot F1 score
│   ├── confusion_grader.py   # Confusion matrix, safety checks
│   └── performance_grader.py # Latency, benchmark, profile graders
├── metrics/                  # Business metrics
│   ├── business_metrics.yaml # L2-L5 metric definitions
│   ├── business_impact.py    # Tech→business impact mapping
│   ├── latency_requirements.yaml
│   └── industry_benchmarks.yaml
├── ab_testing/               # A/B testing framework
│   ├── experiment.py         # ABExperiment, Variant, ExperimentRegistry
│   ├── analyzer.py           # Statistical tests (t-test, z-test, FDR)
│   └── runner.py             # ABTestRunner, TrialRecord
├── portal/                   # Operations Portal backend
│   ├── api.py                # FastAPI routes (/api/portal/*)
│   └── business_dashboard.py # Dashboard widgets
├── tasks/                    # Test case definitions (YAML)
└── fixtures/                 # Test data, personas
```

### Running Evaluations

```bash
# Run evaluation suite
uv run python -c "from evals.harness.runner import EvalRunner; ..."

# Test A/B framework
uv run python -c "from evals.ab_testing import ABExperiment, ABTestAnalyzer; ..."
```

### Grader Types

| Type | Description |
|------|-------------|
| intent_accuracy | Intent classification accuracy |
| slot_f1 | Slot extraction F1 score |
| fuzzy_match | Fuzzy expression matching |
| constraint_validation | Product constraint validation |
| confusion_matrix | Intent confusion monitoring |
| safety_check | Health/dietary constraint handling |
| latency | P50/P95/P99 latency SLA |
| benchmark | Industry benchmark comparison |
| performance_profile | Combined latency + accuracy |

## Operations Portal

Unified portal at `/portal` with:

| Page | Description |
|------|-------------|
| 概览 | System overview, metrics, trends |
| 评估管理 | Run evaluations, view history |
| Bad Case | Error analysis and tracking |
| 优化追踪 | Optimization rounds tracking |
| 数据管理 | Data augmentation, embeddings |
| 实时监控 | Real-time metrics monitoring |
| 系统状态 | System health, components |
| 业务看板 | Business KPIs, health score |
| 工作流图 | LangGraph workflow visualization |
| A/B 测试 | Experiment management |

### Portal API Endpoints

```
GET  /api/portal/overview          # System overview
GET  /api/portal/business/dashboard # Business dashboard data
GET  /api/portal/ab/experiments    # List A/B experiments
POST /api/portal/ab/experiments    # Create experiment
POST /api/portal/ab/experiments/{id}/start   # Start experiment
GET  /api/portal/ab/experiments/{id}/analysis # Statistical analysis
```
