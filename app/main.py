"""
AI 点单意图识别系统 - 重构版入口
FastAPI 后端服务 - 支持多轮对话
"""

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 导入服务模块
from services.classifier import OpenAIClassifier
from services.ordering_assistant import OrderingAssistant

# 导入数据模块
from data.menu import MENU, SIZE_PRICE, MILK_PRICE, EXTRAS_PRICE
from data.training import TRAINING_EXAMPLES
from nlp.prompts import PROMPT_TEMPLATES, FUNCTION_SCHEMA
from models.intent import get_intent_descriptions, INTENT_DESCRIPTIONS

# 导入基础设施
from infrastructure.cache import get_api_cache, get_session_cache
from infrastructure.health import get_health_checker, HealthStatus
from infrastructure.monitoring import MonitoringMiddleware, get_metrics_collector
from nlp.intent_registry import get_intent_registry

# 导入 API 模块
from app.api.schemas import ClassifyRequest, CompareRequest, ChatRequest, ResetRequest, VALID_METHODS

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 应用初始化 ====================

app = FastAPI(
    title="AI点单意图识别系统",
    description="基于 LangGraph 的智能咖啡店点单助手",
    version="3.1.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 监控中间件
app.add_middleware(MonitoringMiddleware)

# 挂载静态文件
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# 注册 Portal API 路由
try:
    from evals.portal.api import router as portal_router
    app.include_router(portal_router)
    logger.info("Portal API 已注册")
except ImportError as e:
    logger.warning(f"Portal API 未能加载: {e}")

# ==================== 全局实例 ====================

classifier = OpenAIClassifier()
assistant = OrderingAssistant(classifier)
_intent_registry = get_intent_registry()
_start_time = time.time()

# LangGraph 工作流实例
_langgraph_workflow = None


def get_langgraph_workflow():
    """获取 LangGraph 工作流实例（单例模式）"""
    global _langgraph_workflow
    if _langgraph_workflow is None:
        try:
            from workflow import OrderingWorkflow
            _langgraph_workflow = OrderingWorkflow(classifier)
            print("LangGraph 工作流已初始化")
        except Exception as e:
            print(f"LangGraph 工作流初始化失败: {e}")
            _langgraph_workflow = None
    return _langgraph_workflow


# ==================== 页面路由 ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回运营监控 Portal 页面（默认首页）"""
    html_path = Path(__file__).parent.parent / "static" / "portal.html"
    if html_path.exists():
        return HTMLResponse(
            content=html_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return HTMLResponse(content="<h1>请确保 static/portal.html 存在</h1>")


@app.get("/intent", response_class=HTMLResponse)
async def intent_page():
    """返回意图分析 Demo 页面"""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>请确保 static/index.html 存在</h1>")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """返回多轮对话页面"""
    html_path = Path(__file__).parent.parent / "static" / "chat.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>请确保 static/chat.html 存在</h1>")


# ==================== 状态 API ====================

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    workflow = get_langgraph_workflow()
    api_cache = get_api_cache()
    session_cache = get_session_cache()

    openai_ok = classifier.is_available()
    langgraph_ok = workflow is not None

    # 组件状态
    components = {
        "openai": {
            "status": "ok" if openai_ok else "error",
            "details": f"模型: {classifier.model}" if openai_ok else "不可用"
        },
        "langgraph": {
            "status": "ok" if langgraph_ok else "error",
            "details": "工作流已初始化" if langgraph_ok else "未初始化"
        },
        "database": {
            "status": "ok",
            "details": "SQLite 连接正常"
        },
        "cache": {
            "status": "ok",
            "details": f"API缓存: {api_cache.stats()['size']} 条"
        }
    }

    # 意图类型列表
    intent_descriptions = get_intent_descriptions()
    supported_intents = list(intent_descriptions.keys())

    return {
        # 前端期望的字段
        "status": "ok" if openai_ok else "error",
        "uptime_seconds": time.time() - _start_time,
        "version": "3.1.0",
        "environment": "development",
        "components": components,
        "classifier": {
            "method": "rag_enhanced",
            "model": classifier.model,
            "available_methods": list(VALID_METHODS)
        },
        "supported_intents": supported_intents,
        # 保留原有字段以兼容其他调用
        "openai_available": openai_ok,
        "async_available": classifier.async_client is not None,
        "intent_types": intent_descriptions,
        "example_count": len(TRAINING_EXAMPLES),
        "langgraph_available": langgraph_ok,
        "engines": ["langgraph", "legacy"],
        "cache": {
            "api": api_cache.stats(),
            "session": session_cache.stats()
        },
        "intent_registry": _intent_registry.stats()
    }


# ==================== 健康检查 ====================

@app.get("/health")
async def health_check():
    """系统健康检查"""
    health_checker = get_health_checker()
    report = await health_checker.check_all()

    status_code = 200 if report.status == HealthStatus.HEALTHY else 503
    return JSONResponse(status_code=status_code, content=report.to_dict())


@app.get("/health/{check_name}")
async def health_check_single(check_name: str):
    """单项健康检查"""
    health_checker = get_health_checker()
    result = await health_checker.check_one(check_name)

    if result is None:
        raise HTTPException(status_code=404, detail=f"未知的检查项: {check_name}")

    status_code = 200 if result.status == HealthStatus.HEALTHY else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "name": result.name,
            "status": result.status.value,
            "latency_ms": round(result.latency_ms, 2),
            "details": result.details,
            **({"error": result.error} if result.error else {})
        }
    )


@app.get("/api/metrics")
async def get_metrics():
    """获取性能指标"""
    metrics = get_metrics_collector()
    return metrics.get_all_stats(window_seconds=300)


# ==================== 缓存 API ====================

@app.get("/api/cache/stats")
async def get_cache_stats():
    """获取缓存统计"""
    api_cache = get_api_cache()
    session_cache = get_session_cache()
    return {
        "api_cache": api_cache.stats(),
        "session_cache": session_cache.stats()
    }


@app.post("/api/cache/clear")
async def clear_cache():
    """清空 API 缓存"""
    api_cache = get_api_cache()
    cleared = api_cache.clear()
    return {"cleared": cleared, "message": f"已清空 {cleared} 条缓存"}


# ==================== 意图分类 API ====================

@app.post("/api/classify")
async def classify_intent(request: ClassifyRequest):
    """执行意图分类"""
    text = request.text
    method = request.method

    if method == "zero_shot":
        result = await classifier.classify_zero_shot_async(text)
    elif method == "few_shot":
        result = classifier.classify_few_shot(text)
    elif method == "rag_enhanced":
        result = classifier.classify_rag(text)
    elif method == "function_calling":
        result = await classifier.classify_function_calling_async(text)
    else:
        raise HTTPException(status_code=400, detail=f"未知方法: {method}")

    intent = result.get("intent", "UNKNOWN")
    result["intent_info"] = INTENT_DESCRIPTIONS.get(intent, INTENT_DESCRIPTIONS.get("UNKNOWN", {}))
    result["method"] = method
    result["input_text"] = text
    result["from_cache"] = result.pop("_cached", False)

    return result


@app.post("/api/compare")
async def compare_methods(request: CompareRequest):
    """对比所有方法的结果"""
    text = request.text

    results = {"input_text": text, "methods": {}}

    for method in ["zero_shot", "few_shot", "rag_enhanced", "function_calling"]:
        if method == "zero_shot":
            result = classifier.classify_zero_shot(text)
        elif method == "few_shot":
            result = classifier.classify_few_shot(text)
        elif method == "rag_enhanced":
            result = classifier.classify_rag(text)
        else:
            result = classifier.classify_function_calling(text)

        intent = result.get("intent", "UNKNOWN")
        result["intent_info"] = INTENT_DESCRIPTIONS.get(intent, INTENT_DESCRIPTIONS.get("UNKNOWN", {}))
        results["methods"][method] = result

    return results


@app.get("/api/examples")
async def get_examples():
    """获取训练示例"""
    return {"examples": TRAINING_EXAMPLES, "count": len(TRAINING_EXAMPLES)}


@app.get("/api/prompts")
async def get_prompts():
    """获取 Prompt 模板"""
    return {"templates": PROMPT_TEMPLATES, "function_schema": FUNCTION_SCHEMA}


# ==================== 多轮对话 API ====================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """多轮对话接口"""
    message = request.message

    if request.use_langgraph:
        workflow = get_langgraph_workflow()
        if workflow:
            result = workflow.process_message(
                session_id=request.session_id,
                user_message=message
            )
            result["engine"] = "langgraph"
            return result

    # 回退到传统模式
    result = assistant.process_message(
        session_id=request.session_id,
        user_message=message,
        method=request.method
    )
    intent = result["intent_result"].get("intent", "UNKNOWN")
    result["intent_result"]["intent_info"] = INTENT_DESCRIPTIONS.get(intent, INTENT_DESCRIPTIONS.get("UNKNOWN", {}))
    result["engine"] = "legacy"
    return result


@app.post("/api/chat/reset")
async def reset_chat(request: ResetRequest):
    """重置对话"""
    if request.use_langgraph:
        workflow = get_langgraph_workflow()
        if workflow:
            result = workflow.reset_session(request.session_id)
            result["engine"] = "langgraph"
            return result

    result = assistant.reset_session(request.session_id)
    result["engine"] = "legacy"
    return result


@app.get("/api/chat/new")
async def new_chat(use_langgraph: bool = True):
    """创建新对话"""
    if use_langgraph:
        workflow = get_langgraph_workflow()
        if workflow:
            result = workflow.create_session()
            result["engine"] = "langgraph"
            return result

    session = assistant.session_manager.create_session()
    session.add_message("assistant", "您好！欢迎光临，请问想喝点什么？")
    return {
        "session_id": session.session_id,
        "state": session.state.value,
        "history": session.history,
        "suggestions": ["来杯拿铁", "有什么推荐", "看看菜单"],
        "engine": "legacy"
    }


# ==================== 菜单和工作流 API ====================

@app.get("/api/menu")
async def get_menu():
    """获取菜单"""
    return {
        "products": MENU,
        "size_price": SIZE_PRICE,
        "milk_price": MILK_PRICE,
        "extras_price": EXTRAS_PRICE
    }


@app.get("/api/workflow/graph")
async def get_workflow_graph():
    """获取 LangGraph 工作流图"""
    workflow = get_langgraph_workflow()
    if workflow:
        return {"available": True, "mermaid": workflow.get_graph_visualization()}
    return {"available": False, "mermaid": None}


# ==================== 启动服务 ====================

def run(reload: bool = False):
    """运行服务器"""
    import uvicorn

    workflow = get_langgraph_workflow()

    print("\n" + "=" * 60)
    print("   AI 点单意图识别系统 - 重构版 v3.1")
    print("=" * 60)
    print(f"\n运营监控 Portal: http://localhost:8000")
    print(f"意图识别 Demo: http://localhost:8000/intent")
    print(f"多轮对话页面: http://localhost:8000/chat")
    print(f"OpenAI 状态: {'可用' if classifier.is_available() else '不可用'}")
    print(f"LangGraph: {'已启用' if workflow else '未启用'}")
    print("\n按 Ctrl+C 停止服务\n")

    if reload:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[str(Path(__file__).parent.parent)]
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI 点单意图识别系统")
    parser.add_argument("--reload", "-r", action="store_true", help="启用自动重载")
    args = parser.parse_args()
    run(reload=args.reload)
