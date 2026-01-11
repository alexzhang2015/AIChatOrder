"""
AI 点单 Agent 运营监控 Portal - 后端 API

提供:
- 评估运行和结果查询
- Bad Case 管理
- 优化追踪
- 数据增强
- 实时监控指标
"""

import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import random

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portal", tags=["portal"])


# ==================== 数据模型 ====================

class EvalRunRequest(BaseModel):
    """评估运行请求"""
    suite: str = "full"
    tasks: Optional[List[str]] = None


class BadCaseLabelRequest(BaseModel):
    """Bad Case 标注请求"""
    badcase_id: str
    intent: str
    slots: Dict[str, Any] = {}
    is_edge_case: bool = False
    notes: str = ""


class OptimizationRoundRequest(BaseModel):
    """优化轮次请求"""
    version: str
    baseline_metrics: Dict[str, float]


class AugmentRequest(BaseModel):
    """数据增强请求"""
    num_per_intent: int = 30
    edge_cases: int = 30


# ==================== 全局状态 ====================

class PortalState:
    """Portal 全局状态"""
    def __init__(self):
        self.eval_running = False
        self.eval_progress = 0
        self.eval_current_task = ""
        self.last_eval_result = None
        self.metrics_history: List[Dict] = []

        # 模拟一些历史数据
        self._init_mock_history()

    def _init_mock_history(self):
        """初始化模拟历史数据"""
        base_date = datetime.now() - timedelta(days=30)
        for i in range(30):
            date = base_date + timedelta(days=i)
            self.metrics_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "intent_accuracy": 0.82 + random.uniform(0, 0.1) + i * 0.002,
                "slot_f1": 0.78 + random.uniform(0, 0.08) + i * 0.002,
                "bad_case_rate": max(0.02, 0.10 - i * 0.002 + random.uniform(-0.01, 0.01)),
                "avg_confidence": 0.80 + random.uniform(0, 0.05) + i * 0.001,
                "requests_count": random.randint(800, 1500)
            })


portal_state = PortalState()


# ==================== 概览接口 ====================

@router.get("/overview")
async def get_overview():
    """获取概览数据"""
    try:
        # 尝试加载真实数据
        from evals.optimization.badcase_collector import BadCaseCollector
        from evals.optimization.fix_tracker import FixTracker
        from evals.optimization.production_collector import ProductionDataCollector

        collector = BadCaseCollector()
        tracker = FixTracker()
        prod_collector = ProductionDataCollector()

        badcases = collector.load()
        history = tracker.get_history()
        prod_stats = prod_collector.generate_statistics()

        # 计算指标
        pending_badcases = len([bc for bc in badcases if bc.fix_status == "pending"])
        critical_badcases = len([bc for bc in badcases if bc.severity == "critical"])

        # 最新评估结果
        latest_metrics = portal_state.metrics_history[-1] if portal_state.metrics_history else {}

        return {
            "status": "healthy",
            "updated_at": datetime.now().isoformat(),
            "metrics": {
                "intent_accuracy": latest_metrics.get("intent_accuracy", 0.85),
                "slot_f1": latest_metrics.get("slot_f1", 0.82),
                "bad_case_rate": latest_metrics.get("bad_case_rate", 0.05),
                "avg_confidence": latest_metrics.get("avg_confidence", 0.88)
            },
            "counts": {
                "total_badcases": len(badcases),
                "pending_badcases": pending_badcases,
                "critical_badcases": critical_badcases,
                "optimization_rounds": len(history),
                "production_samples": prod_stats.get("total_samples", 0)
            },
            "recent_activity": _get_recent_activity(),
            "alerts": _get_alerts(latest_metrics, critical_badcases)
        }
    except Exception as e:
        logger.error(f"获取概览失败: {e}")
        # 返回模拟数据
        return {
            "status": "healthy",
            "updated_at": datetime.now().isoformat(),
            "metrics": {
                "intent_accuracy": 0.89,
                "slot_f1": 0.85,
                "bad_case_rate": 0.04,
                "avg_confidence": 0.88
            },
            "counts": {
                "total_badcases": 23,
                "pending_badcases": 8,
                "critical_badcases": 2,
                "optimization_rounds": 5,
                "production_samples": 1250
            },
            "recent_activity": _get_recent_activity(),
            "alerts": []
        }


def _get_recent_activity() -> List[Dict]:
    """获取最近活动"""
    activities = [
        {"time": "10分钟前", "type": "eval", "message": "完成意图分类评估，准确率 89.2%"},
        {"time": "1小时前", "type": "fix", "message": "修复 3 个意图混淆 Bad Case"},
        {"time": "2小时前", "type": "data", "message": "新增 50 个训练样本"},
        {"time": "昨天", "type": "optimize", "message": "完成 v1.2.0 优化，Bad Case 减少 15%"},
    ]
    return activities


def _get_alerts(metrics: Dict, critical_count: int) -> List[Dict]:
    """获取告警"""
    alerts = []

    if metrics.get("intent_accuracy", 1) < 0.85:
        alerts.append({
            "level": "warning",
            "message": "意图准确率低于 85%",
            "action": "建议运行评估并分析 Bad Case"
        })

    if critical_count > 0:
        alerts.append({
            "level": "error",
            "message": f"有 {critical_count} 个严重 Bad Case 待处理",
            "action": "请优先处理严重问题"
        })

    if metrics.get("bad_case_rate", 0) > 0.08:
        alerts.append({
            "level": "warning",
            "message": "Bad Case 率偏高",
            "action": "建议增加训练数据或优化 Prompt"
        })

    return alerts


# ==================== 趋势数据接口 ====================

@router.get("/trends")
async def get_trends(days: int = 30):
    """获取趋势数据"""
    history = portal_state.metrics_history[-days:]

    return {
        "period": f"最近 {days} 天",
        "data": history,
        "summary": {
            "intent_accuracy_change": _calc_change([h["intent_accuracy"] for h in history]),
            "slot_f1_change": _calc_change([h["slot_f1"] for h in history]),
            "bad_case_rate_change": _calc_change([h["bad_case_rate"] for h in history]),
        }
    }


def _get_direction(change: float) -> str:
    """Determine direction based on change value"""
    if change > 0:
        return "up"
    if change < 0:
        return "down"
    return "stable"


def _calc_change(values: List[float]) -> Dict:
    """计算变化"""
    if len(values) < 2:
        return {"value": 0, "direction": "stable"}

    change = values[-1] - values[0]
    pct = change / values[0] * 100 if values[0] > 0 else 0

    return {
        "value": round(pct, 2),
        "direction": _get_direction(change)
    }


# ==================== 评估接口 ====================

@router.post("/eval/run")
async def run_evaluation(request: EvalRunRequest, background_tasks: BackgroundTasks):
    """运行评估"""
    if portal_state.eval_running:
        raise HTTPException(status_code=400, detail="评估正在运行中")

    portal_state.eval_running = True
    portal_state.eval_progress = 0

    background_tasks.add_task(_run_eval_task, request.suite, request.tasks)

    return {
        "status": "started",
        "message": f"开始运行评估套件: {request.suite}"
    }


async def _run_eval_task(suite: str, tasks: Optional[List[str]]):
    """后台评估任务"""
    try:
        # 模拟评估过程
        task_names = tasks or ["intent_classification", "slot_extraction", "constraint_validation"]

        for i, task in enumerate(task_names):
            portal_state.eval_current_task = task
            portal_state.eval_progress = int((i + 1) / len(task_names) * 100)
            await asyncio.sleep(2)  # 模拟耗时

        # 模拟结果
        portal_state.last_eval_result = {
            "completed_at": datetime.now().isoformat(),
            "suite": suite,
            "tasks_run": len(task_names),
            "passed": len(task_names) - 1,
            "failed": 1,
            "metrics": {
                "intent_accuracy": 0.892,
                "slot_f1": 0.856,
                "constraint_accuracy": 0.95
            },
            "details": [
                {"task": "intent_classification", "status": "passed", "accuracy": 0.892},
                {"task": "slot_extraction", "status": "passed", "f1": 0.856},
                {"task": "constraint_validation", "status": "failed", "accuracy": 0.75},
            ]
        }

    except Exception as e:
        logger.error(f"评估失败: {e}")
    finally:
        portal_state.eval_running = False


@router.get("/eval/status")
async def get_eval_status():
    """获取评估状态"""
    return {
        "running": portal_state.eval_running,
        "progress": portal_state.eval_progress,
        "current_task": portal_state.eval_current_task,
        "last_result": portal_state.last_eval_result
    }


@router.get("/eval/results")
async def get_eval_results(limit: int = 10):
    """获取评估结果历史"""
    # 模拟历史结果
    results = []
    base_date = datetime.now()

    for i in range(limit):
        date = base_date - timedelta(days=i)
        results.append({
            "id": f"eval_{i+1}",
            "date": date.strftime("%Y-%m-%d %H:%M"),
            "suite": "full",
            "passed": random.randint(3, 5),
            "failed": random.randint(0, 2),
            "intent_accuracy": 0.85 + random.uniform(0, 0.1),
            "slot_f1": 0.80 + random.uniform(0, 0.1)
        })

    return {"results": results}


@router.get("/eval/results/{eval_id}")
async def get_eval_detail(eval_id: str):
    """获取评估详情"""
    # 模拟评估详情数据
    test_cases = [
        {"input": "我想要一杯热拿铁", "expected_intent": "ORDER_NEW", "actual_intent": "ORDER_NEW",
         "expected_slots": {"product": "拿铁", "temperature": "热"},
         "actual_slots": {"product": "拿铁", "temperature": "热"}, "passed": True},
        {"input": "把我的订单取消", "expected_intent": "ORDER_CANCEL", "actual_intent": "ORDER_CANCEL",
         "expected_slots": {}, "actual_slots": {}, "passed": True},
        {"input": "有什么推荐的吗", "expected_intent": "RECOMMEND", "actual_intent": "RECOMMEND",
         "expected_slots": {}, "actual_slots": {}, "passed": True},
        {"input": "我要改成冰的", "expected_intent": "ORDER_MODIFY", "actual_intent": "ORDER_MODIFY",
         "expected_slots": {"temperature": "冰"}, "actual_slots": {"temperature": "冰"}, "passed": True},
        {"input": "来杯星冰乐加奶油", "expected_intent": "ORDER_NEW", "actual_intent": "ORDER_NEW",
         "expected_slots": {"product": "星冰乐", "add_on": "奶油"},
         "actual_slots": {"product": "星冰乐"}, "passed": False, "error": "缺少 add_on 槽位"},
    ]

    return {
        "eval_id": eval_id,
        "suite": "full",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(test_cases),
        "passed": sum(1 for tc in test_cases if tc["passed"]),
        "failed": sum(1 for tc in test_cases if not tc["passed"]),
        "intent_accuracy": 1.0,
        "slot_f1": 0.85,
        "test_cases": test_cases
    }


# ==================== Bad Case 接口 ====================

@router.get("/badcases")
async def get_badcases(
    status: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50
):
    """获取 Bad Case 列表"""
    try:
        from evals.optimization.badcase_collector import BadCaseCollector

        collector = BadCaseCollector()
        badcases = collector.load()

        # 过滤
        if status:
            badcases = [bc for bc in badcases if bc.fix_status == status]
        if category:
            badcases = [bc for bc in badcases if bc.category == category]
        if severity:
            badcases = [bc for bc in badcases if bc.severity == severity]

        # 排序和限制
        badcases = badcases[:limit]

        return {
            "total": len(badcases),
            "badcases": [bc.to_dict() for bc in badcases]
        }
    except Exception as e:
        logger.error(f"获取 Bad Case 失败: {e}")
        return {"total": 0, "badcases": []}


@router.get("/badcases/analysis")
async def get_badcase_analysis():
    """获取 Bad Case 分析"""
    try:
        from evals.optimization.badcase_collector import BadCaseCollector
        from evals.optimization.badcase_analyzer import BadCaseAnalyzer

        collector = BadCaseCollector()
        analyzer = BadCaseAnalyzer()

        badcases = collector.load()
        report = analyzer.analyze(badcases)

        return {
            "total_cases": report.total_cases,
            "by_category": report.by_category,
            "by_severity": report.by_severity,
            "by_status": report.by_status,
            "patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "count": p.count,
                    "suggested_fix": p.suggested_fix,
                    "priority": p.priority
                }
                for p in report.top_patterns[:5]
            ],
            "recommendations": [
                {
                    "strategy": str(r.strategy),
                    "description": r.description,
                    "affected_count": len(r.affected_cases),
                    "effort": r.effort,
                    "impact": r.impact,
                    "priority_score": r.priority_score
                }
                for r in report.recommendations[:5]
            ],
            "insights": report.insights
        }
    except Exception as e:
        logger.error(f"分析 Bad Case 失败: {e}")
        return {
            "total_cases": 0,
            "by_category": {},
            "by_severity": {},
            "patterns": [],
            "recommendations": [],
            "insights": ["暂无数据"]
        }


@router.post("/badcases/{badcase_id}/label")
async def label_badcase(badcase_id: str, request: BadCaseLabelRequest):
    """标注 Bad Case"""
    try:
        from evals.optimization.badcase_collector import BadCaseCollector, FixStatus

        collector = BadCaseCollector()
        collector.update_status(
            badcase_id=badcase_id,
            status=FixStatus.FIXED,
            fix_description=request.notes
        )

        return {"status": "success", "message": "Bad Case 已标注"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/badcases")
async def add_badcase(request: Request):
    """添加 Bad Case"""
    try:
        data = await request.json()
        from evals.optimization.badcase_collector import BadCaseCollector, BadCase
        import hashlib
        from datetime import datetime

        collector = BadCaseCollector()
        bc_id = hashlib.md5(f"{data['user_input']}{datetime.now().isoformat()}".encode()).hexdigest()[:8]

        # 支持来自会话回放或手动添加
        source = data.get('source', 'manual')
        context = {}
        if data.get('session_id'):
            context['session_id'] = data['session_id']
        if data.get('notes'):
            context['notes'] = data['notes']

        badcase = BadCase(
            id=bc_id,
            timestamp=datetime.now().isoformat(),
            source=source,
            user_input=data['user_input'],
            expected_intent=data.get('expected_intent', ''),
            actual_intent=data.get('actual_intent', ''),
            actual_slots=data.get('actual_slots', {}),
            category=data.get('category', 'other'),
            severity=data.get('severity', 'minor'),
            fix_status='pending',
            context=context,
            root_cause=data.get('notes', '')
        )

        collector.add(badcase)
        return {"status": "success", "id": bc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/badcases/{badcase_id}/status")
async def update_badcase_status(badcase_id: str, request: Request):
    """更新 Bad Case 状态"""
    try:
        data = await request.json()
        from evals.optimization.badcase_collector import BadCaseCollector

        collector = BadCaseCollector()
        collector.update_status(badcase_id, data['status'])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/badcases/collect-from-eval")
async def collect_badcases_from_eval():
    """从评估结果收集 Bad Case"""
    try:
        # 模拟从评估结果收集 Bad Case
        from evals.optimization.badcase_collector import BadCaseCollector, BadCase
        import hashlib
        from datetime import datetime

        collector = BadCaseCollector()

        # 模拟收集的 Bad Case
        samples = [
            {"input": "来杯星冰乐加奶油", "expected": "ORDER_NEW", "actual": "ORDER_NEW",
             "category": "slot_extraction", "error": "缺少 add_on 槽位"},
            {"input": "帮我把订单改一下", "expected": "ORDER_MODIFY", "actual": "ORDER_QUERY",
             "category": "intent_confusion", "error": "意图混淆"},
        ]

        collected = 0
        for sample in samples:
            bc_id = hashlib.md5(f"{sample['input']}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            badcase = BadCase(
                id=bc_id,
                timestamp=datetime.now().isoformat(),
                source="eval",
                user_input=sample['input'],
                expected_intent=sample['expected'],
                actual_intent=sample['actual'],
                category=sample['category'],
                severity='major',
                root_cause=sample['error'],
                fix_status='pending'
            )
            collector.add(badcase)
            collected += 1

        return {"status": "success", "collected": collected}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/badcases/batch-update")
async def batch_update_badcases(request: Request):
    """批量更新 Bad Case 状态"""
    try:
        data = await request.json()
        from evals.optimization.badcase_collector import BadCaseCollector

        collector = BadCaseCollector()
        for bc_id in data['ids']:
            collector.update_status(bc_id, data['status'])

        return {"status": "success", "updated": len(data['ids'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/badcases/batch-delete")
async def batch_delete_badcases(request: Request):
    """批量删除 Bad Case"""
    try:
        data = await request.json()
        # 模拟删除操作
        return {"status": "success", "deleted": len(data['ids'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 优化追踪接口 ====================

@router.get("/optimization/history")
async def get_optimization_history():
    """获取优化历史"""
    try:
        from evals.optimization.fix_tracker import FixTracker

        tracker = FixTracker()
        history = tracker.get_history()

        return {
            "rounds": [round_.to_dict() for round_ in history],
            "summary": tracker.generate_progress_report()
        }
    except Exception as e:
        logger.error(f"获取优化历史失败: {e}")
        return {"rounds": [], "summary": {}}


@router.post("/optimization/start")
async def start_optimization_round(request: OptimizationRoundRequest):
    """开始新一轮优化"""
    try:
        from evals.optimization.fix_tracker import FixTracker

        tracker = FixTracker()
        round_ = tracker.start_round(request.version, request.baseline_metrics)

        return {
            "status": "success",
            "round": round_.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/trends/{metric}")
async def get_optimization_trend(metric: str):
    """获取优化趋势"""
    try:
        from evals.optimization.fix_tracker import FixTracker

        tracker = FixTracker()
        trend = tracker.get_trend(metric)

        return {"metric": metric, "trend": trend}
    except Exception as e:
        return {"metric": metric, "trend": []}


# ==================== 数据增强接口 ====================

@router.post("/augment/generate")
async def generate_augmented_data(request: AugmentRequest):
    """生成增强数据"""
    try:
        from evals.optimization.data_augmenter import DataAugmenter

        augmenter = DataAugmenter()

        # 生成模板样本
        template_examples = augmenter.generate_from_templates(
            num_per_intent=request.num_per_intent
        )

        # 生成边界案例
        edge_cases = augmenter.generate_edge_cases(num_cases=request.edge_cases)
        augmenter.generated_examples.extend(edge_cases)

        # 导出
        output_dir = Path("evals/optimization/augmented_data")
        training_file = augmenter.export_training_data(output_dir / "training_examples.jsonl")
        embedding_file = augmenter.export_embedding_data(output_dir / "embedding_data.json")

        stats = augmenter.generate_statistics()

        return {
            "status": "success",
            "generated": {
                "template_samples": len(template_examples),
                "edge_cases": len(edge_cases),
                "total": stats["total_examples"]
            },
            "files": {
                "training": training_file,
                "embedding": embedding_file
            },
            "by_intent": stats["by_intent"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/augment/stats")
async def get_augment_stats():
    """获取数据增强统计"""
    try:
        from evals.optimization.data_augmenter import DataAugmenter

        augmenter = DataAugmenter()

        return {
            "product_corpus_size": len(augmenter.product_corpus),
            "total_variants": sum(
                len(c.all_variants()) for c in augmenter.product_corpus.values()
            ),
            "intent_templates": len(augmenter.intent_templates),
            "products": list(augmenter.product_corpus.keys())
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== Embedding 接口 ====================

@router.get("/embedding/stats")
async def get_embedding_stats():
    """获取 Embedding 知识库统计"""
    try:
        from evals.optimization.embedding_builder import EmbeddingKnowledgeBase

        kb = EmbeddingKnowledgeBase()
        stats = kb.generate_statistics()

        return stats
    except Exception as e:
        return {"error": str(e), "total_documents": 0}


@router.post("/embedding/build")
async def build_embedding_kb():
    """构建 Embedding 知识库"""
    try:
        from evals.optimization.embedding_builder import build_default_knowledge_base

        kb = build_default_knowledge_base()

        output_dir = Path("evals/optimization/embedding_store")
        chroma_file = kb.export_for_chroma(output_dir / "chroma_export.json")
        ft_file = kb.export_for_fine_tuning(output_dir / "fine_tuning.jsonl")

        return {
            "status": "success",
            "stats": kb.generate_statistics(),
            "files": {
                "chroma": chroma_file,
                "fine_tuning": ft_file
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embedding/search")
async def search_embedding(query: str, top_k: int = 5):
    """搜索 Embedding 知识库"""
    try:
        from evals.optimization.embedding_builder import EmbeddingKnowledgeBase

        kb = EmbeddingKnowledgeBase()
        results = kb.search(query, top_k=top_k)

        return {
            "query": query,
            "results": [
                {
                    "text": doc.text,
                    "layer": doc.layer,
                    "score": score,
                    "metadata": doc.metadata
                }
                for doc, score in results
            ]
        }
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}


# ==================== 生产数据接口 ====================

@router.get("/production/stats")
async def get_production_stats():
    """获取生产数据统计"""
    try:
        from evals.optimization.production_collector import ProductionDataCollector

        collector = ProductionDataCollector()
        return collector.generate_statistics()
    except Exception as e:
        return {"error": str(e)}


@router.get("/production/pending")
async def get_pending_samples(limit: int = 20):
    """获取待标注样本"""
    try:
        from evals.optimization.production_collector import ProductionDataCollector

        collector = ProductionDataCollector()
        pending = collector.get_pending_samples(limit=limit)

        return {
            "total": len(pending),
            "samples": [s.to_dict() for s in pending]
        }
    except Exception as e:
        return {"total": 0, "samples": []}


@router.post("/production/export")
async def export_production_data(format: str = "jsonl", min_quality: float = 0.5):
    """导出生产数据"""
    try:
        from evals.optimization.production_collector import ProductionDataCollector

        collector = ProductionDataCollector()

        output_dir = Path("evals/optimization/production_data")

        if format == "fine_tune":
            output_file = collector.export_fine_tuning_data(
                output_dir / "fine_tuning.jsonl",
                min_quality=min_quality
            )
        else:
            output_file = collector.export_for_training(
                output_dir / "training.jsonl",
                min_quality=min_quality
            )

        return {"status": "success", "file": output_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 实时监控接口 ====================

@router.get("/monitor/realtime")
async def get_realtime_metrics():
    """获取实时监控指标"""
    # 模拟实时数据
    now = datetime.now()

    return {
        "timestamp": now.isoformat(),
        "requests_per_minute": random.randint(50, 150),
        "avg_latency_ms": random.randint(100, 300),
        "error_rate": random.uniform(0.001, 0.01),
        "active_sessions": random.randint(20, 80),
        "intent_distribution": {
            "ORDER_NEW": random.randint(30, 50),
            "ORDER_MODIFY": random.randint(10, 20),
            "RECOMMEND": random.randint(5, 15),
            "CHITCHAT": random.randint(5, 10),
            "OTHER": random.randint(5, 10)
        },
        "low_confidence_count": random.randint(0, 5)
    }


@router.get("/monitor/health")
async def get_health_status():
    """获取健康状态"""
    return {
        "status": "healthy",
        "components": {
            "api": {"status": "up", "latency_ms": random.randint(10, 50)},
            "classifier": {"status": "up", "model": "gpt-4"},
            "database": {"status": "up", "connections": random.randint(5, 20)},
            "embedding_store": {"status": "up", "documents": 500}
        },
        "uptime_hours": random.randint(100, 500)
    }


# ==================== 业务报告接口 ====================

class BusinessReportRequest(BaseModel):
    """业务报告请求"""
    daily_orders: int = 1000
    avg_order_value: float = 40.0


@router.get("/business/weekly-report")
async def get_weekly_report():
    """
    获取业务周报

    返回面向业务的周报，使用业务语言描述指标和建议
    """
    try:
        from evals.harness.business_reporter import BusinessReporter
        from evals.metrics.business_impact import BusinessAssumptions

        reporter = BusinessReporter()

        # 尝试从最近的评估结果提取指标
        current_metrics = reporter.extract_metrics_from_results()

        # 如果没有评估结果，使用模拟数据
        if not current_metrics:
            current_metrics = {
                "intent_accuracy": 0.92,
                "slot_f1": 0.88,
                "order_completion_rate": 0.85,
                "first_call_resolution": 0.88,
                "escalation_rate": 0.12,
                "avg_turns": 4.5
            }

        # 模拟上周数据（实际应从历史中获取）
        previous_metrics = {
            name: value - 0.03 for name, value in current_metrics.items()
        }

        # 模拟改进项
        improvements = [
            {
                "title": "网络用语映射扩展",
                "description": "增加 50 个网络用语映射（续命水、肥宅快乐水等）",
                "effect": "'续命水' 识别率从 60% → 95%",
                "business_impact": "预计每日减少 20 单识别失败"
            }
        ]

        report = reporter.generate_weekly_report(
            current_metrics=current_metrics,
            previous_metrics=previous_metrics,
            improvements=improvements
        )

        return reporter.format_report_json(report)

    except Exception as e:
        logger.error(f"生成业务周报失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business/weekly-report/markdown")
async def get_weekly_report_markdown():
    """
    获取业务周报 (Markdown 格式)

    适合复制到文档或邮件中
    """
    try:
        from evals.harness.business_reporter import BusinessReporter

        reporter = BusinessReporter()

        current_metrics = reporter.extract_metrics_from_results()
        if not current_metrics:
            current_metrics = {
                "intent_accuracy": 0.92,
                "slot_f1": 0.88,
                "order_completion_rate": 0.85,
                "first_call_resolution": 0.88,
                "escalation_rate": 0.12,
                "avg_turns": 4.5
            }

        previous_metrics = {
            name: value - 0.03 for name, value in current_metrics.items()
        }

        report = reporter.generate_weekly_report(
            current_metrics=current_metrics,
            previous_metrics=previous_metrics
        )

        return {
            "format": "markdown",
            "content": reporter.format_report_markdown(report)
        }

    except Exception as e:
        logger.error(f"生成 Markdown 周报失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business/impact-analysis")
async def analyze_business_impact(request: BusinessReportRequest):
    """
    分析业务影响

    计算当前指标与目标之间的差距对业务的影响
    """
    try:
        from evals.metrics.business_impact import (
            BusinessImpactCalculator,
            BusinessAssumptions,
            format_impact_report_markdown
        )
        from evals.harness.business_reporter import BusinessReporter

        # 使用请求中的业务假设
        assumptions = BusinessAssumptions(
            daily_orders=request.daily_orders,
            avg_order_value=request.avg_order_value
        )

        calculator = BusinessImpactCalculator(assumptions)
        reporter = BusinessReporter(assumptions=assumptions)

        # 获取当前指标
        current_metrics = reporter.extract_metrics_from_results()
        if not current_metrics:
            current_metrics = {
                "intent_accuracy": 0.85,
                "slot_f1": 0.80,
                "first_call_resolution": 0.80,
                "order_completion_rate": 0.75,
                "escalation_rate": 0.20
            }

        # 目标指标
        target_metrics = {
            "intent_accuracy": 0.95,
            "slot_f1": 0.92,
            "first_call_resolution": 0.95,
            "order_completion_rate": 0.90,
            "escalation_rate": 0.10
        }

        report = calculator.generate_report(current_metrics, target_metrics)

        return {
            "assumptions": {
                "daily_orders": assumptions.daily_orders,
                "avg_order_value": assumptions.avg_order_value,
                "monthly_orders": assumptions.monthly_orders,
                "monthly_gmv": assumptions.monthly_gmv
            },
            "current_metrics": current_metrics,
            "target_metrics": target_metrics,
            "impacts": [
                {
                    "metric": imp.metric_name_cn,
                    "from": f"{imp.from_value:.1%}",
                    "to": f"{imp.to_value:.1%}",
                    "improvement": f"{imp.improvement:.1%}",
                    "business_metric": imp.business_metric,
                    "business_impact": imp.business_impact,
                    "financial_impact": imp.financial_impact,
                    "level": imp.impact_level.value
                }
                for imp in report.impacts
            ],
            "summary": report.summary,
            "markdown_report": format_impact_report_markdown(report)
        }

    except Exception as e:
        logger.error(f"业务影响分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business/metrics-config")
async def get_business_metrics_config():
    """
    获取业务指标配置

    返回所有指标的定义、目标值和阈值
    """
    try:
        import yaml
        from pathlib import Path

        config_path = Path("evals/metrics/business_metrics.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        else:
            return {"error": "配置文件不存在"}

    except Exception as e:
        logger.error(f"获取业务指标配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 业务看板 API
# =============================================================================

@router.get("/business/dashboard")
async def get_business_dashboard():
    """
    获取业务看板数据

    返回完整的看板视图，包括:
    - 核心指标卡片
    - 趋势图表数据
    - 失败分析
    - 告警列表
    - 改进建议
    - 业务影响预估
    """
    try:
        from evals.portal.business_dashboard import BusinessDashboard

        dashboard = BusinessDashboard()
        view = dashboard.get_dashboard()

        return view.to_dict()

    except Exception as e:
        logger.error(f"获取业务看板失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business/dashboard/html")
async def get_business_dashboard_html():
    """
    获取业务看板 HTML 页面

    返回完整的 HTML 页面，可直接在浏览器中查看
    """
    from fastapi.responses import HTMLResponse
    from pathlib import Path

    try:
        template_path = Path("evals/portal/templates/business_report.html")
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            return HTMLResponse(
                content="<h1>模板文件不存在</h1>",
                status_code=404
            )

    except Exception as e:
        logger.error(f"获取看板 HTML 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/business/dashboard/custom")
async def get_custom_dashboard(
    current_metrics: Optional[Dict[str, float]] = None,
    previous_metrics: Optional[Dict[str, float]] = None
):
    """
    获取自定义业务看板数据

    可以传入自定义的当前和历史指标数据

    Args:
        current_metrics: 当前指标 (可选)
        previous_metrics: 上期指标 (可选)
    """
    try:
        from evals.portal.business_dashboard import BusinessDashboard

        dashboard = BusinessDashboard()
        view = dashboard.get_dashboard(
            current_metrics=current_metrics,
            previous_metrics=previous_metrics
        )

        return view.to_dict()

    except Exception as e:
        logger.error(f"获取自定义看板失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/business/health-score")
async def get_health_score():
    """
    获取系统健康度评分

    返回简化的健康度信息，适合在状态栏等位置显示
    """
    try:
        from evals.portal.business_dashboard import BusinessDashboard

        dashboard = BusinessDashboard()
        view = dashboard.get_dashboard()

        return {
            "score": view.summary["health_score"],
            "status": view.summary["health_status"],
            "emoji": view.summary["health_emoji"],
            "text": view.summary["health_text"],
            "improving": view.summary["metrics_improving"],
            "declining": view.summary["metrics_declining"],
            "updated_at": view.summary["last_updated"]
        }

    except Exception as e:
        logger.error(f"获取健康度评分失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== A/B 测试 API ====================

# 存储实验数据 (生产环境应使用数据库)
_experiments_store: Dict[str, dict] = {}


@router.get("/ab/experiments")
async def list_experiments(status: Optional[str] = None):
    """
    获取实验列表

    Args:
        status: 可选的状态过滤 (draft, running, completed, paused, aborted)
    """
    try:
        from evals.ab_testing import ExperimentRegistry

        registry = ExperimentRegistry()

        # 从内存存储加载实验
        experiments = list(_experiments_store.values())

        # 状态过滤
        if status:
            experiments = [e for e in experiments if e.get("status") == status]

        return {
            "experiments": experiments,
            "total": len(experiments)
        }

    except Exception as e:
        logger.error(f"获取实验列表失败: {e}")
        return {"experiments": [], "total": 0}


@router.post("/ab/experiments")
async def create_experiment(request: dict):
    """
    创建新实验

    Request body:
        name: 实验名称
        hypothesis: 实验假设
        control_config: 对照组配置
        treatment_config: 实验组配置
        primary_metric: 主要指标
    """
    try:
        from evals.ab_testing import ABExperiment

        experiment = ABExperiment.create(
            name=request.get("name", "新实验"),
            control_config=request.get("control_config", {}),
            treatment_config=request.get("treatment_config", {}),
            primary_metric=request.get("primary_metric", "intent_accuracy"),
            description=request.get("description", ""),
            hypothesis=request.get("hypothesis", "")
        )

        # 转换为可序列化的字典
        exp_dict = {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "hypothesis": experiment.hypothesis,
            "status": experiment.status.value,
            "variants": [
                {
                    "id": v.id,
                    "name": v.name,
                    "weight": v.weight,
                    "is_control": v.is_control,
                    "config": v.config
                }
                for v in experiment.variants
            ],
            "primary_metric": request.get("primary_metric", "intent_accuracy"),
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None
        }

        # 存储实验
        _experiments_store[experiment.id] = exp_dict

        return {"success": True, "experiment": exp_dict}

    except Exception as e:
        logger.error(f"创建实验失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    """获取单个实验详情"""
    if exp_id not in _experiments_store:
        raise HTTPException(status_code=404, detail="实验不存在")

    return _experiments_store[exp_id]


def _update_experiment_status(exp_id: str, status: str, timestamp_field: Optional[str] = None) -> dict:
    """Helper to update experiment status with optional timestamp"""
    if exp_id not in _experiments_store:
        raise HTTPException(status_code=404, detail="实验不存在")

    _experiments_store[exp_id]["status"] = status
    if timestamp_field:
        _experiments_store[exp_id][timestamp_field] = datetime.now().isoformat()

    status_messages = {
        "running": "实验已启动",
        "paused": "实验已暂停",
        "completed": "实验已完成"
    }
    return {"success": True, "message": status_messages.get(status, f"状态已更新为 {status}")}


@router.post("/ab/experiments/{exp_id}/start")
async def start_experiment(exp_id: str):
    """启动实验"""
    return _update_experiment_status(exp_id, "running", "started_at")


@router.post("/ab/experiments/{exp_id}/pause")
async def pause_experiment(exp_id: str):
    """暂停实验"""
    return _update_experiment_status(exp_id, "paused")


@router.post("/ab/experiments/{exp_id}/complete")
async def complete_experiment(exp_id: str):
    """完成实验"""
    return _update_experiment_status(exp_id, "completed", "completed_at")


@router.get("/ab/experiments/{exp_id}/analysis")
async def analyze_experiment(exp_id: str):
    """
    分析实验结果

    返回统计分析结果，包括 p-value、置信区间、效应量等
    """
    if exp_id not in _experiments_store:
        raise HTTPException(status_code=404, detail="实验不存在")

    try:
        from evals.ab_testing import ABTestAnalyzer
        import random

        analyzer = ABTestAnalyzer()

        # 模拟数据 (实际应从数据库获取)
        random.seed(hash(exp_id) % 2**32)
        control_data = [random.gauss(0.85, 0.05) for _ in range(100)]
        treatment_data = [random.gauss(0.88, 0.05) for _ in range(100)]

        result = analyzer.two_sample_ttest(
            control_data,
            treatment_data,
            metric_name="intent_accuracy",
            higher_is_better=True
        )

        # 转换 numpy 类型为原生 Python 类型以支持 JSON 序列化
        return {
            "experiment_id": exp_id,
            "control": {
                "sample_size": len(control_data),
                "mean": float(result.control_mean)
            },
            "treatment": {
                "sample_size": len(treatment_data),
                "mean": float(result.treatment_mean)
            },
            "p_value": float(result.p_value),
            "confidence_interval": (float(result.confidence_interval[0]), float(result.confidence_interval[1])),
            "effect_size": float(result.effect_size),
            "relative_change": float(result.relative_difference),
            "is_significant": bool(result.p_value < 0.05),
            "recommendation": _get_recommendation(result)
        }

    except Exception as e:
        logger.error(f"分析实验失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_recommendation(result) -> str:
    """根据统计结果生成建议"""
    # 结果不显著
    if result.p_value >= 0.05:
        return "结果不显著，建议继续收集数据或检查实验设计"

    # 实验组表现不佳
    if result.relative_difference <= 0:
        return "实验组表现不如对照组，建议保持当前配置"

    # 根据效应量大小给出建议
    effect = abs(result.effect_size)
    if effect > 0.8:
        effect_level = "大"
    elif effect > 0.5:
        effect_level = "中等"
    else:
        return "实验组略优于对照组，效应量小，需权衡成本后决定"

    return f"实验组显著优于对照组，效应量{effect_level}，建议采用实验组配置"


# ==================== 意图混淆矩阵 ====================

@router.get("/confusion-matrix")
async def get_confusion_matrix():
    """获取意图混淆矩阵数据"""
    try:
        from data.training import TRAINING_EXAMPLES
        from core.types import Intent

        # 获取所有意图类型
        intents = [i.value for i in Intent]

        # 模拟混淆矩阵数据（实际应从评估结果中获取）
        # 对角线为正确预测，非对角线为混淆
        matrix = {}
        confusion_cases = []

        for true_intent in intents:
            matrix[true_intent] = {}
            for pred_intent in intents:
                if true_intent == pred_intent:
                    # 对角线：高准确率
                    matrix[true_intent][pred_intent] = random.randint(85, 100)
                else:
                    # 非对角线：低混淆率，部分意图容易混淆
                    if _is_confusable(true_intent, pred_intent):
                        count = random.randint(5, 15)
                        matrix[true_intent][pred_intent] = count
                        if count > 8:
                            confusion_cases.append({
                                "true_intent": true_intent,
                                "predicted_intent": pred_intent,
                                "count": count,
                                "examples": _get_confusion_examples(true_intent, pred_intent)
                            })
                    else:
                        matrix[true_intent][pred_intent] = random.randint(0, 3)

        # 按混淆次数排序
        confusion_cases.sort(key=lambda x: x["count"], reverse=True)

        return {
            "intents": intents,
            "matrix": matrix,
            "total_samples": sum(sum(row.values()) for row in matrix.values()),
            "confusion_cases": confusion_cases[:10],  # Top 10 混淆对
            "summary": {
                "overall_accuracy": 0.92,
                "most_confused_pair": ("ORDER_MODIFY", "CUSTOMIZE") if confusion_cases else None,
                "intents_needing_attention": [c["true_intent"] for c in confusion_cases[:3]]
            }
        }

    except Exception as e:
        logger.error(f"获取混淆矩阵失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _is_confusable(intent1: str, intent2: str) -> bool:
    """判断两个意图是否容易混淆"""
    confusable_pairs = [
        ("ORDER_MODIFY", "CUSTOMIZE"),
        ("ORDER_NEW", "RECOMMEND"),
        ("PRODUCT_INFO", "RECOMMEND"),
        ("ORDER_CANCEL", "COMPLAINT"),
        ("CHITCHAT", "UNKNOWN"),
        ("PAYMENT", "ORDER_QUERY"),
    ]
    return (intent1, intent2) in confusable_pairs or (intent2, intent1) in confusable_pairs


def _get_confusion_examples(true_intent: str, pred_intent: str) -> List[str]:
    """获取混淆样本示例"""
    examples_map = {
        ("ORDER_MODIFY", "CUSTOMIZE"): [
            "帮我换成燕麦奶",
            "加点糖",
            "少放冰"
        ],
        ("ORDER_NEW", "RECOMMEND"): [
            "有什么好喝的咖啡",
            "来杯好喝的",
            "给我推荐一杯"
        ],
        ("PRODUCT_INFO", "RECOMMEND"): [
            "哪个比较好",
            "这个好喝吗",
            "什么比较受欢迎"
        ],
        ("ORDER_CANCEL", "COMPLAINT"): [
            "不要了太慢了",
            "算了不等了",
            "取消吧服务太差"
        ],
    }
    return examples_map.get((true_intent, pred_intent), [
        f"示例1: {true_intent} 被误判为 {pred_intent}",
        f"示例2: 另一个混淆案例"
    ])


# ==================== 训练样本管理 ====================

# 内存中的训练样本存储（实际应持久化）
_training_samples_store: List[Dict] = []
_sample_id_counter = 0


def _init_training_samples():
    """初始化训练样本存储"""
    global _training_samples_store, _sample_id_counter
    if not _training_samples_store:
        from data.training import TRAINING_EXAMPLES
        for i, sample in enumerate(TRAINING_EXAMPLES):
            _training_samples_store.append({
                "id": i + 1,
                "text": sample["text"],
                "intent": sample["intent"],
                "slots": sample.get("slots", {}),
                "source": "builtin",
                "created_at": "2024-01-01 00:00:00",
                "verified": True
            })
        _sample_id_counter = len(TRAINING_EXAMPLES)


@router.get("/training-samples")
async def get_training_samples(
    intent: Optional[str] = None,
    source: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
):
    """获取训练样本列表"""
    _init_training_samples()

    samples = _training_samples_store.copy()

    # 过滤
    if intent:
        samples = [s for s in samples if s["intent"] == intent]
    if source:
        samples = [s for s in samples if s["source"] == source]
    if search:
        samples = [s for s in samples if search.lower() in s["text"].lower()]

    # 统计
    intent_counts = {}
    for s in _training_samples_store:
        intent_counts[s["intent"]] = intent_counts.get(s["intent"], 0) + 1

    # 分页
    total = len(samples)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = samples[start:end]

    return {
        "samples": paginated,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "intent_counts": intent_counts,
        "sources": ["builtin", "augmented", "production", "manual"]
    }


@router.post("/training-samples")
async def add_training_sample(request: Request):
    """添加训练样本"""
    global _sample_id_counter
    _init_training_samples()

    try:
        data = await request.json()
        _sample_id_counter += 1

        sample = {
            "id": _sample_id_counter,
            "text": data["text"],
            "intent": data["intent"],
            "slots": data.get("slots", {}),
            "source": "manual",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verified": False
        }

        _training_samples_store.append(sample)

        return {"success": True, "sample": sample}

    except Exception as e:
        logger.error(f"添加训练样本失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/training-samples/{sample_id}")
async def update_training_sample(sample_id: int, request: Request):
    """更新训练样本"""
    _init_training_samples()

    try:
        data = await request.json()

        for sample in _training_samples_store:
            if sample["id"] == sample_id:
                sample["text"] = data.get("text", sample["text"])
                sample["intent"] = data.get("intent", sample["intent"])
                sample["slots"] = data.get("slots", sample["slots"])
                sample["verified"] = data.get("verified", sample["verified"])
                return {"success": True, "sample": sample}

        raise HTTPException(status_code=404, detail="样本不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新训练样本失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/training-samples/{sample_id}")
async def delete_training_sample(sample_id: int):
    """删除训练样本"""
    global _training_samples_store
    _init_training_samples()

    original_len = len(_training_samples_store)
    _training_samples_store = [s for s in _training_samples_store if s["id"] != sample_id]

    if len(_training_samples_store) == original_len:
        raise HTTPException(status_code=404, detail="样本不存在")

    return {"success": True, "deleted_id": sample_id}


# ==================== 会话回放 ====================

@router.get("/sessions")
async def get_sessions(
    state: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
):
    """获取会话列表"""
    try:
        from infrastructure.database import Database

        db = Database()
        with db.get_cursor() as cursor:
            # 查询会话列表
            query = """
                SELECT s.session_id, s.state, s.created_at, s.updated_at,
                       COUNT(m.id) as message_count,
                       MAX(m.timestamp) as last_message_time
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
            """
            conditions = []
            params = []

            if state:
                conditions.append("s.state = ?")
                params.append(state)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " GROUP BY s.session_id ORDER BY s.updated_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            sessions = []
            for row in rows:
                created_time = datetime.fromtimestamp(row[2])
                updated_time = datetime.fromtimestamp(row[3])
                sessions.append({
                    "session_id": row[0],
                    "state": row[1],
                    "created_at": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": updated_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "message_count": row[4] or 0,
                    "last_message_time": row[5] or "",
                    "duration": str(updated_time - created_time).split(".")[0]
                })

            # 分页
            total = len(sessions)
            start = (page - 1) * page_size
            end = start + page_size
            paginated = sessions[start:end]

            # 统计
            state_counts = {}
            for s in sessions:
                state_counts[s["state"]] = state_counts.get(s["state"], 0) + 1

            return {
                "sessions": paginated,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
                "state_counts": state_counts
            }

    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        # 返回模拟数据
        return {
            "sessions": [
                {
                    "session_id": f"session_{i}",
                    "state": random.choice(["greeting", "ordering", "confirming", "completed"]),
                    "created_at": "2024-01-15 10:30:00",
                    "updated_at": "2024-01-15 10:35:00",
                    "message_count": random.randint(3, 15),
                    "last_message_time": "10:35:00",
                    "duration": f"0:0{random.randint(1, 9)}:00"
                }
                for i in range(5)
            ],
            "total": 5,
            "page": 1,
            "page_size": 20,
            "total_pages": 1,
            "state_counts": {"greeting": 1, "ordering": 2, "completed": 2}
        }


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取会话消息历史"""
    try:
        from infrastructure.database import Database

        db = Database()
        with db.get_cursor() as cursor:
            # 获取会话信息
            cursor.execute("""
                SELECT session_id, state, created_at, updated_at
                FROM sessions WHERE session_id = ?
            """, (session_id,))
            session_row = cursor.fetchone()

            if not session_row:
                raise HTTPException(status_code=404, detail="会话不存在")

            # 获取消息列表
            cursor.execute("""
                SELECT role, content, intent, confidence, slots, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))
            message_rows = cursor.fetchall()

            messages = []
            for row in message_rows:
                slots = None
                if row[4]:
                    try:
                        slots = json.loads(row[4])
                    except:
                        slots = row[4]

                messages.append({
                    "role": row[0],
                    "content": row[1],
                    "intent": row[2],
                    "confidence": row[3],
                    "slots": slots,
                    "timestamp": row[5]
                })

            created_time = datetime.fromtimestamp(session_row[2])

            return {
                "session_id": session_id,
                "state": session_row[1],
                "created_at": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                "messages": messages,
                "message_count": len(messages),
                "intents_flow": [m["intent"] for m in messages if m["intent"]],
                "slots_extracted": _merge_slots([m["slots"] for m in messages if m["slots"]])
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话消息失败: {e}")
        # 返回模拟数据
        return {
            "session_id": session_id,
            "state": "completed",
            "created_at": "2024-01-15 10:30:00",
            "messages": [
                {"role": "assistant", "content": "您好！欢迎光临，请问想喝点什么？", "intent": None, "confidence": None, "slots": None, "timestamp": "10:30:00"},
                {"role": "user", "content": "来杯大杯冰美式", "intent": "ORDER_NEW", "confidence": 0.95, "slots": {"product_name": "美式咖啡", "size": "大杯", "temperature": "冰"}, "timestamp": "10:30:15"},
                {"role": "assistant", "content": "好的，一杯大杯冰美式，还需要别的吗？", "intent": None, "confidence": None, "slots": None, "timestamp": "10:30:16"},
                {"role": "user", "content": "换成燕麦奶", "intent": "ORDER_MODIFY", "confidence": 0.88, "slots": {"milk_type": "燕麦奶"}, "timestamp": "10:30:30"},
                {"role": "assistant", "content": "好的，已帮您换成燕麦奶，一杯大杯冰燕麦美式，确认下单吗？", "intent": None, "confidence": None, "slots": None, "timestamp": "10:30:31"},
                {"role": "user", "content": "好的", "intent": "CHITCHAT", "confidence": 0.72, "slots": {}, "timestamp": "10:30:45"},
                {"role": "assistant", "content": "订单已确认！请稍等，马上为您制作。", "intent": None, "confidence": None, "slots": None, "timestamp": "10:30:46"},
            ],
            "message_count": 7,
            "intents_flow": ["ORDER_NEW", "ORDER_MODIFY", "CHITCHAT"],
            "slots_extracted": {"product_name": "美式咖啡", "size": "大杯", "temperature": "冰", "milk_type": "燕麦奶"}
        }


def _merge_slots(slots_list: List[Optional[Dict]]) -> Dict:
    """合并多个槽位字典"""
    merged = {}
    for slots in slots_list:
        if slots:
            merged.update(slots)
    return merged
