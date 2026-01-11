"""
评估执行器

负责加载任务、执行评估、收集结果
"""

import os
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from datetime import datetime

from evals.harness.models import (
    TaskConfig, TestCase, GraderConfig, GraderType,
    Trial, EvalResult, EvalSuiteResult, GraderResult, TaskCategory
)
from evals.harness.environment import EvalEnvironment
from evals.graders.base import BaseGrader
from evals.graders.intent_grader import IntentGrader
from evals.graders.slot_grader import SlotGrader
from evals.graders.llm_grader import LLMRubricGrader
from evals.graders.state_grader import StateGrader
from evals.graders.transcript_grader import TranscriptGrader
from evals.graders.fuzzy_grader import FuzzyMatchGrader
from evals.graders.constraint_grader import ConstraintGrader
from evals.graders.confusion_grader import ConfusionMatrixGrader, SafetyCheckGrader
from evals.graders.performance_grader import (
    LatencyGrader, BenchmarkGrader, PerformanceProfileGrader
)
from evals.harness.user_simulator import (
    LLMUserSimulator, SimulationEvaluator, SimulationStatus, UserPersona
)
from evals.harness.latency_collector import (
    LatencyCollector, LatencyComponent, measure_latency, get_latency_collector
)

logger = logging.getLogger(__name__)


class EvalRunner:
    """
    评估执行器

    负责:
    1. 加载评估任务
    2. 执行单个/批量评估
    3. 收集和聚合结果
    4. 保存 Transcript
    """

    # 评分器类型映射
    GRADER_CLASSES: Dict[GraderType, Type[BaseGrader]] = {
        GraderType.INTENT_ACCURACY: IntentGrader,
        GraderType.SLOT_F1: SlotGrader,
        GraderType.SLOT_NORMALIZATION: SlotGrader,
        GraderType.FUZZY_MATCH: FuzzyMatchGrader,
        GraderType.CONSTRAINT_VALIDATION: ConstraintGrader,
        GraderType.STATE_CHECK: StateGrader,
        GraderType.LLM_RUBRIC: LLMRubricGrader,
        GraderType.TRANSCRIPT: TranscriptGrader,
        GraderType.BEHAVIOR_CHECK: LLMRubricGrader,
        GraderType.CONFUSION_MATRIX: ConfusionMatrixGrader,
        GraderType.SAFETY_CHECK: SafetyCheckGrader,
        # Phase 5: 性能评分器
        GraderType.LATENCY: LatencyGrader,
        GraderType.BENCHMARK: BenchmarkGrader,
        GraderType.PERFORMANCE_PROFILE: PerformanceProfileGrader,
    }

    def __init__(
        self,
        tasks_dir: str = "evals/tasks",
        results_dir: str = "evals/results",
        environment: Optional[EvalEnvironment] = None,
        simulator: Optional[LLMUserSimulator] = None
    ):
        """
        初始化评估执行器

        Args:
            tasks_dir: 任务定义目录
            results_dir: 结果存储目录
            environment: 评估环境（可选）
            simulator: 用户模拟器（可选）
        """
        self.tasks_dir = Path(tasks_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.environment = environment or EvalEnvironment()
        self.simulator = simulator
        self._graders: Dict[GraderType, BaseGrader] = {}

    def get_grader(self, grader_config: GraderConfig) -> BaseGrader:
        """获取或创建评分器实例"""
        grader_type = grader_config.type

        # 每次都创建新实例以使用最新配置
        grader_class = self.GRADER_CLASSES.get(grader_type)
        if grader_class:
            return grader_class(config=grader_config)

        logger.warning(f"未知的评分器类型: {grader_type}, 使用 IntentGrader")
        return IntentGrader(config=grader_config)

    def load_task(self, task_path: str) -> TaskConfig:
        """
        加载单个任务

        Args:
            task_path: 任务文件路径 (YAML)

        Returns:
            TaskConfig
        """
        path = Path(task_path)
        if not path.is_absolute():
            path = self.tasks_dir / path

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return TaskConfig.from_dict(data)

    def load_suite(self, suite_name: str) -> List[TaskConfig]:
        """
        加载任务套件

        Args:
            suite_name: 套件名称 (regression, capability, edge_case)

        Returns:
            任务列表
        """
        tasks = []

        # 遍历所有任务文件
        for task_file in self.tasks_dir.rglob("*.yaml"):
            try:
                task = self.load_task(str(task_file))

                # 根据套件名称过滤
                if suite_name == "regression" and task.category == TaskCategory.REGRESSION:
                    tasks.append(task)
                elif suite_name == "capability" and task.category == TaskCategory.CAPABILITY:
                    tasks.append(task)
                elif suite_name == "edge_case" and task.category == TaskCategory.EDGE_CASE:
                    tasks.append(task)
                elif suite_name == "all":
                    tasks.append(task)

            except Exception as e:
                logger.warning(f"加载任务失败 {task_file}: {e}")

        logger.info(f"加载了 {len(tasks)} 个 {suite_name} 任务")
        return tasks

    def run_task(
        self,
        task: TaskConfig,
        num_trials: int = 3,
        save_transcript: bool = True
    ) -> EvalResult:
        """
        执行单个评估任务

        Args:
            task: 任务配置
            num_trials: 试验次数
            save_transcript: 是否保存对话记录

        Returns:
            EvalResult
        """
        logger.info(f"开始执行任务: {task.id} ({task.name})")
        trials = []

        for i in range(num_trials):
            trial = self._run_single_trial(task, i)
            trials.append(trial)
            logger.debug(f"  Trial {i+1}/{num_trials}: {'PASS' if trial.passed else 'FAIL'}")

        result = self._compute_results(task, trials)

        if save_transcript:
            self._save_result(result)

        return result

    def run_suite(
        self,
        suite_name: str,
        num_trials: int = 3,
        fail_threshold: float = 0.0,
        save_transcript: bool = True
    ) -> EvalSuiteResult:
        """
        执行评估套件

        Args:
            suite_name: 套件名称
            num_trials: 每个任务的试验次数
            fail_threshold: 失败阈值（低于此值则整体失败）
            save_transcript: 是否保存对话记录

        Returns:
            EvalSuiteResult
        """
        tasks = self.load_suite(suite_name)
        if not tasks:
            logger.warning(f"套件 {suite_name} 没有任务")
            return EvalSuiteResult(suite_name=suite_name)

        results = []
        started_at = datetime.now()

        for task in tasks:
            try:
                result = self.run_task(task, num_trials, save_transcript)
                results.append(result)
            except Exception as e:
                logger.error(f"任务 {task.id} 执行失败: {e}")
                results.append(EvalResult(
                    task_id=task.id,
                    task_name=task.name,
                    category=task.category,
                    pass_at_1=0.0,
                    pass_at_k=0.0,
                    pass_all=0.0
                ))

        # 计算统计
        tasks_passed = sum(1 for r in results if r.pass_all >= 1.0)
        tasks_failed = len(results) - tasks_passed
        overall_pass_rate = tasks_passed / len(results) if results else 0

        # 按类别统计
        by_category = {}
        for category in TaskCategory:
            cat_results = [r for r in results if r.category == category]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r.pass_all >= 1.0)
                by_category[category.value] = {
                    "total": len(cat_results),
                    "passed": cat_passed,
                    "pass_rate": cat_passed / len(cat_results)
                }

        completed_at = datetime.now()
        total_duration = (completed_at - started_at).total_seconds() * 1000

        suite_result = EvalSuiteResult(
            suite_name=suite_name,
            tasks_total=len(results),
            tasks_passed=tasks_passed,
            tasks_failed=tasks_failed,
            overall_pass_rate=overall_pass_rate,
            results=results,
            by_category=by_category,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            total_duration_ms=total_duration
        )

        # 保存套件结果
        self._save_suite_result(suite_result)

        # 检查是否达到阈值
        if fail_threshold > 0 and overall_pass_rate < fail_threshold:
            logger.error(f"套件通过率 {overall_pass_rate:.2%} 低于阈值 {fail_threshold:.2%}")

        return suite_result

    def _run_single_trial(self, task: TaskConfig, trial_num: int) -> Trial:
        """执行单次试验"""
        start_time = time.time()

        # 重置环境
        self.environment.reset()

        transcript = []
        outcome = {}
        grader_results = {}
        all_passed = True

        try:
            # 根据任务类型执行
            if task.test_cases:
                # 批量测试用例
                outcome = self._run_test_cases(task)
            elif task.simulation:
                # 对话模拟
                outcome, transcript = self._run_simulation(task)
            else:
                outcome = {"predictions": []}

            # 运行所有评分器
            for grader_config in task.graders:
                grader = self.get_grader(grader_config)
                result = grader.evaluate(
                    predictions=outcome.get("predictions", []),
                    test_cases=task.test_cases,
                    transcript=transcript,
                    outcome=outcome,
                    expect=grader_config.expect
                )
                grader_results[grader_config.type.value] = result
                if not result.passed and grader_config.required:
                    all_passed = False

        except Exception as e:
            logger.error(f"试验执行失败: {e}")
            all_passed = False
            grader_results["error"] = GraderResult(
                grader_type=GraderType.INTENT_ACCURACY,
                passed=False,
                score=0.0,
                details={"error": str(e)}
            )

        duration_ms = (time.time() - start_time) * 1000

        # 收集指标
        metrics = self._collect_metrics(outcome, transcript, task.tracked_metrics)
        metrics["duration_ms"] = duration_ms

        return Trial(
            task_id=task.id,
            trial_number=trial_num,
            transcript=transcript,
            outcome=outcome,
            grader_results=grader_results,
            metrics=metrics,
            passed=all_passed,
            duration_ms=duration_ms
        )

    def _run_test_cases(self, task: TaskConfig) -> Dict[str, Any]:
        """运行批量测试用例"""
        predictions = []
        latency_collector = get_latency_collector()

        for test_case in task.test_cases:
            input_data = test_case.input
            start_time = time.time()

            # 根据输入类型调用不同的处理
            if isinstance(input_data, str):
                # 文本输入 - 调用分类器
                result = self.environment.classify(input_data)
                latency_ms = (time.time() - start_time) * 1000

                # 记录延迟
                latency_collector.record(
                    LatencyComponent.END_TO_END,
                    latency_ms,
                    intent=result.get("intent")
                )

                predictions.append({
                    "input": input_data,
                    "intent": result.get("intent"),
                    "confidence": result.get("confidence", 0),
                    "slots": result.get("slots", {}),
                    "latency_ms": latency_ms
                })
            elif isinstance(input_data, dict):
                # 结构化输入 - 可能是约束验证或模糊匹配
                if "product_name" in input_data or "temperature" in input_data:
                    # 约束验证
                    result = self.environment.validate_constraints(input_data)
                    latency_ms = (time.time() - start_time) * 1000
                    predictions.append({
                        "input": input_data,
                        "latency_ms": latency_ms,
                        **result
                    })
                else:
                    # 其他结构化输入
                    predictions.append({"input": input_data})

        return {"predictions": predictions}

    def _run_simulation(self, task: TaskConfig) -> tuple:
        """
        运行对话模拟

        使用 LLMUserSimulator 模拟用户与 Agent 的多轮对话

        Args:
            task: 任务配置，包含 simulation 字段

        Returns:
            (outcome, transcript) - 结果数据和对话记录
        """
        if not task.simulation:
            logger.warning(f"任务 {task.id} 没有 simulation 配置")
            return {}, []

        # 确保模拟器可用
        if self.simulator is None:
            try:
                self.simulator = LLMUserSimulator()
            except Exception as e:
                logger.error(f"创建用户模拟器失败: {e}")
                return {"error": str(e)}, []

        # 获取用户画像
        persona_id = task.simulation.user_persona
        persona = self.simulator.get_persona(persona_id)

        if persona is None:
            # 如果找不到预定义画像，使用任务配置创建临时画像
            logger.info(f"未找到画像 {persona_id}，使用任务配置创建临时画像")
            persona = UserPersona(
                id=persona_id,
                name=persona_id,
                description=task.simulation.user_goal,
                goal=task.simulation.user_goal,
                max_patience_turns=task.simulation.max_turns
            )

        # 重置环境
        self.environment.reset()

        # 创建 Agent 响应函数
        def agent_respond_func(user_input: str) -> Dict[str, Any]:
            return self.environment.agent_respond(user_input)

        # 运行模拟
        logger.info(f"开始模拟对话: {persona.name} - {persona.goal[:30]}...")

        try:
            result = self.simulator.simulate_conversation(
                agent_respond_func=agent_respond_func,
                persona=persona,
                max_turns=task.simulation.max_turns,
                initial_greeting="您好，欢迎光临！请问想喝点什么？"
            )

            # 转换为标准格式
            transcript = [
                {
                    "turn": turn.turn_number,
                    "role": turn.role,
                    "content": turn.content,
                    "metadata": turn.metadata
                }
                for turn in result.transcript
            ]

            outcome = {
                "persona_id": result.persona_id,
                "persona_name": result.persona_name,
                "goal": result.goal,
                "status": result.status.value,
                "success": result.success,
                "final_order": result.final_order,
                "total_turns": result.total_turns,
                "user_turns": result.user_turns,
                "duration_ms": result.duration_ms,
                "metrics": result.metrics
            }

            # 评估结果
            evaluator = SimulationEvaluator(self.simulator)
            evaluation = evaluator.evaluate_result(result, persona)
            outcome["evaluation"] = evaluation

            logger.info(
                f"模拟完成: {result.status.value}, "
                f"轮数: {result.user_turns}, "
                f"评分: {evaluation['scores'].get('overall', 0):.2f}"
            )

            return outcome, transcript

        except Exception as e:
            logger.error(f"对话模拟失败: {e}")
            return {"error": str(e), "status": "error"}, []

    def _collect_metrics(
        self,
        outcome: Dict[str, Any],
        transcript: List[Dict],
        tracked_metrics: List[str]
    ) -> Dict[str, float]:
        """收集追踪指标"""
        metrics = {}

        predictions = outcome.get("predictions", [])

        if "n_turns" in tracked_metrics:
            metrics["n_turns"] = len([m for m in transcript if m.get("role") == "user"])

        if "total_predictions" in tracked_metrics:
            metrics["total_predictions"] = len(predictions)

        if "avg_confidence" in tracked_metrics:
            confidences = [p.get("confidence", 0) for p in predictions if "confidence" in p]
            metrics["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0

        return metrics

    def _compute_results(self, task: TaskConfig, trials: List[Trial]) -> EvalResult:
        """计算评估结果统计"""
        passes = [t.passed for t in trials]
        k = len(trials)

        # 聚合指标
        aggregate_metrics = {}
        if trials:
            all_metrics_keys = set()
            for t in trials:
                all_metrics_keys.update(t.metrics.keys())

            for key in all_metrics_keys:
                values = [t.metrics.get(key, 0) for t in trials]
                aggregate_metrics[f"{key}_avg"] = sum(values) / len(values)
                aggregate_metrics[f"{key}_min"] = min(values)
                aggregate_metrics[f"{key}_max"] = max(values)

        return EvalResult(
            task_id=task.id,
            task_name=task.name,
            category=task.category,
            trials=trials,
            pass_at_1=1.0 if passes[0] else 0.0,
            pass_at_k=1.0 if any(passes) else 0.0,
            pass_all=1.0 if all(passes) else 0.0,
            aggregate_metrics=aggregate_metrics,
            completed_at=datetime.now().isoformat()
        )

    def _save_result(self, result: EvalResult):
        """保存单个任务结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.task_id}_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.debug(f"结果已保存: {filepath}")

    def _save_suite_result(self, result: EvalSuiteResult):
        """保存套件结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"suite_{result.suite_name}_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"套件结果已保存: {filepath}")


# 便捷函数
def run_eval(
    task_path: str,
    num_trials: int = 3,
    tasks_dir: str = "evals/tasks",
    results_dir: str = "evals/results"
) -> EvalResult:
    """
    运行单个评估任务的便捷函数

    Args:
        task_path: 任务文件路径
        num_trials: 试验次数
        tasks_dir: 任务目录
        results_dir: 结果目录

    Returns:
        EvalResult
    """
    runner = EvalRunner(tasks_dir=tasks_dir, results_dir=results_dir)
    task = runner.load_task(task_path)
    return runner.run_task(task, num_trials=num_trials)


def run_suite(
    suite_name: str,
    num_trials: int = 3,
    fail_threshold: float = 0.0,
    tasks_dir: str = "evals/tasks",
    results_dir: str = "evals/results"
) -> EvalSuiteResult:
    """
    运行评估套件的便捷函数

    Args:
        suite_name: 套件名称
        num_trials: 每个任务的试验次数
        fail_threshold: 失败阈值
        tasks_dir: 任务目录
        results_dir: 结果目录

    Returns:
        EvalSuiteResult
    """
    runner = EvalRunner(tasks_dir=tasks_dir, results_dir=results_dir)
    return runner.run_suite(
        suite_name=suite_name,
        num_trials=num_trials,
        fail_threshold=fail_threshold
    )
