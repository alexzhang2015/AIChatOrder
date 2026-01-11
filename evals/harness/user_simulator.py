"""
LLM 用户模拟器

使用 LLM 模拟真实用户进行多轮对话测试
参考: Anthropic 在 alignment auditing 中使用的方法
"""

import yaml
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class SimulationStatus(Enum):
    """模拟状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    USER_ABANDONED = "user_abandoned"
    MAX_TURNS_REACHED = "max_turns_reached"
    ERROR = "error"


@dataclass
class UserPersona:
    """用户画像"""
    id: str
    name: str
    description: str
    goal: str
    traits: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    behavior_rules: List[str] = field(default_factory=list)
    example_utterances: List[str] = field(default_factory=list)
    target_order: Optional[Dict[str, Any]] = None
    target_behavior: Optional[str] = None
    max_patience_turns: int = 6
    priority: str = "medium"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPersona":
        """从字典创建"""
        return cls(
            id=data.get("id", "unknown"),
            name=data.get("name", "未命名用户"),
            description=data.get("description", ""),
            goal=data.get("goal", ""),
            traits=data.get("traits", []),
            constraints=data.get("constraints", []),
            behavior_rules=data.get("behavior_rules", []),
            example_utterances=data.get("example_utterances", []),
            target_order=data.get("target_order"),
            target_behavior=data.get("target_behavior"),
            max_patience_turns=data.get("max_patience_turns", 6),
            priority=data.get("priority", "medium")
        )


@dataclass
class DialogueTurn:
    """对话轮次"""
    turn_number: int
    role: str  # "user" or "agent"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """模拟结果"""
    persona_id: str
    persona_name: str
    goal: str
    status: SimulationStatus
    transcript: List[DialogueTurn]
    final_order: Optional[Dict[str, Any]]
    total_turns: int
    user_turns: int
    duration_ms: float
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "goal": self.goal,
            "status": self.status.value,
            "transcript": [
                {
                    "turn": t.turn_number,
                    "role": t.role,
                    "content": t.content,
                    "metadata": t.metadata
                }
                for t in self.transcript
            ],
            "final_order": self.final_order,
            "total_turns": self.total_turns,
            "user_turns": self.user_turns,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "metrics": self.metrics,
            "error_message": self.error_message
        }


class LLMUserSimulator:
    """
    LLM 驱动的用户模拟器

    使用 LLM 扮演特定画像的用户，与 Agent 进行多轮对话测试
    """

    SYSTEM_PROMPT_TEMPLATE = """你现在扮演一个咖啡店顾客，正在使用语音/文字点单系统。

## 你的人设
{persona_description}

## 你的目标
{goal}

## 你的特点
{traits}

## 约束条件
{constraints}

## 行为规则
{behavior_rules}

## 示例表达
以下是你可能会说的话（仅供参考，不要完全照搬）：
{example_utterances}

## 输出规则
1. 直接输出你要说的话，不要加引号或其他格式
2. 像真实用户一样说话，包括口语化表达、省略、不完整句子
3. 根据你的人设特点调整说话风格
4. 每次回复不要太长，一般 5-15 个字
5. 如果系统没听懂，你会尝试换种方式说
6. 如果系统多次不理解（超过 {max_patience} 轮），你可能会不耐烦或放弃

## 特殊输出
- 如果你决定放弃点单，输出: [USER_GIVES_UP]
- 如果你确认完成点单，输出: [ORDER_CONFIRMED]
- 如果你想取消订单，按你的人设方式表达（不要直接输出标签）"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        personas_path: str = "evals/fixtures/personas.yaml",
        default_model: str = "gpt-4o-mini",
        default_temperature: float = 0.7
    ):
        """
        初始化模拟器

        Args:
            llm_client: OpenAI 客户端（可选，不传则创建默认客户端）
            personas_path: 用户画像配置路径
            default_model: 默认模型
            default_temperature: 默认温度
        """
        self.llm_client = llm_client
        self.personas_path = Path(personas_path)
        self.default_model = default_model
        self.default_temperature = default_temperature

        self.personas: Dict[str, UserPersona] = {}
        self._load_personas()

    def _load_personas(self):
        """加载用户画像"""
        if self.personas_path.exists():
            with open(self.personas_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            for persona_data in data.get("personas", []):
                persona = UserPersona.from_dict(persona_data)
                self.personas[persona.id] = persona

            logger.info(f"加载了 {len(self.personas)} 个用户画像")
        else:
            logger.warning(f"用户画像文件不存在: {self.personas_path}")

    def _get_llm_client(self):
        """获取或创建 LLM 客户端"""
        if self.llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI()
            except Exception as e:
                logger.error(f"创建 OpenAI 客户端失败: {e}")
                raise
        return self.llm_client

    def get_persona(self, persona_id: str) -> Optional[UserPersona]:
        """获取用户画像"""
        return self.personas.get(persona_id)

    def list_personas(self, priority: Optional[str] = None) -> List[UserPersona]:
        """列出用户画像"""
        personas = list(self.personas.values())
        if priority:
            personas = [p for p in personas if p.priority == priority]
        return personas

    def _build_system_prompt(self, persona: UserPersona) -> str:
        """构建系统提示词"""
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            persona_description=persona.description.strip(),
            goal=persona.goal,
            traits=", ".join(persona.traits) if persona.traits else "普通用户",
            constraints="\n".join(f"- {c}" for c in persona.constraints) if persona.constraints else "无特殊限制",
            behavior_rules="\n".join(f"- {r}" for r in persona.behavior_rules) if persona.behavior_rules else "正常交流",
            example_utterances="\n".join(f"- {u}" for u in persona.example_utterances) if persona.example_utterances else "正常对话",
            max_patience=persona.max_patience_turns
        )

    def generate_user_response(
        self,
        persona: UserPersona,
        conversation_history: List[Dict[str, str]],
        agent_message: str
    ) -> str:
        """
        生成用户响应

        Args:
            persona: 用户画像
            conversation_history: 对话历史
            agent_message: Agent 的最新消息

        Returns:
            用户的回复
        """
        client = self._get_llm_client()

        # 构建消息
        messages = [
            {"role": "system", "content": self._build_system_prompt(persona)}
        ]

        # 添加对话历史
        for turn in conversation_history:
            messages.append(turn)

        # 添加 Agent 的最新消息
        messages.append({"role": "assistant", "content": f"[店员说] {agent_message}"})

        try:
            response = client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=self.default_temperature,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"生成用户响应失败: {e}")
            return "[ERROR]"

    async def generate_user_response_async(
        self,
        persona: UserPersona,
        conversation_history: List[Dict[str, str]],
        agent_message: str
    ) -> str:
        """异步生成用户响应"""
        # 使用线程池执行同步调用
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_user_response,
            persona,
            conversation_history,
            agent_message
        )

    def simulate_conversation(
        self,
        agent_respond_func,
        persona: UserPersona,
        max_turns: int = 10,
        initial_greeting: Optional[str] = None
    ) -> SimulationResult:
        """
        运行同步模拟对话

        Args:
            agent_respond_func: Agent 响应函数，接收用户输入，返回回复
            persona: 用户画像
            max_turns: 最大轮数
            initial_greeting: Agent 的初始问候语

        Returns:
            SimulationResult
        """
        import time
        start_time = time.time()

        transcript = []
        conversation_history = []
        turn_number = 0
        status = SimulationStatus.RUNNING
        final_order = None

        # Agent 初始问候
        agent_greeting = initial_greeting or "您好，欢迎光临！请问想喝点什么？"
        transcript.append(DialogueTurn(
            turn_number=turn_number,
            role="agent",
            content=agent_greeting
        ))
        turn_number += 1

        while turn_number < max_turns * 2:  # *2 因为一轮包含用户和Agent
            # 生成用户回复
            user_response = self.generate_user_response(
                persona,
                conversation_history,
                transcript[-1].content if transcript else agent_greeting
            )

            # 检查特殊标记
            if "[USER_GIVES_UP]" in user_response:
                status = SimulationStatus.USER_ABANDONED
                transcript.append(DialogueTurn(
                    turn_number=turn_number,
                    role="user",
                    content=user_response.replace("[USER_GIVES_UP]", "算了不点了"),
                    metadata={"abandoned": True}
                ))
                break

            if "[ORDER_CONFIRMED]" in user_response:
                status = SimulationStatus.COMPLETED
                transcript.append(DialogueTurn(
                    turn_number=turn_number,
                    role="user",
                    content=user_response.replace("[ORDER_CONFIRMED]", "好的"),
                    metadata={"confirmed": True}
                ))
                break

            if "[ERROR]" in user_response:
                status = SimulationStatus.ERROR
                break

            # 记录用户回复
            transcript.append(DialogueTurn(
                turn_number=turn_number,
                role="user",
                content=user_response
            ))
            conversation_history.append({"role": "user", "content": user_response})
            turn_number += 1

            # 获取 Agent 回复
            try:
                agent_response = agent_respond_func(user_response)

                # 检查是否完成订单
                if isinstance(agent_response, dict):
                    agent_text = agent_response.get("response", str(agent_response))
                    if agent_response.get("order_complete"):
                        final_order = agent_response.get("order")
                        status = SimulationStatus.COMPLETED
                else:
                    agent_text = str(agent_response)

                transcript.append(DialogueTurn(
                    turn_number=turn_number,
                    role="agent",
                    content=agent_text
                ))
                conversation_history.append({"role": "assistant", "content": f"[店员说] {agent_text}"})
                turn_number += 1

                # 检查是否有订单确认的迹象
                if "确认" in agent_text and "订单" in agent_text:
                    # 让用户确认
                    continue

            except Exception as e:
                logger.error(f"Agent 响应失败: {e}")
                status = SimulationStatus.ERROR
                break

        # 检查是否达到最大轮数
        if status == SimulationStatus.RUNNING:
            status = SimulationStatus.MAX_TURNS_REACHED

        duration_ms = (time.time() - start_time) * 1000
        user_turns = len([t for t in transcript if t.role == "user"])

        # 计算成功
        success = (
            status == SimulationStatus.COMPLETED and
            user_turns <= persona.max_patience_turns
        )

        return SimulationResult(
            persona_id=persona.id,
            persona_name=persona.name,
            goal=persona.goal,
            status=status,
            transcript=transcript,
            final_order=final_order,
            total_turns=len(transcript),
            user_turns=user_turns,
            duration_ms=duration_ms,
            success=success,
            metrics={
                "turns_efficiency": user_turns / persona.max_patience_turns if persona.max_patience_turns > 0 else 1,
                "completed": status == SimulationStatus.COMPLETED,
                "abandoned": status == SimulationStatus.USER_ABANDONED
            }
        )

    async def simulate_conversation_async(
        self,
        agent_respond_func,
        persona: UserPersona,
        max_turns: int = 10,
        initial_greeting: Optional[str] = None
    ) -> SimulationResult:
        """异步运行模拟对话"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.simulate_conversation,
            agent_respond_func,
            persona,
            max_turns,
            initial_greeting
        )


class SimulationEvaluator:
    """
    模拟对话评估器

    评估模拟对话的质量
    """

    def __init__(self, simulator: LLMUserSimulator):
        self.simulator = simulator

    def evaluate_result(
        self,
        result: SimulationResult,
        persona: UserPersona
    ) -> Dict[str, Any]:
        """
        评估单个模拟结果

        Args:
            result: 模拟结果
            persona: 用户画像

        Returns:
            评估结果
        """
        scores = {}

        # 1. 完成度评分
        if result.status == SimulationStatus.COMPLETED:
            scores["completion"] = 1.0
        elif result.status == SimulationStatus.MAX_TURNS_REACHED:
            scores["completion"] = 0.5
        else:
            scores["completion"] = 0.0

        # 2. 轮效评分
        if persona.max_patience_turns > 0:
            efficiency = 1 - (result.user_turns / persona.max_patience_turns)
            scores["efficiency"] = max(0, min(1, efficiency + 0.5))  # 调整范围
        else:
            scores["efficiency"] = 0.5

        # 3. 订单匹配评分
        if result.final_order and persona.target_order:
            scores["order_match"] = self._calculate_order_match(
                result.final_order,
                persona.target_order
            )
        else:
            scores["order_match"] = 0.0 if persona.target_order else 1.0

        # 4. 综合评分
        weights = {"completion": 0.4, "efficiency": 0.3, "order_match": 0.3}
        scores["overall"] = sum(
            scores[k] * weights[k] for k in weights
        )

        return {
            "scores": scores,
            "passed": scores["overall"] >= 0.6,
            "details": {
                "user_turns": result.user_turns,
                "max_patience": persona.max_patience_turns,
                "status": result.status.value
            }
        }

    def _calculate_order_match(
        self,
        actual_order: Dict[str, Any],
        expected_order: Dict[str, Any]
    ) -> float:
        """计算订单匹配度"""
        if not expected_order:
            return 1.0

        matches = 0
        total = len(expected_order)

        for key, expected_value in expected_order.items():
            actual_value = actual_order.get(key)
            if actual_value == expected_value:
                matches += 1
            elif isinstance(expected_value, str) and isinstance(actual_value, str):
                # 模糊匹配
                if expected_value.lower() in actual_value.lower() or actual_value.lower() in expected_value.lower():
                    matches += 0.5

        return matches / total if total > 0 else 1.0

    def run_batch_evaluation(
        self,
        agent_respond_func,
        persona_ids: Optional[List[str]] = None,
        trials_per_persona: int = 1
    ) -> Dict[str, Any]:
        """
        批量运行模拟评估

        Args:
            agent_respond_func: Agent 响应函数
            persona_ids: 要测试的画像 ID 列表，不传则测试所有
            trials_per_persona: 每个画像测试次数

        Returns:
            评估报告
        """
        personas = (
            [self.simulator.get_persona(pid) for pid in persona_ids if self.simulator.get_persona(pid)]
            if persona_ids
            else list(self.simulator.personas.values())
        )

        all_results = []
        evaluations = []

        for persona in personas:
            for trial in range(trials_per_persona):
                logger.info(f"运行模拟: {persona.name} (试验 {trial + 1}/{trials_per_persona})")

                result = self.simulator.simulate_conversation(
                    agent_respond_func,
                    persona
                )
                all_results.append(result)

                evaluation = self.evaluate_result(result, persona)
                evaluations.append({
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "trial": trial,
                    **evaluation
                })

        # 汇总统计
        total = len(evaluations)
        passed = sum(1 for e in evaluations if e["passed"])

        return {
            "total_simulations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_overall_score": sum(e["scores"]["overall"] for e in evaluations) / total if total > 0 else 0,
            "by_persona": self._aggregate_by_persona(evaluations),
            "evaluations": evaluations,
            "results": [r.to_dict() for r in all_results]
        }

    def _aggregate_by_persona(self, evaluations: List[Dict]) -> Dict[str, Any]:
        """按画像汇总"""
        by_persona = {}
        for e in evaluations:
            pid = e["persona_id"]
            if pid not in by_persona:
                by_persona[pid] = {
                    "name": e["persona_name"],
                    "trials": 0,
                    "passed": 0,
                    "scores": []
                }
            by_persona[pid]["trials"] += 1
            by_persona[pid]["passed"] += 1 if e["passed"] else 0
            by_persona[pid]["scores"].append(e["scores"]["overall"])

        # 计算平均分
        for pid, data in by_persona.items():
            data["avg_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            data["pass_rate"] = data["passed"] / data["trials"] if data["trials"] > 0 else 0
            del data["scores"]  # 移除原始分数列表

        return by_persona


# =============================================================================
# 便捷函数
# =============================================================================

def load_personas(path: str = "evals/fixtures/personas.yaml") -> Dict[str, UserPersona]:
    """加载用户画像"""
    simulator = LLMUserSimulator(personas_path=path)
    return simulator.personas


def create_simulator(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> LLMUserSimulator:
    """创建模拟器"""
    return LLMUserSimulator(
        default_model=model,
        default_temperature=temperature
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM 用户模拟器测试")
    print("=" * 60)

    # 创建模拟器
    simulator = LLMUserSimulator()

    print(f"\n加载了 {len(simulator.personas)} 个用户画像:")
    for pid, persona in simulator.personas.items():
        print(f"  - {persona.name} ({pid}): {persona.goal[:30]}...")

    # 测试系统提示词生成
    print("\n--- 系统提示词示例 ---")
    if simulator.personas:
        test_persona = list(simulator.personas.values())[0]
        prompt = simulator._build_system_prompt(test_persona)
        print(prompt[:500] + "...")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print("\n注意: 完整的对话模拟需要 OpenAI API 密钥")
