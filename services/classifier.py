"""OpenAI 意图分类器"""

import json
import logging
import re
from typing import Dict, List, Optional, Any

from infrastructure.exceptions import FatalError, RetryableError, classify_openai_error
from nlp.vector_store import create_retriever, is_chroma_available
from infrastructure.cache import get_api_cache
from nlp.intent_registry import get_intent_registry
from infrastructure.retry_manager import create_openai_retry_manager
from infrastructure.resilience import get_circuit_breaker, CircuitOpenError
from config import get_openai_settings
from nlp.extractor import SlotExtractor
from nlp.prompts import PROMPT_TEMPLATES, FUNCTION_SCHEMA
from data.training import TRAINING_EXAMPLES

# 尝试导入 OpenAI
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenAIClassifier:
    """基于 OpenAI 的意图分类器"""

    def __init__(self):
        self.client = None
        self.async_client = None

        # 从配置获取设置
        openai_settings = get_openai_settings()
        self.model = openai_settings.model

        # 使用 Chroma 向量检索器（如果可用）
        self.retriever = create_retriever(
            examples=TRAINING_EXAMPLES,
            use_chroma=is_chroma_available(),
            collection_name="coffee_order_examples"
        )
        self.slot_extractor = SlotExtractor()

        # API 响应缓存
        self._cache = get_api_cache()

        # 意图注册中心（用于规则匹配）
        self._intent_registry = get_intent_registry()

        # 重试管理器
        self._retry_manager = create_openai_retry_manager(
            max_attempts=openai_settings.max_retries
        )

        # 熔断器
        self._circuit_breaker = get_circuit_breaker(
            name="openai",
            failure_threshold=5,
            success_threshold=3,
            timeout=30.0
        )

        if OPENAI_AVAILABLE and openai_settings.api_key:
            try:
                # 同步客户端
                self.client = OpenAI(
                    api_key=openai_settings.api_key,
                    base_url=openai_settings.base_url or None,
                    timeout=openai_settings.timeout
                )

                # 异步客户端
                self.async_client = AsyncOpenAI(
                    api_key=openai_settings.api_key,
                    base_url=openai_settings.base_url or None,
                    timeout=openai_settings.timeout
                )

                logger.info(f"OpenAI 客户端初始化成功，模型: {self.model}")
            except Exception as e:
                logger.error(f"OpenAI 客户端初始化失败: {e}")
                self.client = None
                self.async_client = None

    def is_available(self) -> bool:
        return self.client is not None

    def _call_openai_with_retry(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        tool_choice: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Any:
        """带重试逻辑的 OpenAI API 调用"""
        # 检查熔断器
        if not self._circuit_breaker.allow_request():
            raise CircuitOpenError("OpenAI 熔断器处于开启状态，请稍后重试")

        def _make_request():
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 500
            }
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

            try:
                response = self.client.chat.completions.create(**kwargs)
                self._circuit_breaker.record_success()
                return response
            except Exception as e:
                classified_error = classify_openai_error(e)
                self._circuit_breaker.record_failure()
                raise classified_error

        return self._retry_manager.execute(_make_request)

    def _rule_based_intent(self, text: str) -> tuple:
        """基于规则的意图识别 (fallback)"""
        return self._intent_registry.match_rules(text)

    async def _call_openai_async(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        tool_choice: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Any:
        """异步 OpenAI API 调用"""
        if not self.async_client:
            raise FatalError("异步 OpenAI 客户端不可用")

        # 检查熔断器
        if not self._circuit_breaker.allow_request():
            raise CircuitOpenError("OpenAI 熔断器处于开启状态，请稍后重试")

        async def _make_request():
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 500
            }
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

            try:
                response = await self.async_client.chat.completions.create(**kwargs)
                self._circuit_breaker.record_success()
                return response
            except Exception as e:
                classified_error = classify_openai_error(e)
                self._circuit_breaker.record_failure()
                raise classified_error

        return await self._retry_manager.execute_async(_make_request)

    async def classify_zero_shot_async(self, text: str) -> Dict:
        """异步零样本分类"""
        # 检查缓存
        cached = self._cache.get(text, "zero_shot")
        if cached:
            return cached

        prompt = PROMPT_TEMPLATES["zero_shot"].format(user_input=text)

        if not self.async_client:
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": "规则引擎 fallback (OpenAI 不可用)",
                "prompt": prompt,
                "llm_response": None
            }

        try:
            response = await self._call_openai_async(
                messages=[{"role": "user", "content": prompt}]
            )

            llm_response = response.choices[0].message.content
            result = self._parse_json_response(llm_response)
            result["prompt"] = prompt
            result["llm_response"] = llm_response

            # 缓存结果
            self._cache.set(text, "zero_shot", result)
            return result

        except (FatalError, RetryableError) as e:
            logger.warning(f"异步零样本分类失败，使用规则引擎: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {e.message}",
                "prompt": prompt,
                "llm_response": None
            }

    async def classify_function_calling_async(self, text: str) -> Dict:
        """异步 Function Calling 分类"""
        # 检查缓存
        cached = self._cache.get(text, "function_calling")
        if cached:
            return cached

        if not self.async_client:
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": "规则引擎 fallback (OpenAI 不可用)",
                "tool_call": None
            }

        try:
            response = await self._call_openai_async(
                messages=[
                    {"role": "system", "content": "你是一个咖啡店点单助手，负责理解顾客的意图。"},
                    {"role": "user", "content": text}
                ],
                tools=[{"type": "function", "function": FUNCTION_SCHEMA}],
                tool_choice={"type": "function", "function": {"name": "classify_intent"}}
            )

            message = response.choices[0].message
            if message.tool_calls and len(message.tool_calls) > 0:
                tool_call = message.tool_calls[0]
                arguments = json.loads(tool_call.function.arguments)

                result = {
                    "intent": arguments.get("intent", "UNKNOWN"),
                    "confidence": arguments.get("confidence", 0.0),
                    "slots": arguments.get("slots", {}),
                    "reasoning": arguments.get("reasoning", ""),
                    "tool_call": {
                        "name": tool_call.function.name,
                        "arguments": arguments
                    }
                }

                # 缓存结果
                self._cache.set(text, "function_calling", result)
                return result

            # 无工具调用，降级到规则引擎
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": "无工具调用，规则引擎 fallback",
                "tool_call": None
            }

        except (FatalError, RetryableError) as e:
            logger.warning(f"Function Calling 失败，使用规则引擎: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {e.message}",
                "tool_call": None
            }

    def classify_zero_shot(self, text: str) -> Dict:
        """零样本分类"""
        prompt = PROMPT_TEMPLATES["zero_shot"].format(user_input=text)

        if not self.is_available():
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": "规则引擎 fallback (OpenAI 不可用)",
                "prompt": prompt,
                "llm_response": None
            }

        try:
            response = self._call_openai_with_retry(
                messages=[{"role": "user", "content": prompt}]
            )

            llm_response = response.choices[0].message.content
            result = self._parse_json_response(llm_response)
            result["prompt"] = prompt
            result["llm_response"] = llm_response
            return result

        except (FatalError, RetryableError) as e:
            logger.warning(f"零样本分类失败，使用规则引擎: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {e.message}",
                "prompt": prompt,
                "llm_response": None
            }
        except Exception as e:
            logger.error(f"零样本分类未知错误: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {str(e)}",
                "prompt": prompt,
                "llm_response": None
            }

    def classify_few_shot(self, text: str) -> Dict:
        """少样本分类"""
        prompt = PROMPT_TEMPLATES["few_shot"].format(user_input=text)

        if not self.is_available():
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": min(confidence + 0.03, 0.98),
                "slots": slots,
                "reasoning": "规则引擎 fallback (OpenAI 不可用)",
                "prompt": prompt,
                "llm_response": None
            }

        try:
            response = self._call_openai_with_retry(
                messages=[{"role": "user", "content": prompt}]
            )

            llm_response = response.choices[0].message.content
            result = self._parse_json_response(llm_response)
            result["prompt"] = prompt
            result["llm_response"] = llm_response
            return result

        except (FatalError, RetryableError) as e:
            logger.warning(f"少样本分类失败，使用规则引擎: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {e.message}",
                "prompt": prompt,
                "llm_response": None
            }
        except Exception as e:
            logger.error(f"少样本分类未知错误: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {str(e)}",
                "prompt": prompt,
                "llm_response": None
            }

    def classify_rag(self, text: str, top_k: int = 3) -> Dict:
        """RAG 增强分类"""
        similar_examples = self.retriever.retrieve(text, top_k)

        examples_text = "\n".join([
            f"{i+1}. \"{ex['text']}\" → 意图: {ex['intent']}, 槽位: {json.dumps(ex['slots'], ensure_ascii=False)} (相似度: {ex['similarity']:.2f})"
            for i, ex in enumerate(similar_examples)
        ])

        prompt = PROMPT_TEMPLATES["rag_enhanced"].format(
            retrieved_examples=examples_text,
            user_input=text
        )

        if not self.is_available():
            # 基于检索结果投票
            intent_votes = {}
            for ex in similar_examples:
                intent_votes[ex['intent']] = intent_votes.get(ex['intent'], 0) + ex['similarity']

            if intent_votes:
                best_intent = max(intent_votes, key=intent_votes.get)
                confidence = min(0.7 + intent_votes[best_intent] * 0.3, 0.98)
            else:
                best_intent, confidence = self._rule_based_intent(text)

            slots = self.slot_extractor.extract(text)
            return {
                "intent": best_intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"基于检索投票 (OpenAI 不可用): 最相似案例 \"{similar_examples[0]['text'] if similar_examples else 'N/A'}\"",
                "retrieved_examples": similar_examples,
                "prompt": prompt,
                "llm_response": None
            }

        try:
            response = self._call_openai_with_retry(
                messages=[{"role": "user", "content": prompt}]
            )

            llm_response = response.choices[0].message.content
            result = self._parse_json_response(llm_response)
            result["retrieved_examples"] = similar_examples
            result["prompt"] = prompt
            result["llm_response"] = llm_response
            return result

        except (FatalError, RetryableError) as e:
            logger.warning(f"RAG分类失败，使用规则引擎: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {e.message}",
                "retrieved_examples": similar_examples,
                "prompt": prompt,
                "llm_response": None
            }
        except Exception as e:
            logger.error(f"RAG分类未知错误: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {str(e)}",
                "retrieved_examples": similar_examples,
                "prompt": prompt,
                "llm_response": None
            }

    def classify_function_calling(self, text: str) -> Dict:
        """Function Calling 分类"""
        if not self.is_available():
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": "规则引擎 fallback (OpenAI 不可用)",
                "function_schema": FUNCTION_SCHEMA,
                "llm_response": None
            }

        try:
            response = self._call_openai_with_retry(
                messages=[
                    {"role": "system", "content": "你是一个智能咖啡店点单助手，负责识别用户的点单意图。"},
                    {"role": "user", "content": text}
                ],
                tools=[{"type": "function", "function": FUNCTION_SCHEMA}],
                tool_choice={"type": "function", "function": {"name": "process_order_intent"}}
            )

            # 安全地访问 tool_calls
            if not response.choices or not response.choices[0].message.tool_calls:
                logger.warning("Function Calling 未返回工具调用，使用规则引擎")
                intent, confidence = self._rule_based_intent(text)
                slots = self.slot_extractor.extract(text)
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "slots": slots,
                    "reasoning": "Function Calling 未返回工具调用",
                    "function_schema": FUNCTION_SCHEMA,
                    "llm_response": None
                }

            tool_call = response.choices[0].message.tool_calls[0]

            try:
                result = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(f"Function Calling 返回的 JSON 解析失败: {e}")
                intent, confidence = self._rule_based_intent(text)
                slots = self.slot_extractor.extract(text)
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "slots": slots,
                    "reasoning": "Function Calling 返回的 JSON 解析失败",
                    "function_schema": FUNCTION_SCHEMA,
                    "llm_response": tool_call.function.arguments
                }

            # 标准化结果
            slots = result.get("order_details", {})
            return {
                "intent": result.get("intent", "UNKNOWN"),
                "confidence": result.get("confidence", 0.5),
                "slots": slots,
                "reasoning": result.get("reasoning", "Function Calling 结构化输出"),
                "requires_clarification": result.get("requires_clarification", False),
                "clarification_question": result.get("clarification_question"),
                "function_schema": FUNCTION_SCHEMA,
                "llm_response": tool_call.function.arguments
            }

        except (FatalError, RetryableError) as e:
            logger.warning(f"Function Calling 失败，使用规则引擎: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {e.message}",
                "function_schema": FUNCTION_SCHEMA,
                "llm_response": None
            }
        except Exception as e:
            logger.error(f"Function Calling 未知错误: {e}")
            intent, confidence = self._rule_based_intent(text)
            slots = self.slot_extractor.extract(text)
            return {
                "intent": intent,
                "confidence": confidence,
                "slots": slots,
                "reasoning": f"LLM 调用失败: {str(e)}",
                "function_schema": FUNCTION_SCHEMA,
                "llm_response": None
            }

    def _parse_json_response(self, response: str) -> Dict:
        """解析 LLM 返回的 JSON"""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.debug(f"直接 JSON 解析失败: {e}")

        # 尝试提取 JSON 块
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.debug(f"提取 JSON 块解析失败: {e}")

        # 解析失败
        logger.warning(f"无法解析 JSON，原始响应: {response[:200]}")
        return {
            "intent": "UNKNOWN",
            "confidence": 0.5,
            "slots": {},
            "reasoning": "JSON 解析失败"
        }
