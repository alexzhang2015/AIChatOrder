"""
AI 点单意图识别系统 - 可视化 Demo
FastAPI 后端服务 - 支持多轮对话
"""

import os
import sys
import json
import re
import uuid
import time
import asyncio
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np

# 尝试导入 OpenAI (同步和异步)
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 导入自定义模块
from exceptions import (
    APIError, RetryableError, FatalError, RateLimitError, NetworkError,
    ServiceError, AuthError, BadRequestError, SessionNotFoundError,
    classify_openai_error
)
# retry 模块已废弃，使用 retry_manager 替代
from database import (
    Database, SessionRepository, OrderRepository, MessageRepository,
    SessionModel, OrderModel, OrderItemModel, MessageModel
)
from vector_store import create_retriever, is_chroma_available
from cache import get_api_cache, get_session_cache, APICache
from intent_registry import get_intent_registry, IntentRegistry
from health import get_health_checker, HealthStatus
from retry_manager import RetryManager, ExponentialBackoffPolicy, create_openai_retry_manager
from monitoring import (
    MonitoringMiddleware, get_metrics_collector, get_structured_logger,
    setup_logging, monitor_performance
)
from config import get_settings, get_openai_settings
from resilience import (
    get_rate_limiter, get_circuit_breaker, CircuitOpenError,
    RateLimitMiddleware
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 意图与槽位定义 ====================

# 从配置文件加载意图定义
_intent_registry = get_intent_registry()

class Intent(str, Enum):
    """点单系统意图分类（从配置加载）"""
    ORDER_NEW = "ORDER_NEW"
    ORDER_MODIFY = "ORDER_MODIFY"
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_QUERY = "ORDER_QUERY"
    PRODUCT_INFO = "PRODUCT_INFO"
    RECOMMEND = "RECOMMEND"
    CUSTOMIZE = "CUSTOMIZE"
    PAYMENT = "PAYMENT"
    COMPLAINT = "COMPLAINT"
    CHITCHAT = "CHITCHAT"
    UNKNOWN = "UNKNOWN"


# 从意图注册中心获取描述（支持热更新）
def get_intent_descriptions() -> Dict:
    """获取意图描述（动态加载）"""
    return _intent_registry.get_intent_descriptions()

# 兼容旧代码的静态引用
INTENT_DESCRIPTIONS = _intent_registry.get_intent_descriptions()


# ==================== 训练数据 ====================

TRAINING_EXAMPLES = [
    {"text": "我要一杯拿铁", "intent": "ORDER_NEW", "slots": {"product_name": "拿铁"}},
    {"text": "来杯美式咖啡", "intent": "ORDER_NEW", "slots": {"product_name": "美式咖啡"}},
    {"text": "给我点一杯大杯冰美式", "intent": "ORDER_NEW", "slots": {"product_name": "美式咖啡", "size": "大杯", "temperature": "冰"}},
    {"text": "要一杯中杯热拿铁加燕麦奶", "intent": "ORDER_NEW", "slots": {"product_name": "拿铁", "size": "中杯", "temperature": "热", "milk_type": "燕麦奶"}},
    {"text": "两杯卡布奇诺", "intent": "ORDER_NEW", "slots": {"product_name": "卡布奇诺", "quantity": 2}},
    {"text": "想喝星冰乐", "intent": "ORDER_NEW", "slots": {"product_name": "星冰乐"}},
    {"text": "来份摩卡吧", "intent": "ORDER_NEW", "slots": {"product_name": "摩卡"}},
    {"text": "帮我点杯馥芮白", "intent": "ORDER_NEW", "slots": {"product_name": "馥芮白"}},
    {"text": "一杯超大杯冰拿铁少糖", "intent": "ORDER_NEW", "slots": {"product_name": "拿铁", "size": "超大杯", "temperature": "冰", "sweetness": "少糖"}},
    {"text": "换成大杯", "intent": "ORDER_MODIFY", "slots": {"size": "大杯"}},
    {"text": "改成冰的", "intent": "ORDER_MODIFY", "slots": {"temperature": "冰"}},
    {"text": "能换燕麦奶吗", "intent": "ORDER_MODIFY", "slots": {"milk_type": "燕麦奶"}},
    {"text": "少冰少糖", "intent": "ORDER_MODIFY", "slots": {"temperature": "少冰", "sweetness": "少糖"}},
    {"text": "加一份浓缩", "intent": "ORDER_MODIFY", "slots": {"extras": ["浓缩shot"]}},
    {"text": "我要改成无糖的", "intent": "ORDER_MODIFY", "slots": {"sweetness": "无糖"}},
    {"text": "不要奶油", "intent": "ORDER_MODIFY", "slots": {}},
    {"text": "取消订单", "intent": "ORDER_CANCEL", "slots": {}},
    {"text": "不要了", "intent": "ORDER_CANCEL", "slots": {}},
    {"text": "算了不点了", "intent": "ORDER_CANCEL", "slots": {}},
    {"text": "帮我把刚才的单取消", "intent": "ORDER_CANCEL", "slots": {}},
    {"text": "我的订单到哪了", "intent": "ORDER_QUERY", "slots": {}},
    {"text": "订单什么时候好", "intent": "ORDER_QUERY", "slots": {}},
    {"text": "查一下订单状态", "intent": "ORDER_QUERY", "slots": {}},
    {"text": "还要等多久", "intent": "ORDER_QUERY", "slots": {}},
    {"text": "拿铁多少钱", "intent": "PRODUCT_INFO", "slots": {"product_name": "拿铁"}},
    {"text": "这个有多少卡路里", "intent": "PRODUCT_INFO", "slots": {}},
    {"text": "美式咖啡里面有什么", "intent": "PRODUCT_INFO", "slots": {"product_name": "美式咖啡"}},
    {"text": "有什么新品吗", "intent": "PRODUCT_INFO", "slots": {}},
    {"text": "燕麦奶要加钱吗", "intent": "PRODUCT_INFO", "slots": {"milk_type": "燕麦奶"}},
    {"text": "有什么推荐的", "intent": "RECOMMEND", "slots": {}},
    {"text": "什么比较好喝", "intent": "RECOMMEND", "slots": {}},
    {"text": "帮我推荐一款不太甜的", "intent": "RECOMMEND", "slots": {}},
    {"text": "适合减肥的有吗", "intent": "RECOMMEND", "slots": {}},
    {"text": "夏天喝什么好", "intent": "RECOMMEND", "slots": {}},
    {"text": "可以用微信支付吗", "intent": "PAYMENT", "slots": {}},
    {"text": "有优惠券吗", "intent": "PAYMENT", "slots": {}},
    {"text": "积分可以抵扣吗", "intent": "PAYMENT", "slots": {}},
    {"text": "买一送一还有吗", "intent": "PAYMENT", "slots": {}},
    {"text": "咖啡太苦了", "intent": "COMPLAINT", "slots": {}},
    {"text": "做错了我要的是冰的", "intent": "COMPLAINT", "slots": {}},
    {"text": "等太久了", "intent": "COMPLAINT", "slots": {}},
    {"text": "今天天气真好", "intent": "CHITCHAT", "slots": {}},
    {"text": "你好", "intent": "CHITCHAT", "slots": {}},
    {"text": "谢谢", "intent": "CHITCHAT", "slots": {}},
]


# ==================== Prompt模板 ====================

PROMPT_TEMPLATES = {
    "zero_shot": """你是一个智能咖啡店点单助手的意图识别模块。请分析用户输入，识别意图和槽位。

## 可识别的意图类型
- ORDER_NEW: 新建订单（用户想点新饮品）
- ORDER_MODIFY: 修改订单（修改已点饮品的配置）
- ORDER_CANCEL: 取消订单
- ORDER_QUERY: 查询订单状态
- PRODUCT_INFO: 商品信息咨询（价格、成分、卡路里等）
- RECOMMEND: 请求推荐
- CUSTOMIZE: 特殊定制需求
- PAYMENT: 支付相关
- COMPLAINT: 投诉反馈
- CHITCHAT: 闲聊
- UNKNOWN: 无法识别

## 槽位提取规则
- product_name: 饮品名称（拿铁、美式、卡布奇诺、摩卡、星冰乐、馥芮白等）
- size: 杯型（中杯/大杯/超大杯）
- temperature: 温度（热/冰/温/去冰/少冰/多冰）
- sweetness: 甜度（全糖/半糖/三分糖/无糖/少糖）
- milk_type: 奶类（全脂奶/脱脂奶/燕麦奶/椰奶/豆奶）
- extras: 额外配料（浓缩shot、香草糖浆、焦糖糖浆、奶油、珍珠）
- quantity: 数量

## 输出格式
请严格以JSON格式返回：
{{
  "intent": "意图类型",
  "confidence": 置信度(0-1),
  "slots": {{提取的槽位}},
  "reasoning": "推理过程简述"
}}

## 用户输入
{user_input}

## 分析结果""",

    "few_shot": """你是咖啡店智能点单系统的意图识别模块。根据以下示例学习分类规则。

## 示例

用户: "我要一杯大杯冰美式"
输出: {{"intent": "ORDER_NEW", "confidence": 0.95, "slots": {{"product_name": "美式咖啡", "size": "大杯", "temperature": "冰"}}}}

用户: "换成燕麦奶"
输出: {{"intent": "ORDER_MODIFY", "confidence": 0.92, "slots": {{"milk_type": "燕麦奶"}}}}

用户: "有什么推荐的吗"
输出: {{"intent": "RECOMMEND", "confidence": 0.88, "slots": {{}}}}

用户: "拿铁多少钱"
输出: {{"intent": "PRODUCT_INFO", "confidence": 0.90, "slots": {{"product_name": "拿铁"}}}}

用户: "取消订单"
输出: {{"intent": "ORDER_CANCEL", "confidence": 0.95, "slots": {{}}}}

用户: "订单到哪了"
输出: {{"intent": "ORDER_QUERY", "confidence": 0.93, "slots": {{}}}}

用户: "可以用积分吗"
输出: {{"intent": "PAYMENT", "confidence": 0.89, "slots": {{}}}}

## 当前输入
用户: "{user_input}"
输出:""",

    "rag_enhanced": """你是咖啡店智能点单系统的意图识别模块。

## 检索到的相似历史案例
{retrieved_examples}

## 分析任务
基于上述相似案例的模式，分析当前用户输入的意图和槽位。
注意槽位的细节提取，包括饮品名称、杯型、温度、甜度、奶类选择等。

## 当前用户输入
"{user_input}"

## 输出格式
请返回JSON格式：
{{
  "intent": "意图类型",
  "confidence": 置信度(0-1),
  "slots": {{提取的槽位}},
  "matched_pattern": "匹配的模式说明",
  "reasoning": "推理过程"
}}

分析结果:"""
}


FUNCTION_SCHEMA = {
    "name": "process_order_intent",
    "description": "处理咖啡点单相关的用户意图，识别意图类型并提取订单相关信息",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["ORDER_NEW", "ORDER_MODIFY", "ORDER_CANCEL", "ORDER_QUERY",
                         "PRODUCT_INFO", "RECOMMEND", "CUSTOMIZE", "PAYMENT",
                         "COMPLAINT", "CHITCHAT", "UNKNOWN"],
                "description": "识别的用户意图类型"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "意图识别置信度"
            },
            "order_details": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "饮品名称"},
                    "size": {"type": "string", "enum": ["中杯", "大杯", "超大杯"]},
                    "temperature": {"type": "string", "enum": ["热", "冰", "温", "去冰", "少冰", "多冰"]},
                    "sweetness": {"type": "string", "enum": ["全糖", "半糖", "三分糖", "无糖", "少糖"]},
                    "milk_type": {"type": "string", "enum": ["全脂奶", "脱脂奶", "燕麦奶", "椰奶", "豆奶"]},
                    "extras": {"type": "array", "items": {"type": "string"}},
                    "quantity": {"type": "integer", "minimum": 1, "default": 1}
                },
                "description": "订单详细信息"
            },
            "requires_clarification": {
                "type": "boolean",
                "description": "是否需要进一步澄清用户意图"
            },
            "clarification_question": {
                "type": "string",
                "description": "如需澄清，向用户提出的问题"
            },
            "reasoning": {
                "type": "string",
                "description": "推理过程说明"
            }
        },
        "required": ["intent", "confidence"]
    }
}


# ==================== 向量检索器 ====================

class SimpleVectorRetriever:
    """简化的向量检索器"""

    def __init__(self, examples: List[Dict]):
        self.examples = examples
        self.keywords = {
            'order': ['要', '来', '点', '给我', '帮我', '杯', '份'],
            'modify': ['换', '改', '加', '减', '不要', '少', '多'],
            'cancel': ['取消', '不要了', '算了', '不点'],
            'query': ['查', '到哪', '多久', '状态', '什么时候'],
            'info': ['多少钱', '价格', '什么', '有什么', '卡路里', '成分'],
            'recommend': ['推荐', '好喝', '建议', '适合'],
            'payment': ['支付', '付款', '优惠', '积分', '微信', '支付宝'],
            'complaint': ['投诉', '错了', '太久', '不满意', '差评'],
            'product': ['拿铁', '美式', '卡布奇诺', '摩卡', '星冰乐', '馥芮白', '抹茶']
        }

    def _extract_features(self, text: str) -> Dict[str, float]:
        features = {}
        for category, words in self.keywords.items():
            features[category] = sum(1 for w in words if w in text)
        return features

    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        keys = set(vec1.keys()) | set(vec2.keys())
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
        norm1 = np.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = np.sqrt(sum(v**2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_features = self._extract_features(query)
        scored_examples = []

        for example in self.examples:
            example_features = self._extract_features(example['text'])
            similarity = self._cosine_similarity(query_features, example_features)
            common_chars = len(set(query) & set(example['text']))
            char_bonus = common_chars * 0.02

            scored_examples.append({
                **example,
                'similarity': round(similarity + char_bonus, 4)
            })

        scored_examples.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_examples[:top_k]


# ==================== 槽位提取器 ====================

class SlotExtractor:
    """槽位提取器"""

    def __init__(self):
        self.product_patterns = [
            (r'拿铁', '拿铁'), (r'美式', '美式咖啡'), (r'卡布奇诺', '卡布奇诺'),
            (r'摩卡', '摩卡'), (r'星冰乐', '星冰乐'), (r'馥芮白|澳白', '馥芮白'),
            (r'抹茶', '抹茶拿铁'), (r'焦糖玛奇朵', '焦糖玛奇朵'), (r'香草拿铁', '香草拿铁'),
        ]
        self.size_patterns = [(r'超大杯', '超大杯'), (r'大杯', '大杯'), (r'中杯', '中杯')]
        self.temperature_patterns = [
            (r'去冰', '去冰'), (r'少冰', '少冰'), (r'多冰', '多冰'),
            (r'冰', '冰'), (r'热', '热'), (r'温', '温'),
        ]
        self.sweetness_patterns = [
            (r'无糖', '无糖'), (r'三分糖', '三分糖'), (r'半糖', '半糖'),
            (r'少糖', '少糖'), (r'全糖', '全糖'),
        ]
        self.milk_patterns = [
            (r'燕麦奶', '燕麦奶'), (r'椰奶', '椰奶'), (r'豆奶', '豆奶'),
            (r'脱脂奶|脱脂', '脱脂奶'), (r'全脂奶|全脂', '全脂奶'),
        ]
        self.extras_patterns = [
            (r'浓缩|extra shot', '浓缩shot'), (r'香草糖浆', '香草糖浆'),
            (r'焦糖糖浆', '焦糖糖浆'), (r'奶油', '奶油'), (r'珍珠', '珍珠'),
        ]
        self.quantity_map = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5}

    def extract(self, text: str) -> Dict:
        slots = {}

        for pattern, name in self.product_patterns:
            if re.search(pattern, text):
                slots['product_name'] = name
                break

        for pattern, size in self.size_patterns:
            if re.search(pattern, text):
                slots['size'] = size
                break

        for pattern, temp in self.temperature_patterns:
            if re.search(pattern, text):
                slots['temperature'] = temp
                break

        for pattern, sweetness in self.sweetness_patterns:
            if re.search(pattern, text):
                slots['sweetness'] = sweetness
                break

        for pattern, milk in self.milk_patterns:
            if re.search(pattern, text):
                slots['milk_type'] = milk
                break

        extras = []
        for pattern, extra in self.extras_patterns:
            if re.search(pattern, text):
                extras.append(extra)
        if extras:
            slots['extras'] = extras

        quantity_match = re.search(r'([一二三四五六七八九十两]|[1-9])[份杯]', text)
        if quantity_match:
            q = quantity_match.group(1)
            qty = self.quantity_map.get(q, int(q) if q.isdigit() else 1)
            if qty != 1:
                slots['quantity'] = qty

        return slots


# ==================== OpenAI 分类器 ====================

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
        """带重试逻辑的 OpenAI API 调用（使用 retry_manager 和熔断器）

        Args:
            messages: 消息列表
            tools: Function calling 工具定义
            tool_choice: 工具选择
            max_retries: 最大重试次数（已废弃，使用配置）

        Returns:
            OpenAI API 响应

        Raises:
            FatalError: 不可重试的错误
            CircuitOpenError: 熔断器开启
            RetryableError: 重试后仍然失败
        """
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
        """基于规则的意图识别 (fallback)

        使用意图注册中心的规则进行匹配。
        """
        return self._intent_registry.match_rules(text)

    async def _call_openai_async(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        tool_choice: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Any:
        """异步 OpenAI API 调用（使用 retry_manager 和熔断器）"""
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


# ==================== 菜单数据 ====================

MENU = {
    "拿铁": {"price": 32, "calories": 190, "desc": "经典意式浓缩咖啡与蒸奶的完美结合"},
    "美式咖啡": {"price": 28, "calories": 15, "desc": "浓缩咖啡加水，简单纯粹"},
    "卡布奇诺": {"price": 32, "calories": 120, "desc": "浓缩咖啡、蒸奶和奶泡的经典组合"},
    "摩卡": {"price": 36, "calories": 290, "desc": "浓缩咖啡、巧克力和蒸奶的完美融合"},
    "星冰乐": {"price": 38, "calories": 350, "desc": "冰爽混合饮品，多种口味可选"},
    "馥芮白": {"price": 34, "calories": 140, "desc": "澳洲风格，浓缩咖啡与丝滑奶泡"},
    "抹茶拿铁": {"price": 34, "calories": 240, "desc": "优质抹茶与蒸奶的清新搭配"},
    "焦糖玛奇朵": {"price": 35, "calories": 250, "desc": "香草糖浆、蒸奶、浓缩咖啡和焦糖淋酱"},
}

SIZE_PRICE = {"中杯": 0, "大杯": 4, "超大杯": 7}
MILK_PRICE = {"燕麦奶": 6, "椰奶": 6, "豆奶": 4, "脱脂奶": 0, "全脂奶": 0}
EXTRAS_PRICE = {"浓缩shot": 6, "香草糖浆": 4, "焦糖糖浆": 4, "奶油": 4, "珍珠": 6}


# ==================== 订单和会话管理 ====================

@dataclass
class OrderItem:
    """订单项"""
    product_name: str
    size: str = "中杯"
    temperature: str = "热"
    sweetness: str = "标准"
    milk_type: str = "全脂奶"
    extras: List[str] = field(default_factory=list)
    quantity: int = 1

    def get_price(self) -> float:
        base = MENU.get(self.product_name, {}).get("price", 30)
        size_add = SIZE_PRICE.get(self.size, 0)
        milk_add = MILK_PRICE.get(self.milk_type, 0)
        extras_add = sum(EXTRAS_PRICE.get(e, 0) for e in self.extras)
        return (base + size_add + milk_add + extras_add) * self.quantity

    def to_string(self) -> str:
        parts = []
        if self.quantity > 1:
            parts.append(f"{self.quantity}杯")
        parts.append(self.size)
        parts.append(self.temperature)
        if self.sweetness != "标准":
            parts.append(self.sweetness)
        if self.milk_type != "全脂奶":
            parts.append(self.milk_type)
        parts.append(self.product_name)
        if self.extras:
            parts.append(f"加{'/'.join(self.extras)}")
        return "".join(parts)


@dataclass
class Order:
    """订单"""
    order_id: str
    items: List[OrderItem] = field(default_factory=list)
    status: str = "pending"  # pending, confirmed, preparing, ready, completed, cancelled
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_total(self) -> float:
        return sum(item.get_price() for item in self.items)


class ConversationState(str, Enum):
    """对话状态"""
    GREETING = "greeting"
    TAKING_ORDER = "taking_order"
    CONFIRMING = "confirming"
    MODIFYING = "modifying"
    PAYMENT = "payment"
    COMPLETED = "completed"


@dataclass
class Session:
    """会话"""
    session_id: str
    state: ConversationState = ConversationState.GREETING
    current_order: Optional[Order] = None
    pending_item: Optional[Dict] = None  # 正在收集信息的订单项
    history: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, intent_info: Dict = None):
        self.history.append({
            "role": role,
            "content": content,
            "intent_info": intent_info,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })


class SessionManager:
    """会话管理器（线程安全，支持数据库持久化）"""

    def __init__(self, use_db: bool = True):
        """初始化会话管理器

        Args:
            use_db: 是否使用数据库持久化，默认 True
        """
        self.sessions: Dict[str, Session] = {}  # 内存缓存
        self.session_timeout = 1800  # 30分钟超时
        self._lock = threading.Lock()  # 线程锁
        self.use_db = use_db

        if use_db:
            try:
                self._db = Database()
                self._session_repo = SessionRepository(self._db)
                self._message_repo = MessageRepository(self._db)
                logger.info("会话管理器已启用数据库持久化")
            except Exception as e:
                logger.warning(f"数据库初始化失败，使用纯内存模式: {e}")
                self.use_db = False

    def create_session(self) -> Session:
        """创建新会话"""
        with self._lock:
            session_id = str(uuid.uuid4())[:8]
            session = Session(session_id=session_id)
            self.sessions[session_id] = session

            # 持久化到数据库
            if self.use_db:
                try:
                    self._session_repo.create(session_id)
                except Exception as e:
                    logger.error(f"会话持久化失败: {e}")

            logger.debug(f"创建会话: {session_id}")
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        with self._lock:
            # 先从内存缓存获取
            session = self.sessions.get(session_id)

            if session:
                # 检查是否过期
                if (time.time() - session.created_at) > self.session_timeout:
                    self._delete_session_internal(session_id)
                    return None
                return session

            # 内存中没有，尝试从数据库恢复
            if self.use_db:
                try:
                    db_session = self._session_repo.get(session_id)
                    if db_session:
                        # 检查是否过期
                        if (time.time() - db_session.updated_at) > self.session_timeout:
                            self._session_repo.delete(session_id)
                            return None

                        # 恢复到内存
                        session = Session(
                            session_id=db_session.session_id,
                            state=ConversationState(db_session.state),
                            created_at=db_session.created_at
                        )

                        # 恢复消息历史
                        messages = self._message_repo.get_by_session(session_id)
                        for msg in messages:
                            session.history.append({
                                "role": msg.role,
                                "content": msg.content,
                                "intent_info": {
                                    "intent": msg.intent,
                                    "confidence": msg.confidence,
                                    "slots": msg.slots
                                } if msg.intent else None,
                                "timestamp": msg.timestamp
                            })

                        self.sessions[session_id] = session
                        return session
                except Exception as e:
                    logger.error(f"从数据库恢复会话失败: {e}")

            return None

    def update_session(self, session: Session):
        """更新会话（持久化）"""
        with self._lock:
            self.sessions[session.session_id] = session

            if self.use_db:
                try:
                    db_session = self._session_repo.get(session.session_id)
                    if db_session:
                        db_session.state = session.state.value
                        db_session.current_order_id = session.current_order.order_id if session.current_order else None
                        self._session_repo.update(db_session)
                except Exception as e:
                    logger.error(f"更新会话失败: {e}")

    def add_message(self, session_id: str, role: str, content: str,
                   intent: str = None, confidence: float = None, slots: Dict = None):
        """添加消息并持久化"""
        if self.use_db:
            try:
                message = MessageModel(
                    session_id=session_id,
                    role=role,
                    content=content,
                    intent=intent,
                    confidence=confidence,
                    slots=slots
                )
                self._message_repo.add(message)
            except Exception as e:
                logger.error(f"消息持久化失败: {e}")

    def delete_session(self, session_id: str):
        """删除会话"""
        with self._lock:
            self._delete_session_internal(session_id)

    def _delete_session_internal(self, session_id: str):
        """内部删除会话（不加锁）"""
        if session_id in self.sessions:
            del self.sessions[session_id]

        if self.use_db:
            try:
                self._session_repo.delete(session_id)
            except Exception as e:
                logger.error(f"删除会话失败: {e}")

        logger.debug(f"删除会话: {session_id}")

    def cleanup_expired(self) -> int:
        """清理过期会话"""
        with self._lock:
            # 清理内存中的过期会话
            expired = []
            for session_id, session in self.sessions.items():
                if (time.time() - session.created_at) > self.session_timeout:
                    expired.append(session_id)

            for session_id in expired:
                del self.sessions[session_id]

            # 清理数据库中的过期会话
            db_cleaned = 0
            if self.use_db:
                try:
                    db_cleaned = self._session_repo.cleanup_expired(self.session_timeout)
                except Exception as e:
                    logger.error(f"清理过期会话失败: {e}")

            total = len(expired) + db_cleaned
            if total > 0:
                logger.info(f"清理了 {total} 个过期会话")
            return total


# ==================== 对话助手 ====================

class OrderingAssistant:
    """点单对话助手"""

    def __init__(self, classifier: 'OpenAIClassifier'):
        self.classifier = classifier
        self.session_manager = SessionManager()

    def process_message(self, session_id: Optional[str], user_message: str, method: str = "function_calling") -> Dict:
        """处理用户消息"""
        # 获取或创建会话
        if session_id:
            session = self.session_manager.get_session(session_id)
        else:
            session = None

        if not session:
            session = self.session_manager.create_session()

        # 添加用户消息
        session.add_message("user", user_message)

        # 意图识别
        if method == "zero_shot":
            intent_result = self.classifier.classify_zero_shot(user_message)
        elif method == "few_shot":
            intent_result = self.classifier.classify_few_shot(user_message)
        elif method == "rag_enhanced":
            intent_result = self.classifier.classify_rag(user_message)
        else:
            intent_result = self.classifier.classify_function_calling(user_message)

        intent = intent_result.get("intent", "UNKNOWN")
        slots = intent_result.get("slots", {})
        confidence = intent_result.get("confidence", 0)

        # 根据意图和状态生成回复
        response = self._generate_response(session, intent, slots, intent_result)

        # 添加助手回复
        session.add_message("assistant", response["reply"], {
            "intent": intent,
            "confidence": confidence,
            "slots": slots
        })

        # 构建返回结果
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "reply": response["reply"],
            "intent_result": intent_result,
            "order": self._serialize_order(session.current_order) if session.current_order else None,
            "history": session.history,
            "suggestions": response.get("suggestions", []),
            "actions": response.get("actions", [])
        }

    def _generate_response(self, session: Session, intent: str, slots: Dict, intent_result: Dict) -> Dict:
        """根据意图生成回复"""
        response = {"reply": "", "suggestions": [], "actions": []}

        if intent == "CHITCHAT":
            response = self._handle_chitchat(session, slots)

        elif intent == "ORDER_NEW":
            # 检查是否实际上是确认下单
            if session.current_order and session.current_order.items:
                # 如果已有订单且没有提取到产品名，可能是确认下单
                if not slots.get("product_name"):
                    response = self._handle_payment(session)
                else:
                    response = self._handle_new_order(session, slots)
            else:
                response = self._handle_new_order(session, slots)

        elif intent == "ORDER_MODIFY" or intent == "CUSTOMIZE":
            response = self._handle_modify_order(session, slots)

        elif intent == "ORDER_CANCEL":
            response = self._handle_cancel_order(session)

        elif intent == "ORDER_QUERY":
            response = self._handle_query_order(session)

        elif intent == "PRODUCT_INFO":
            response = self._handle_product_info(slots)

        elif intent == "RECOMMEND":
            response = self._handle_recommend(session)

        elif intent == "PAYMENT":
            response = self._handle_payment(session)

        elif intent == "COMPLAINT":
            response = self._handle_complaint(session)

        else:
            response["reply"] = "抱歉，我没有理解您的意思。您可以说'我要点一杯拿铁'来点单，或者问我'有什么推荐的'。"
            response["suggestions"] = ["我要点杯拿铁", "有什么推荐的", "菜单有什么"]

        return response

    def _handle_chitchat(self, session: Session, slots: Dict) -> Dict:
        greetings = [
            "您好！欢迎光临，请问想喝点什么？",
            "您好！今天想来杯什么咖啡呢？",
            "欢迎光临！我们有多款特色饮品，需要我推荐一下吗？"
        ]
        if session.state == ConversationState.GREETING:
            session.state = ConversationState.TAKING_ORDER

        return {
            "reply": greetings[hash(session.session_id) % len(greetings)],
            "suggestions": ["来杯拿铁", "有什么推荐", "看看菜单"],
            "actions": []
        }

    def _normalize_product_name(self, name: str) -> str:
        """规范化产品名称"""
        if not name:
            return None
        # 处理常见变体
        name_map = {
            "冰美式": "美式咖啡", "热美式": "美式咖啡", "美式": "美式咖啡",
            "冰拿铁": "拿铁", "热拿铁": "拿铁",
            "冰摩卡": "摩卡", "热摩卡": "摩卡",
            "澳白": "馥芮白",
        }
        normalized = name_map.get(name, name)
        # 检查是否在菜单中
        if normalized in MENU:
            return normalized
        # 模糊匹配
        for menu_name in MENU:
            if menu_name in name or name in menu_name:
                return menu_name
        return normalized

    def _handle_new_order(self, session: Session, slots: Dict) -> Dict:
        product_name = self._normalize_product_name(slots.get("product_name"))

        if not product_name:
            return {
                "reply": "请问您想点什么饮品呢？我们有拿铁、美式、卡布奇诺、摩卡等。",
                "suggestions": ["拿铁", "美式咖啡", "卡布奇诺", "有什么推荐"],
                "actions": []
            }

        # 创建订单项
        item = OrderItem(
            product_name=product_name,
            size=slots.get("size", "中杯"),
            temperature=slots.get("temperature", "热"),
            sweetness=slots.get("sweetness", "标准"),
            milk_type=slots.get("milk_type", "全脂奶"),
            extras=slots.get("extras", []),
            quantity=slots.get("quantity", 1)
        )

        # 创建或更新订单
        if not session.current_order:
            session.current_order = Order(order_id=f"ORD{int(time.time()) % 100000:05d}")

        session.current_order.items.append(item)
        session.state = ConversationState.CONFIRMING

        item_desc = item.to_string()
        price = item.get_price()
        total = session.current_order.get_total()

        reply = f"好的，已添加 {item_desc}（¥{price:.0f}）。"
        if len(session.current_order.items) > 1:
            reply += f"\n当前订单共 {len(session.current_order.items)} 件商品，合计 ¥{total:.0f}。"
        reply += "\n\n请问还需要别的吗？或者确认下单？"

        return {
            "reply": reply,
            "suggestions": ["确认下单", "再来一杯", "换成大杯", "取消订单"],
            "actions": ["confirm_order", "add_item", "modify", "cancel"]
        }

    def _handle_modify_order(self, session: Session, slots: Dict) -> Dict:
        if not session.current_order or not session.current_order.items:
            return {
                "reply": "您还没有点单哦，请先告诉我您想喝什么？",
                "suggestions": ["来杯拿铁", "看看菜单", "有什么推荐"],
                "actions": []
            }

        # 修改最后一个订单项
        last_item = session.current_order.items[-1]
        modified = []

        if "size" in slots:
            last_item.size = slots["size"]
            modified.append(f"杯型改为{slots['size']}")

        if "temperature" in slots:
            last_item.temperature = slots["temperature"]
            modified.append(f"温度改为{slots['temperature']}")

        if "sweetness" in slots:
            last_item.sweetness = slots["sweetness"]
            modified.append(f"甜度改为{slots['sweetness']}")

        if "milk_type" in slots:
            last_item.milk_type = slots["milk_type"]
            modified.append(f"奶类改为{slots['milk_type']}")

        if "extras" in slots:
            last_item.extras.extend(slots["extras"])
            modified.append(f"添加{'/'.join(slots['extras'])}")

        # 处理产品名称为配料的情况（如 "加一份浓缩"）
        product_in_slot = slots.get("product_name", "")
        if product_in_slot and product_in_slot not in MENU:
            extra_map = {"浓缩": "浓缩shot", "香草": "香草糖浆", "焦糖": "焦糖糖浆"}
            for key, extra in extra_map.items():
                if key in product_in_slot and extra not in last_item.extras:
                    last_item.extras.append(extra)
                    modified.append(f"添加{extra}")

        if modified:
            reply = f"好的，已为您{', '.join(modified)}。\n"
            reply += f"现在是：{last_item.to_string()}（¥{last_item.get_price():.0f}）\n"
            reply += "还需要其他调整吗？"
        else:
            reply = "请问您想修改什么呢？可以换杯型、温度、甜度或奶类。"

        return {
            "reply": reply,
            "suggestions": ["确认下单", "换成冰的", "少糖", "加燕麦奶"],
            "actions": ["confirm_order", "modify"]
        }

    def _handle_cancel_order(self, session: Session) -> Dict:
        if not session.current_order or not session.current_order.items:
            return {
                "reply": "您还没有点单哦，不需要取消。请问想喝点什么？",
                "suggestions": ["来杯拿铁", "有什么推荐"],
                "actions": []
            }

        session.current_order.status = "cancelled"
        items_count = len(session.current_order.items)
        session.current_order = None
        session.state = ConversationState.TAKING_ORDER

        return {
            "reply": f"好的，已为您取消订单（共{items_count}件商品）。\n有需要随时再点哦！",
            "suggestions": ["重新点单", "看看菜单"],
            "actions": []
        }

    def _handle_query_order(self, session: Session) -> Dict:
        if not session.current_order:
            return {
                "reply": "您目前没有进行中的订单。请问需要点单吗？",
                "suggestions": ["我要点单", "看看菜单"],
                "actions": []
            }

        order = session.current_order
        status_text = {
            "pending": "待确认",
            "confirmed": "已确认，准备制作",
            "preparing": "制作中",
            "ready": "已完成，请取餐",
            "completed": "已完成",
            "cancelled": "已取消"
        }

        items_text = "\n".join([f"  • {item.to_string()} ¥{item.get_price():.0f}" for item in order.items])

        return {
            "reply": f"您的订单 {order.order_id}：\n{items_text}\n\n合计：¥{order.get_total():.0f}\n状态：{status_text.get(order.status, order.status)}",
            "suggestions": ["确认下单", "修改订单", "取消订单"],
            "actions": ["confirm_order", "modify", "cancel"]
        }

    def _handle_product_info(self, slots: Dict) -> Dict:
        product_name = slots.get("product_name")

        if product_name and product_name in MENU:
            info = MENU[product_name]
            reply = f"【{product_name}】\n"
            reply += f"价格：¥{info['price']}（中杯）\n"
            reply += f"热量：{info['calories']} 大卡\n"
            reply += f"介绍：{info['desc']}\n\n"
            reply += "需要来一杯吗？"
            suggestions = [f"来杯{product_name}", "看看其他", "有什么推荐"]
        else:
            # 显示菜单
            reply = "我们的菜单：\n\n"
            for name, info in MENU.items():
                reply += f"• {name}  ¥{info['price']}\n"
            reply += "\n升杯：大杯+4元，超大杯+7元\n换奶：燕麦奶/椰奶+6元\n\n请问想喝什么？"
            suggestions = ["拿铁", "美式咖啡", "推荐一下"]

        return {
            "reply": reply,
            "suggestions": suggestions,
            "actions": []
        }

    def _handle_recommend(self, session: Session) -> Dict:
        recommendations = [
            ("拿铁", "最受欢迎的经典选择，奶香与咖啡香完美平衡"),
            ("馥芮白", "澳洲风味，比拿铁更浓郁顺滑"),
            ("美式咖啡", "低卡之选，适合注重健康的您"),
        ]

        reply = "为您推荐：\n\n"
        for name, reason in recommendations:
            info = MENU[name]
            reply += f"⭐ {name}（¥{info['price']}）\n   {reason}\n\n"
        reply += "请问想试试哪一款？"

        return {
            "reply": reply,
            "suggestions": ["来杯拿铁", "来杯馥芮白", "来杯美式"],
            "actions": []
        }

    def _handle_payment(self, session: Session) -> Dict:
        if not session.current_order or not session.current_order.items:
            return {
                "reply": "您还没有点单哦，请先告诉我您想喝什么？",
                "suggestions": ["来杯拿铁", "看看菜单"],
                "actions": []
            }

        order = session.current_order
        total = order.get_total()

        # 模拟确认订单
        order.status = "confirmed"
        session.state = ConversationState.COMPLETED

        items_text = "\n".join([f"  • {item.to_string()} ¥{item.get_price():.0f}" for item in order.items])

        reply = f"订单已确认！\n\n"
        reply += f"订单号：{order.order_id}\n"
        reply += f"{items_text}\n"
        reply += f"──────────\n"
        reply += f"合计：¥{total:.0f}\n\n"
        reply += "支持微信、支付宝、Apple Pay\n"
        reply += "请稍候，您的饮品马上就好！"

        # 模拟制作
        order.status = "preparing"

        return {
            "reply": reply,
            "suggestions": ["再点一单", "查看订单"],
            "actions": ["new_order", "query_order"]
        }

    def _handle_complaint(self, session: Session) -> Dict:
        return {
            "reply": "非常抱歉给您带来不好的体验！我已经记录下您的反馈，会尽快为您处理。\n\n请问具体是什么问题呢？我们会尽力改进。",
            "suggestions": ["重新制作", "取消订单", "联系经理"],
            "actions": []
        }

    def _serialize_order(self, order: Order) -> Dict:
        """序列化订单"""
        return {
            "order_id": order.order_id,
            "items": [
                {
                    "product_name": item.product_name,
                    "size": item.size,
                    "temperature": item.temperature,
                    "sweetness": item.sweetness,
                    "milk_type": item.milk_type,
                    "extras": item.extras,
                    "quantity": item.quantity,
                    "price": item.get_price(),
                    "description": item.to_string()
                }
                for item in order.items
            ],
            "total": order.get_total(),
            "status": order.status,
            "created_at": order.created_at
        }

    def reset_session(self, session_id: str) -> Dict:
        """重置会话"""
        self.session_manager.delete_session(session_id)
        session = self.session_manager.create_session()
        session.add_message("assistant", "您好！欢迎光临，请问想喝点什么？")
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "history": session.history
        }


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="AI 点单意图识别系统",
    description="基于大语言模型的咖啡店智能点单意图识别可视化 Demo - 支持多轮对话 (LangGraph 架构)",
    version="3.1.0"  # 版本升级：添加异步支持和缓存
)

# ==================== CORS 中间件 ====================

# 允许的来源（生产环境应该限制为具体域名）
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Cache-Status"],
    max_age=600  # 预检请求缓存10分钟
)

# 添加监控中间件
app.add_middleware(MonitoringMiddleware, logger=get_structured_logger("http"))

# ==================== 异常处理器 ====================

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """处理自定义 API 异常"""
    logger.error(f"API 错误: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code or 500,
        content=exc.to_dict()
    )


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
    """处理会话不存在异常"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "SessionNotFound",
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理未捕获的异常"""
    logger.exception(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "服务器内部错误，请稍后重试"
        }
    )


# 全局实例
classifier = OpenAIClassifier()
assistant = OrderingAssistant(classifier)

# LangGraph 工作流实例 (延迟初始化)
_langgraph_workflow = None

def get_langgraph_workflow():
    """获取 LangGraph 工作流实例（单例模式）"""
    global _langgraph_workflow
    if _langgraph_workflow is None:
        try:
            from workflow import OrderingWorkflow
            _langgraph_workflow = OrderingWorkflow(classifier)
            print("✅ LangGraph 工作流已初始化")
        except Exception as e:
            print(f"⚠️ LangGraph 工作流初始化失败: {e}")
            _langgraph_workflow = None
    return _langgraph_workflow


# ==================== 请求模型（增强验证）====================

VALID_METHODS = {"zero_shot", "few_shot", "rag_enhanced", "function_calling"}
MAX_TEXT_LENGTH = 500
MIN_TEXT_LENGTH = 1


class ClassifyRequest(BaseModel):
    """意图分类请求"""
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="要分类的用户输入文本"
    )
    method: str = Field(
        default="function_calling",
        description="分类方法: zero_shot, few_shot, rag_enhanced, function_calling"
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("输入文本不能为空")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"输入文本过长，最大 {MAX_TEXT_LENGTH} 字符")
        return v

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in VALID_METHODS:
            raise ValueError(f"无效的分类方法: {v}，可选: {', '.join(VALID_METHODS)}")
        return v


class CompareRequest(BaseModel):
    """方法对比请求"""
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="要对比的用户输入文本"
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("输入文本不能为空")
        return v


class ChatRequest(BaseModel):
    """对话请求"""
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="会话ID"
    )
    message: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="用户消息"
    )

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("消息不能为空")
        return v


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面 - 意图分析"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>请确保 static/index.html 存在</h1>")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """返回多轮对话页面"""
    html_path = Path(__file__).parent / "static" / "chat.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>请确保 static/chat.html 存在</h1>")


@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    workflow = get_langgraph_workflow()
    api_cache = get_api_cache()
    session_cache = get_session_cache()

    return {
        "openai_available": classifier.is_available(),
        "async_available": classifier.async_client is not None,
        "model": classifier.model,
        "methods": list(VALID_METHODS),
        "intent_types": get_intent_descriptions(),
        "example_count": len(TRAINING_EXAMPLES),
        "langgraph_available": workflow is not None,
        "version": "3.1.0",
        "engines": ["langgraph", "legacy"],
        "cache": {
            "api": api_cache.stats(),
            "session": session_cache.stats()
        },
        "intent_registry": _intent_registry.stats()
    }


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


# ==================== 健康检查端点 ====================

@app.get("/health")
async def health_check():
    """系统健康检查

    检查数据库、OpenAI API、缓存等服务状态。
    """
    health_checker = get_health_checker()
    report = await health_checker.check_all()

    status_code = 200 if report.status == HealthStatus.HEALTHY else 503
    return JSONResponse(
        status_code=status_code,
        content=report.to_dict()
    )


@app.get("/health/{check_name}")
async def health_check_single(check_name: str):
    """单项健康检查

    Args:
        check_name: 检查项名称 (database, openai, cache, vector_store, intent_registry)
    """
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


@app.post("/api/classify")
async def classify_intent(request: ClassifyRequest):
    """执行意图分类（异步，带缓存）"""
    text = request.text  # 已在 validator 中 strip
    method = request.method

    # 优先使用异步方法
    if method == "zero_shot":
        result = await classifier.classify_zero_shot_async(text)
    elif method == "few_shot":
        # few_shot 暂时使用同步（可后续添加异步版本）
        result = classifier.classify_few_shot(text)
    elif method == "rag_enhanced":
        result = classifier.classify_rag(text)
    elif method == "function_calling":
        result = await classifier.classify_function_calling_async(text)
    else:
        # 不应该到这里，validator 已经检查过
        raise HTTPException(status_code=400, detail=f"未知方法: {method}")

    # 添加意图描述信息
    intent = result.get("intent", "UNKNOWN")
    result["intent_info"] = INTENT_DESCRIPTIONS.get(intent, INTENT_DESCRIPTIONS["UNKNOWN"])
    result["method"] = method
    result["input_text"] = text

    # 标记是否来自缓存
    result["from_cache"] = result.pop("_cached", False)

    return result


@app.post("/api/compare")
async def compare_methods(request: CompareRequest):
    """对比所有方法的结果"""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="输入文本不能为空")

    results = {
        "input_text": text,
        "methods": {}
    }

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
        result["intent_info"] = INTENT_DESCRIPTIONS.get(intent, INTENT_DESCRIPTIONS["UNKNOWN"])
        results["methods"][method] = result

    return results


@app.get("/api/examples")
async def get_examples():
    """获取训练示例"""
    return {
        "examples": TRAINING_EXAMPLES,
        "count": len(TRAINING_EXAMPLES)
    }


@app.get("/api/prompts")
async def get_prompts():
    """获取 Prompt 模板"""
    return {
        "templates": PROMPT_TEMPLATES,
        "function_schema": FUNCTION_SCHEMA
    }


# ==================== 多轮对话 API ====================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    method: str = "function_calling"
    use_langgraph: bool = True  # 是否使用 LangGraph 工作流


class ResetRequest(BaseModel):
    session_id: str
    use_langgraph: bool = True


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """多轮对话接口 - 支持 LangGraph 和传统模式"""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 尝试使用 LangGraph 工作流
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

    # 添加意图描述
    intent = result["intent_result"].get("intent", "UNKNOWN")
    result["intent_result"]["intent_info"] = INTENT_DESCRIPTIONS.get(intent, INTENT_DESCRIPTIONS["UNKNOWN"])
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

    # 回退到传统模式
    session = assistant.session_manager.create_session()
    session.add_message("assistant", "您好！欢迎光临，请问想喝点什么？")
    return {
        "session_id": session.session_id,
        "state": session.state.value,
        "history": session.history,
        "suggestions": ["来杯拿铁", "有什么推荐", "看看菜单"],
        "engine": "legacy"
    }


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
        return {
            "available": True,
            "mermaid": workflow.get_graph_visualization()
        }
    return {
        "available": False,
        "mermaid": None
    }


def run(reload: bool = False):
    """
    运行服务器

    Args:
        reload: 是否启用自动重载（开发模式）
    """
    import uvicorn

    # 预初始化 LangGraph 工作流
    workflow = get_langgraph_workflow()

    print("\n" + "=" * 60)
    print("   AI 点单意图识别系统 - 可视化 Demo v3.0")
    print("   Powered by LangGraph")
    print("=" * 60)
    print(f"\n🌐 打开浏览器访问: http://localhost:8000")
    print(f"💬 多轮对话页面: http://localhost:8000/chat")
    print(f"📊 OpenAI 状态: {'✅ 可用' if classifier.is_available() else '❌ 不可用 (将使用规则引擎)'}")
    print(f"🤖 模型: {classifier.model}")
    print(f"🔄 LangGraph: {'✅ 已启用' if workflow else '❌ 未启用 (使用传统模式)'}")
    print(f"🔃 自动重载: {'✅ 已启用' if reload else '❌ 未启用'}")
    print("\n按 Ctrl+C 停止服务\n")

    if reload:
        # 自动重载模式：使用字符串引用 app
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[str(Path(__file__).parent)],
            reload_includes=["*.py", "*.yaml", "*.html"]
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI 点单意图识别系统")
    parser.add_argument("--reload", "-r", action="store_true", help="启用自动重载（开发模式）")
    args = parser.parse_args()
    run(reload=args.reload)
