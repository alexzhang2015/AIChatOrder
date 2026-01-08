"""Prompt 模板和 Function Schema 定义"""

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
