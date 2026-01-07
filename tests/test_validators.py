"""
验证器测试
"""

import pytest
from pydantic import ValidationError

import sys
sys.path.insert(0, '/Users/sawzhang/code/AIChatOrder')

from core.validators import (
    check_dangerous_patterns,
    check_sql_injection,
    sanitize_text,
    validate_session_id,
    ClassifyRequestV2,
    ChatRequestV2,
    OrderItemRequest,
    OrderRequest
)


class TestDangerousPatterns:
    """危险模式检测测试"""

    def test_script_tag(self):
        """测试检测 script 标签"""
        patterns = check_dangerous_patterns("<script>alert('xss')</script>")
        assert len(patterns) > 0

    def test_javascript_protocol(self):
        """测试检测 javascript 协议"""
        patterns = check_dangerous_patterns("javascript:alert(1)")
        assert len(patterns) > 0

    def test_template_injection(self):
        """测试检测模板注入"""
        patterns = check_dangerous_patterns("{{user.password}}")
        assert len(patterns) > 0

    def test_safe_text(self):
        """测试正常文本"""
        patterns = check_dangerous_patterns("我要一杯拿铁")
        assert len(patterns) == 0


class TestSqlInjection:
    """SQL 注入检测测试"""

    def test_select_statement(self):
        """测试检测 SELECT 语句"""
        result = check_sql_injection("SELECT * FROM users")
        assert result is True

    def test_union_injection(self):
        """测试检测 UNION 注入"""
        result = check_sql_injection("1' UNION SELECT password FROM users --")
        assert result is True

    def test_safe_text(self):
        """测试正常文本"""
        result = check_sql_injection("我要一杯拿铁咖啡")
        assert result is False


class TestSanitizeText:
    """文本清理测试"""

    def test_remove_control_chars(self):
        """测试移除控制字符"""
        text = "hello\x00world"
        result = sanitize_text(text)
        assert "\x00" not in result

    def test_normalize_whitespace(self):
        """测试标准化空白"""
        text = "hello   \n  world"
        result = sanitize_text(text)
        assert result == "hello world"


class TestValidateSessionId:
    """会话 ID 验证测试"""

    def test_valid_uuid(self):
        """测试有效 UUID"""
        assert validate_session_id("550e8400-e29b-41d4-a716-446655440000") is True

    def test_valid_simple(self):
        """测试有效简单格式"""
        assert validate_session_id("session_123") is True
        assert validate_session_id("abc-123") is True

    def test_invalid_format(self):
        """测试无效格式"""
        assert validate_session_id("") is False
        assert validate_session_id("a" * 100) is False
        assert validate_session_id("session@123") is False


class TestClassifyRequestV2:
    """分类请求验证测试"""

    def test_valid_request(self):
        """测试有效请求"""
        req = ClassifyRequestV2(text="我要一杯拿铁")
        assert req.text == "我要一杯拿铁"
        assert req.method == "function_calling"

    def test_text_strip(self):
        """测试文本去除空白"""
        req = ClassifyRequestV2(text="  我要咖啡  ")
        assert req.text == "我要咖啡"

    def test_empty_text(self):
        """测试空文本"""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyRequestV2(text="   ")
        assert "输入文本不能为空" in str(exc_info.value)

    def test_dangerous_content(self):
        """测试危险内容"""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyRequestV2(text="<script>alert(1)</script>")
        assert "不允许的内容" in str(exc_info.value)

    def test_invalid_method(self):
        """测试无效方法"""
        with pytest.raises(ValidationError) as exc_info:
            ClassifyRequestV2(text="test", method="invalid_method")
        assert "无效的分类方法" in str(exc_info.value)

    def test_text_too_long(self):
        """测试文本过长"""
        with pytest.raises(ValidationError):
            ClassifyRequestV2(text="a" * 3000)


class TestChatRequestV2:
    """对话请求验证测试"""

    def test_valid_request(self):
        """测试有效请求"""
        req = ChatRequestV2(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            message="你好"
        )
        assert req.message == "你好"

    def test_invalid_session_id(self):
        """测试无效会话 ID"""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequestV2(session_id="invalid@session", message="test")
        assert "无效的会话ID格式" in str(exc_info.value)

    def test_empty_message(self):
        """测试空消息"""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequestV2(session_id="test-123", message="   ")
        assert "消息不能为空" in str(exc_info.value)


class TestOrderItemRequest:
    """订单项请求验证测试"""

    def test_valid_item(self):
        """测试有效订单项"""
        item = OrderItemRequest(product_name="拿铁", quantity=2)
        assert item.product_name == "拿铁"
        assert item.quantity == 2
        assert item.size == "中杯"

    def test_quantity_range(self):
        """测试数量范围"""
        with pytest.raises(ValidationError):
            OrderItemRequest(product_name="拿铁", quantity=0)

        with pytest.raises(ValidationError):
            OrderItemRequest(product_name="拿铁", quantity=100)

    def test_empty_product_name(self):
        """测试空产品名称"""
        with pytest.raises(ValidationError) as exc_info:
            OrderItemRequest(product_name="   ")
        assert "产品名称不能为空" in str(exc_info.value)


class TestOrderRequest:
    """订单请求验证测试"""

    def test_valid_order(self):
        """测试有效订单"""
        order = OrderRequest(
            session_id="test-123",
            items=[OrderItemRequest(product_name="拿铁")]
        )
        assert len(order.items) == 1

    def test_empty_items(self):
        """测试空订单项"""
        with pytest.raises(ValidationError):
            OrderRequest(session_id="test-123", items=[])

    def test_total_quantity_limit(self):
        """测试总数量限制"""
        with pytest.raises(ValidationError) as exc_info:
            OrderRequest(
                session_id="test-123",
                items=[OrderItemRequest(product_name="拿铁", quantity=51)]
            )
        assert "不能超过50杯" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
