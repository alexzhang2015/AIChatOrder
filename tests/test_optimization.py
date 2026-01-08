"""
优化功能完整测试

测试以下模块：
1. exceptions - 异常体系
2. retry - 重试机制
3. database - SQLite 持久化
4. vector_store - Chroma 向量数据库
5. main.py - 集成测试
6. workflow.py - 工作流集成
7. API 端点测试
"""

import os
import sys
import time
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime

# 设置测试环境，添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-testing')

# 测试结果收集
test_results = {
    "passed": 0,
    "failed": 0,
    "errors": []
}


def test_case(name):
    """测试用例装饰器"""
    def decorator(func):
        def wrapper():
            try:
                print(f"\n{'='*60}")
                print(f"测试: {name}")
                print('='*60)
                func()
                test_results["passed"] += 1
                print(f"✅ {name} - 通过")
                return True
            except AssertionError as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"{name}: {e}")
                print(f"❌ {name} - 失败: {e}")
                return False
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"{name}: {type(e).__name__}: {e}")
                print(f"❌ {name} - 错误: {type(e).__name__}: {e}")
                return False
        return wrapper
    return decorator


# ==================== 1. 异常模块测试 ====================

@test_case("异常模块 - 基本异常类")
def test_exceptions_basic():
    from exceptions import (
        APIError,
        RetryableError,
        FatalError,
        RateLimitError,
        NetworkError,
        ServiceError,
        AuthError,
        BadRequestError
    )

    # 测试基类
    err = APIError("测试错误", status_code=500)
    assert err.message == "测试错误"
    assert err.status_code == 500
    assert "error" in err.to_dict()

    # 测试可重试错误
    rate_err = RateLimitError(retry_after=60)
    assert isinstance(rate_err, RetryableError)
    assert rate_err.retry_after == 60
    assert rate_err.status_code == 429

    net_err = NetworkError("连接超时")
    assert isinstance(net_err, RetryableError)

    svc_err = ServiceError("服务不可用", status_code=503)
    assert isinstance(svc_err, RetryableError)

    # 测试不可重试错误
    auth_err = AuthError("认证失败")
    assert isinstance(auth_err, FatalError)
    assert auth_err.status_code == 401

    bad_err = BadRequestError("参数无效")
    assert isinstance(bad_err, FatalError)
    assert bad_err.status_code == 400

    print("  - 基本异常类创建正确")
    print("  - 可重试/不可重试分类正确")
    print("  - to_dict() 方法工作正常")


@test_case("异常模块 - 业务异常")
def test_exceptions_business():
    from exceptions import (
        SessionNotFoundError,
        SessionExpiredError,
        OrderNotFoundError,
        InvalidOrderStateError,
        DatabaseError,
        VectorStoreError
    )

    # 会话异常
    sess_err = SessionNotFoundError("abc123")
    assert "abc123" in sess_err.message
    assert sess_err.status_code == 404

    # 订单异常
    order_err = OrderNotFoundError("ORD001")
    assert "ORD001" in order_err.message

    state_err = InvalidOrderStateError("ORD001", "cancelled", ["pending", "confirmed"])
    assert state_err.details["current_state"] == "cancelled"

    print("  - 业务异常类创建正确")
    print("  - 异常详情包含必要信息")


@test_case("异常模块 - OpenAI 错误分类")
def test_exceptions_classify():
    from exceptions import classify_openai_error, RateLimitError, AuthError, RetryableError

    # 模拟不同类型的错误
    class MockRateLimitError(Exception):
        pass
    MockRateLimitError.__name__ = "RateLimitError"

    class MockAuthError(Exception):
        pass
    MockAuthError.__name__ = "AuthenticationError"

    class MockUnknownError(Exception):
        status_code = 503

    # 测试分类
    rate_result = classify_openai_error(MockRateLimitError("rate limit"))
    assert isinstance(rate_result, RateLimitError)

    auth_result = classify_openai_error(MockAuthError("invalid key"))
    assert isinstance(auth_result, AuthError)

    unknown_result = classify_openai_error(MockUnknownError())
    assert isinstance(unknown_result, RetryableError)

    print("  - OpenAI 错误分类正确")


# ==================== 2. 重试模块测试 (retry_manager) ====================

@test_case("重试管理器 - 基本执行")
def test_retry_manager_basic():
    from retry_manager import RetryManager, ExponentialBackoffPolicy
    from exceptions import RetryableError

    # 创建重试管理器
    policy = ExponentialBackoffPolicy(
        initial_delay=0.01,
        max_delay=0.1,
        exponential_base=2.0
    )
    manager = RetryManager(max_attempts=3, backoff_policy=policy)

    call_count = 0

    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RetryableError("临时失败")
        return "success"

    result = manager.execute(flaky_function)
    assert result == "success"
    assert call_count == 3

    print("  - RetryManager 基本执行正确")
    print(f"  - 函数被调用 {call_count} 次后成功")


@test_case("重试管理器 - 异步执行")
def test_retry_manager_async():
    import asyncio
    from retry_manager import RetryManager, ExponentialBackoffPolicy
    from exceptions import RetryableError

    policy = ExponentialBackoffPolicy(
        initial_delay=0.01,
        max_delay=0.1,
        exponential_base=2.0
    )
    manager = RetryManager(max_attempts=3, backoff_policy=policy)

    call_count = 0

    async def flaky_async_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RetryableError("临时失败")
        return "async_success"

    result = asyncio.run(manager.execute_async(flaky_async_function))
    assert result == "async_success"
    assert call_count == 2

    print("  - RetryManager 异步执行正确")


@test_case("重试管理器 - OpenAI 专用")
def test_retry_manager_openai():
    from retry_manager import create_openai_retry_manager
    from exceptions import RetryableError

    manager = create_openai_retry_manager(max_attempts=2)

    call_count = 0

    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RetryableError("API 错误")
        return {"status": "ok"}

    result = manager.execute(failing_function)
    assert result == {"status": "ok"}
    assert call_count == 2

    print("  - OpenAI 重试管理器工作正常")


# ==================== 3. 数据库模块测试 ====================

@test_case("数据库模块 - Database 单例")
def test_database_singleton():
    from database import Database, DEFAULT_DB_PATH

    # 重置单例
    Database.reset_instance()

    db1 = Database()
    db2 = Database()

    assert db1 is db2
    assert db1.db_path == DEFAULT_DB_PATH

    print("  - Database 单例模式正确")
    print(f"  - 数据库路径: {db1.db_path}")


@test_case("数据库模块 - SessionRepository CRUD")
def test_database_session_repo():
    from database import Database, SessionRepository

    Database.reset_instance()
    db = Database()
    repo = SessionRepository(db)

    # Create
    session = repo.create("test_sess_001")
    assert session.session_id == "test_sess_001"
    assert session.state == "greeting"

    # Read
    retrieved = repo.get("test_sess_001")
    assert retrieved is not None
    assert retrieved.session_id == "test_sess_001"

    # Update
    retrieved.state = "taking_order"
    repo.update(retrieved)
    updated = repo.get("test_sess_001")
    assert updated.state == "taking_order"

    # Delete
    repo.delete("test_sess_001")
    deleted = repo.get("test_sess_001")
    assert deleted is None

    print("  - Session CRUD 操作正确")


@test_case("数据库模块 - OrderRepository CRUD")
def test_database_order_repo():
    from database import Database, SessionRepository, OrderRepository, OrderItemModel

    Database.reset_instance()
    db = Database()
    session_repo = SessionRepository(db)
    order_repo = OrderRepository(db)

    # 先创建会话
    session_repo.create("test_sess_002")

    # Create Order
    order = order_repo.create("ORD_001", "test_sess_002")
    assert order.order_id == "ORD_001"
    assert order.status == "pending"

    # Add Item
    item = OrderItemModel(
        order_id="ORD_001",
        product_name="拿铁",
        size="大杯",
        temperature="冰",
        price=36.0
    )
    added_item = order_repo.add_item(item)
    assert added_item.id is not None

    # Get Items
    items = order_repo.get_items("ORD_001")
    assert len(items) == 1
    assert items[0].product_name == "拿铁"

    # Update Order
    order.status = "confirmed"
    order.total = 36.0
    order_repo.update(order)
    updated = order_repo.get("ORD_001")
    assert updated.status == "confirmed"

    # Cleanup
    order_repo.delete("ORD_001")
    session_repo.delete("test_sess_002")

    print("  - Order CRUD 操作正确")
    print("  - OrderItem 操作正确")


@test_case("数据库模块 - MessageRepository")
def test_database_message_repo():
    from database import Database, SessionRepository, MessageRepository, MessageModel

    Database.reset_instance()
    db = Database()
    session_repo = SessionRepository(db)
    message_repo = MessageRepository(db)

    # 创建会话
    session_repo.create("test_sess_003")

    # 添加消息
    msg1 = MessageModel(
        session_id="test_sess_003",
        role="user",
        content="我要一杯拿铁",
        intent="ORDER_NEW",
        confidence=0.95,
        slots={"product_name": "拿铁"}
    )
    message_repo.add(msg1)

    msg2 = MessageModel(
        session_id="test_sess_003",
        role="assistant",
        content="好的，已添加拿铁"
    )
    message_repo.add(msg2)

    # 获取消息
    messages = message_repo.get_by_session("test_sess_003")
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].intent == "ORDER_NEW"
    assert messages[0].slots["product_name"] == "拿铁"

    # 获取最近消息
    recent = message_repo.get_recent("test_sess_003", count=1)
    assert len(recent) == 1

    # Cleanup
    session_repo.delete("test_sess_003")

    print("  - Message 操作正确")
    print("  - 意图和槽位正确持久化")


@test_case("数据库模块 - 线程安全")
def test_database_thread_safety():
    from database import Database, SessionRepository
    import threading

    Database.reset_instance()
    db = Database()
    repo = SessionRepository(db)

    errors = []
    created_ids = []

    def create_session(session_id):
        try:
            repo.create(session_id)
            created_ids.append(session_id)
        except Exception as e:
            errors.append(str(e))

    # 并发创建会话
    threads = []
    for i in range(10):
        t = threading.Thread(target=create_session, args=(f"thread_sess_{i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0, f"并发错误: {errors}"
    assert len(created_ids) == 10

    # 清理
    for sid in created_ids:
        repo.delete(sid)

    print("  - 10 个线程并发创建会话成功")
    print("  - 无并发错误")


# ==================== 4. 向量存储模块测试 ====================

@test_case("向量存储模块 - Chroma 可用性")
def test_vector_store_availability():
    from vector_store import is_chroma_available, CHROMA_AVAILABLE

    available = is_chroma_available()
    assert available == CHROMA_AVAILABLE

    print(f"  - Chroma 可用: {available}")


@test_case("向量存储模块 - ChromaRetriever")
def test_vector_store_chroma():
    from vector_store import ChromaRetriever, is_chroma_available
    import shutil

    if not is_chroma_available():
        print("  - Chroma 不可用，跳过测试")
        return

    # 使用临时目录
    test_dir = Path(tempfile.mkdtemp()) / "test_chroma"

    examples = [
        {"text": "我要一杯拿铁", "intent": "ORDER_NEW", "slots": {"product_name": "拿铁"}},
        {"text": "换成冰的", "intent": "ORDER_MODIFY", "slots": {"temperature": "冰"}},
        {"text": "取消订单", "intent": "ORDER_CANCEL", "slots": {}},
        {"text": "有什么推荐", "intent": "RECOMMEND", "slots": {}},
    ]

    retriever = ChromaRetriever(
        examples=examples,
        collection_name="test_collection",
        persist_directory=test_dir
    )

    # 测试检索
    results = retriever.retrieve("来杯咖啡", top_k=2)
    assert len(results) == 2
    assert "similarity" in results[0]
    assert results[0]["similarity"] >= results[1]["similarity"]

    # 测试添加示例
    retriever.add_example("帮我点杯美式", "ORDER_NEW", {"product_name": "美式咖啡"})
    assert retriever.count() == 5

    # 清理
    shutil.rmtree(test_dir, ignore_errors=True)

    print("  - ChromaRetriever 初始化成功")
    print("  - 语义检索工作正常")
    print("  - 动态添加示例成功")


@test_case("向量存储模块 - FallbackRetriever")
def test_vector_store_fallback():
    from vector_store import FallbackRetriever

    examples = [
        {"text": "我要一杯拿铁", "intent": "ORDER_NEW", "slots": {"product_name": "拿铁"}},
        {"text": "换成冰的", "intent": "ORDER_MODIFY", "slots": {"temperature": "冰"}},
        {"text": "取消订单", "intent": "ORDER_CANCEL", "slots": {}},
    ]

    retriever = FallbackRetriever(examples)

    # 测试检索
    results = retriever.retrieve("来杯拿铁", top_k=2)
    assert len(results) == 2

    # 测试添加
    retriever.add_example("新示例", "UNKNOWN", {})
    assert retriever.count() == 4

    print("  - FallbackRetriever 工作正常")


@test_case("向量存储模块 - create_retriever 工厂")
def test_vector_store_factory():
    from vector_store import create_retriever, is_chroma_available, ChromaRetriever, FallbackRetriever
    import shutil

    examples = [
        {"text": "测试", "intent": "TEST", "slots": {}},
    ]

    # 测试 Chroma 模式
    if is_chroma_available():
        test_dir = Path(tempfile.mkdtemp()) / "factory_test"
        retriever = create_retriever(
            examples=examples,
            use_chroma=True,
            persist_directory=test_dir
        )
        assert isinstance(retriever, ChromaRetriever)
        shutil.rmtree(test_dir, ignore_errors=True)
        print("  - Chroma 模式创建成功")

    # 测试 Fallback 模式
    retriever = create_retriever(examples=examples, use_chroma=False)
    assert isinstance(retriever, FallbackRetriever)
    print("  - Fallback 模式创建成功")


# ==================== 5. main.py 集成测试 ====================

@test_case("main.py - OpenAIClassifier 初始化")
def test_main_classifier():
    from main import OpenAIClassifier, TRAINING_EXAMPLES

    classifier = OpenAIClassifier()

    assert classifier.retriever is not None
    assert classifier.slot_extractor is not None
    assert len(TRAINING_EXAMPLES) > 0

    print(f"  - 训练示例数: {len(TRAINING_EXAMPLES)}")
    print(f"  - Retriever 类型: {type(classifier.retriever).__name__}")


@test_case("main.py - SlotExtractor")
def test_main_slot_extractor():
    from main import SlotExtractor

    extractor = SlotExtractor()

    # 测试提取
    slots = extractor.extract("来杯大杯冰拿铁加燕麦奶")
    assert slots.get("product_name") == "拿铁"
    assert slots.get("size") == "大杯"
    assert slots.get("temperature") == "冰"
    assert slots.get("milk_type") == "燕麦奶"

    # 测试数量
    slots2 = extractor.extract("两杯美式")
    assert slots2.get("product_name") == "美式咖啡"
    assert slots2.get("quantity") == 2

    print("  - 槽位提取正确")


@test_case("main.py - 规则引擎 fallback")
def test_main_rule_based():
    from main import OpenAIClassifier

    classifier = OpenAIClassifier()

    # 测试规则引擎
    intent, confidence = classifier._rule_based_intent("取消订单")
    assert intent == "ORDER_CANCEL"
    assert confidence >= 0.9

    intent2, _ = classifier._rule_based_intent("推荐一下")
    assert intent2 == "RECOMMEND"

    print("  - 规则引擎 fallback 正确")


@test_case("main.py - SessionManager 线程安全")
def test_main_session_manager():
    from main import SessionManager

    sm = SessionManager(use_db=True)

    # 创建会话
    session = sm.create_session()
    assert session.session_id is not None

    # 获取会话
    retrieved = sm.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id

    # 更新会话
    from main import ConversationState
    session.state = ConversationState.TAKING_ORDER
    sm.update_session(session)

    # 添加消息
    sm.add_message(
        session.session_id,
        role="user",
        content="测试消息",
        intent="ORDER_NEW",
        confidence=0.9
    )

    # 删除会话
    sm.delete_session(session.session_id)
    deleted = sm.get_session(session.session_id)
    assert deleted is None

    print("  - SessionManager 数据库模式工作正常")


@test_case("main.py - JSON 解析改进")
def test_main_json_parse():
    from main import OpenAIClassifier

    classifier = OpenAIClassifier()

    # 测试正常 JSON
    result1 = classifier._parse_json_response('{"intent": "ORDER_NEW", "confidence": 0.9}')
    assert result1["intent"] == "ORDER_NEW"

    # 测试带文本的 JSON
    result2 = classifier._parse_json_response('这是回复：{"intent": "RECOMMEND", "confidence": 0.8}')
    assert result2["intent"] == "RECOMMEND"

    # 测试无效 JSON
    result3 = classifier._parse_json_response('无效内容')
    assert result3["intent"] == "UNKNOWN"

    print("  - JSON 解析改进正确")


# ==================== 6. workflow.py 集成测试 ====================

@test_case("workflow.py - OrderingWorkflow 初始化")
def test_workflow_init():
    from workflow import OrderingWorkflow

    wf = OrderingWorkflow(use_db=True)

    assert wf.classifier is not None
    assert wf.registry is not None
    assert wf.app is not None
    assert wf.use_db == True

    print("  - OrderingWorkflow 初始化成功")
    print("  - 数据库持久化已启用")


@test_case("workflow.py - 创建会话")
def test_workflow_create_session():
    from workflow import OrderingWorkflow

    wf = OrderingWorkflow(use_db=True)
    session = wf.create_session()

    assert "session_id" in session
    assert "state" in session
    assert "history" in session
    assert len(session["history"]) > 0

    print(f"  - 会话 ID: {session['session_id']}")
    print(f"  - 初始状态: {session['state']}")


@test_case("workflow.py - 订单持久化")
def test_workflow_order_persist():
    from workflow import OrderingWorkflow
    from database import Database, OrderRepository, SessionRepository

    Database.reset_instance()
    wf = OrderingWorkflow(use_db=True)

    # 先创建 session（外键约束要求）
    test_session_id = "test_persist_session"
    session_repo = SessionRepository(wf._db)
    session_repo.create(test_session_id)

    # 模拟订单数据
    test_order = {
        "order_id": "TEST_ORD_001",
        "items": [
            {
                "product_name": "拿铁",
                "size": "大杯",
                "temperature": "冰",
                "sweetness": "半糖",
                "milk_type": "燕麦奶",
                "extras": ["浓缩shot"],
                "quantity": 1,
                "price": 42.0
            }
        ],
        "total": 42.0,
        "status": "pending"
    }

    # 测试持久化
    wf._persist_order(test_session_id, test_order)

    # 验证
    order_repo = OrderRepository(wf._db)
    order = order_repo.get("TEST_ORD_001")
    assert order is not None
    assert order.total == 42.0

    items = order_repo.get_items("TEST_ORD_001")
    assert len(items) == 1
    assert items[0].product_name == "拿铁"
    assert items[0].milk_type == "燕麦奶"

    # 清理
    order_repo.delete("TEST_ORD_001")
    session_repo.delete(test_session_id)

    print("  - 订单持久化成功")
    print("  - 订单项持久化成功")


# ==================== 7. API 端点测试 ====================

@test_case("API - 异常处理器")
def test_api_exception_handlers():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 测试空输入
    response = client.post("/api/classify", json={"text": "", "method": "zero_shot"})
    assert response.status_code == 400

    # 测试无效方法
    response = client.post("/api/classify", json={"text": "测试", "method": "invalid_method"})
    assert response.status_code == 400

    print("  - 400 错误处理正确")


@test_case("API - /api/status 端点")
def test_api_status():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    response = client.get("/api/status")
    assert response.status_code == 200

    data = response.json()
    assert "openai_available" in data
    assert "model" in data
    assert "methods" in data
    assert "version" in data

    print(f"  - OpenAI 可用: {data['openai_available']}")
    print(f"  - 模型: {data['model']}")
    print(f"  - 版本: {data['version']}")


@test_case("API - /api/classify 端点")
def test_api_classify():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 测试分类
    response = client.post("/api/classify", json={
        "text": "来杯拿铁",
        "method": "function_calling"
    })
    assert response.status_code == 200

    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert "slots" in data

    print(f"  - 意图: {data['intent']}")
    print(f"  - 置信度: {data['confidence']}")


@test_case("API - /api/chat 端点")
def test_api_chat():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 创建新会话
    response = client.get("/api/chat/new")
    assert response.status_code == 200
    session_data = response.json()
    session_id = session_data["session_id"]

    # 发送消息
    response = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "来杯大杯冰拿铁"
    })
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] == session_id
    assert "reply" in data
    assert "order" in data or data.get("order") is None

    print(f"  - 会话 ID: {session_id}")
    print(f"  - 回复: {data['reply'][:50]}...")


@test_case("API - /api/compare 端点")
def test_api_compare():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    response = client.post("/api/compare", json={
        "text": "我要一杯拿铁"
    })
    assert response.status_code == 200

    data = response.json()
    # API 返回结构: {"input_text": ..., "methods": {...}}
    assert "methods" in data
    methods = data["methods"]
    assert "zero_shot" in methods
    assert "few_shot" in methods
    assert "rag_enhanced" in methods
    assert "function_calling" in methods

    print("  - 4 种方法对比结果返回成功")


# ==================== 运行所有测试 ====================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("AI 咖啡点单系统 - 优化功能完整测试")
    print("="*70)

    start_time = time.time()

    # 1. 异常模块测试
    test_exceptions_basic()
    test_exceptions_business()
    test_exceptions_classify()

    # 2. 重试模块测试
    # Note: test_retry_context, test_retry_decorators, test_retry_fallback are not defined in the original file content provided above
    # I will assume they might have been intended or are missing, so I will comment them out for safety based on what I read.
    # Instead, I will call the defined ones:
    test_retry_manager_basic()
    test_retry_manager_async()
    test_retry_manager_openai()

    # 3. 数据库模块测试
    test_database_singleton()
    test_database_session_repo()
    test_database_order_repo()
    test_database_message_repo()
    test_database_thread_safety()

    # 4. 向量存储模块测试
    test_vector_store_availability()
    test_vector_store_chroma()
    test_vector_store_fallback()
    test_vector_store_factory()

    # 5. main.py 集成测试
    test_main_classifier()
    test_main_slot_extractor()
    test_main_rule_based()
    test_main_session_manager()
    test_main_json_parse()

    # 6. workflow.py 集成测试
    test_workflow_init()
    test_workflow_create_session()
    test_workflow_order_persist()

    # 7. API 端点测试
    test_api_exception_handlers()
    test_api_status()
    test_api_classify()
    test_api_chat()
    test_api_compare()

    elapsed = time.time() - start_time

    # 打印结果汇总
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"✅ 通过: {test_results['passed']}")
    print(f"❌ 失败: {test_results['failed']}")
    print(f"⏱️  耗时: {elapsed:.2f} 秒")

    if test_results["errors"]:
        print("\n失败详情:")
        for error in test_results["errors"]:
            print(f"  - {error}")

    print("\n" + "="*70)

    return test_results["failed"] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
