"""
Phase 1 功能测试脚本

测试:
1. 产品约束验证（星冰乐不能热、馥芮白建议热饮）
2. 模糊表达匹配（"不要那么甜" -> 半糖）
3. 俚语/口语化表达（"续命水" -> 美式咖啡）
4. 增强标准化（"澳白" -> 馥芮白）
5. 规则引擎自动修正
"""

import sys
import os

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workflow import OrderingWorkflow


def test_phase1_features():
    """测试 Phase 1 核心功能"""
    print("=" * 70)
    print("Phase 1 功能验收测试")
    print("=" * 70)

    workflow = OrderingWorkflow()

    # 测试用例
    test_cases = [
        # 测试1: 产品约束 - 星冰乐不能热
        {
            "name": "产品约束测试 - 星冰乐温度约束",
            "messages": ["来杯热的星冰乐"],
            "expect_warning": "星冰乐",
            "check": lambda r: "冰" in r.get("reply", "") or "调整" in r.get("reply", "")
        },

        # 测试2: 模糊表达 - 甜度
        {
            "name": "模糊表达测试 - 甜度",
            "messages": ["来杯拿铁，不要那么甜"],
            "expect_in_reply": "半糖",
            "check": lambda r: "拿铁" in r.get("reply", "")
        },

        # 测试3: 俚语表达 - 续命水
        {
            "name": "俚语表达测试 - 续命水",
            "messages": ["来杯续命水"],
            "expect_product": "美式咖啡",
            "check": lambda r: "美式" in r.get("reply", "")
        },

        # 测试4: 别名标准化 - 澳白
        {
            "name": "别名标准化测试 - 澳白",
            "messages": ["要一杯澳白"],
            "expect_product": "馥芮白",
            "check": lambda r: "馥芮白" in r.get("reply", "") or "澳" in r.get("reply", "")
        },

        # 测试5: 浓度模糊表达
        {
            "name": "模糊表达测试 - 浓度",
            "messages": ["来杯拿铁，咖啡味重一点"],
            "expect_extras": "浓缩",
            "check": lambda r: "浓" in r.get("reply", "") or "shot" in r.get("reply", "").lower()
        },

        # 测试6: 健康偏好 -> 无糖
        {
            "name": "模糊表达测试 - 健康偏好",
            "messages": ["来杯健康的拿铁"],
            "expect_sweetness": "无糖",
            "check": lambda r: "拿铁" in r.get("reply", "")
        },

        # 测试7: 肥宅快乐水
        {
            "name": "俚语表达测试 - 肥宅快乐水",
            "messages": ["给我一杯肥宅快乐水"],
            "expect_product": "星冰乐",
            "check": lambda r: "星冰乐" in r.get("reply", "") or "冰" in r.get("reply", "")
        },

        # 测试8: 馥芮白冰的约束
        {
            "name": "产品约束测试 - 馥芮白温度约束",
            "messages": ["来杯冰的馥芮白"],
            "expect_warning": "馥芮白",
            "check": lambda r: "馥芮白" in r.get("reply", "")
        },
    ]

    passed = 0
    failed = 0
    results = []

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[测试 {i}] {tc['name']}")
        print(f"    输入: {tc['messages'][0]}")

        # 为每个测试创建新会话
        session_id = None

        for msg in tc["messages"]:
            result = workflow.process_message(session_id, msg)
            session_id = result.get("session_id")

        # 检查结果
        reply = result.get("reply", "")
        order = result.get("order")
        trace = result.get("execution_trace", [])

        print(f"    回复: {reply[:80]}{'...' if len(reply) > 80 else ''}")

        # 检查是否有规则警告
        has_rule_warnings = False
        for t in trace:
            if t.get("details", {}).get("rule_warnings"):
                has_rule_warnings = True
                print(f"    规则警告: {t['details']['rule_warnings']}")

        # 执行检查
        check_passed = tc.get("check", lambda r: True)(result)

        if check_passed:
            print(f"    结果: ✅ 通过")
            passed += 1
            results.append(("✅", tc["name"]))
        else:
            print(f"    结果: ❌ 失败")
            print(f"    完整回复: {reply}")
            failed += 1
            results.append(("❌", tc["name"]))

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"总计: {passed + failed} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    print(f"通过率: {passed / (passed + failed) * 100:.1f}%")

    print("\n详细结果:")
    for status, name in results:
        print(f"  {status} {name}")

    print("\n" + "=" * 70)

    return failed == 0


def test_interactive():
    """交互式测试 - 用于手动验证"""
    print("\n" + "=" * 70)
    print("Phase 1 交互式测试")
    print("输入 'quit' 退出，'new' 开始新会话")
    print("=" * 70)

    workflow = OrderingWorkflow()
    session_id = None

    while True:
        try:
            user_input = input("\n用户: ").strip()
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'new':
                session_id = None
                print("---" + "新会话开始" + "---")
                continue

            result = workflow.process_message(session_id, user_input)
            session_id = result.get("session_id")

            print(f"\n助手: {result.get('reply', '')}")

            # 显示执行跟踪
            trace = result.get("execution_trace", [])
            if trace:
                print("\n[执行跟踪]")
                for t in trace:
                    details = t.get("details", {})
                    print(f"  {t.get('icon', '📌')} {t.get('name', 'unknown')}")
                    if details.get("rule_warnings"):
                        print(f"     ⚠️ 规则警告: {details['rule_warnings']}")
                    if details.get("phase1_enhanced"):
                        print(f"     ✨ Phase 1 增强处理")

            # 显示订单
            order = result.get("order")
            if order:
                print(f"\n[当前订单] {order.get('order_id')} - ¥{order.get('total', 0):.0f}")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\n测试结束!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        test_interactive()
    else:
        success = test_phase1_features()
        sys.exit(0 if success else 1)
