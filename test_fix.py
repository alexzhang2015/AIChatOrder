from workflow import OrderingWorkflow
try:
    w = OrderingWorkflow(use_db=False)
    print('Workflow init success')
    # 模拟一次处理
    result = w.process_message(None, "你好")
    print("Process message success")
except Exception as e:
    print(f"Error: {e}")
