import requests
import time
import json
import os

BASE_URL = "http://localhost:8001"
DSL_PATH = os.path.abspath("dsl_files/workflow.yaml")

def run_workflow(dsl_path, input_data):
    print(f"\n[Client] 请求执行: {dsl_path}")
    resp = requests.post(f"{BASE_URL}/run", json={
        "dsl_path": dsl_path,
        "input": input_data
    })
    if resp.status_code == 200:
        data = resp.json()
        print(f"[Client] 提交成功, Trace ID: {data['trace_id']}")
        return data['trace_id']
    else:
        print(f"[Client] 提交失败: {resp.text}")
        return None

def get_logs(trace_id):
    print(f"[Client] 查询日志: {trace_id}")
    # 轮询直到状态不再是 RUNNING
    while True:
        resp = requests.get(f"{BASE_URL}/logs/{trace_id}")
        if resp.status_code == 200:
            data = resp.json()
            status = data['status']
            if status != "RUNNING":
                print(f"[Client] 执行完成，状态: {status}")
                print(f"[Client] 最终结果: {json.dumps(data.get('output'), indent=2, ensure_ascii=False)}")
                print("[Client] 事件日志:")
                for event in data.get('events', []):
                    print(f"  - [{event['timestamp']}] {event['details']}")
                break
            else:
                print("[Client] 仍在运行中...")
                time.sleep(1)
        else:
            print(f"[Client] 获取日志失败: {resp.text}")
            break

def main():
    # 1. 第一次执行 (带参数注入)
    print("--- 测试 1: 首次执行 (带参数注入) ---")
    trace_id = run_workflow(DSL_PATH, {
        "order_id": "ORD-2024-001",
        "user_id": "USER-8888"
    })
    if trace_id:
        get_logs(trace_id)
    
    # 2. 修改 DSL 文件模拟热更新
    print("\n--- 测试 2: 热更新测试 ---")
    print("[Client] 修改 DSL 文件...")
    
    with open(DSL_PATH, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # 修改查询词
    modified_content = original_content.replace("langgraph tutorial", "hot reload test")
    
    with open(DSL_PATH, 'w', encoding='utf-8') as f:
        f.write(modified_content)
        
    time.sleep(1) # 确保文件系统更新时间戳
    
    # 3. 再次执行
    trace_id_2 = run_workflow(DSL_PATH, {})
    if trace_id_2:
        get_logs(trace_id_2)
        
    # 还原文件
    print("\n[Client] 还原 DSL 文件...")
    with open(DSL_PATH, 'w', encoding='utf-8') as f:
        f.write(original_content)

if __name__ == "__main__":
    main()
