import requests
import json
import time

class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test_user_{int(time.time())}"
    
    def send_message(self, message: str) -> dict:
        """发送消息"""
        url = f"{self.base_url}/chat"
        payload = {
            "session_id": self.session_id,
            "message": message
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_history(self) -> dict:
        """获取历史记录"""
        url = f"{self.base_url}/session/{self.session_id}/history"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def clear_session(self) -> dict:
        """清除会话"""
        url = f"{self.base_url}/session/{self.session_id}"
        try:
            response = requests.delete(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def main():
    """测试主函数"""
    client = ChatClient()
    
    print(f"开始测试对话服务 (会话ID: {client.session_id})")
    print("=" * 50)
    
    # 测试对话
    test_messages = [
        "你好",
        "你能做什么？",
        "请介绍一下你自己",
        "谢谢你的帮助"
    ]
    
    for message in test_messages:
        print(f" 用户: {message}")
        
        result = client.send_message(message)
        if "error" in result:
            print(f" 错误: {result['error']}")
        else:
            print(f" 助手: {result['reply']}")
        
        print("-" * 30)
        time.sleep(1)
    
    # 获取历史记录
    print("\n 获取对话历史:")
    history = client.get_history()
    if "error" not in history:
        for i, record in enumerate(history.get("history", []), 1):
            print(f"{i}. 用户: {record['user_message']}")
            print(f"   助手: {record['bot_reply']}")
            print(f"   时间: {record['timestamp']}")
    
    # 清除会话
    print(f"\n 清除会话: {client.clear_session()}")

if __name__ == "__main__":
    main()