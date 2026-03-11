"""
调用智能客服 Agent 服务的 HTTP 客户端

用法:
    from smart_customer_service.client import AgentClient
    client = AgentClient(base_url="http://127.0.0.1:8000")
    print(client.health())
    r = client.chat("my-session", "查订单", history=[])
    print(r["reply"])
"""

import os
from typing import Any, List

import httpx


class AgentClient:
    """智能客服 Agent API 客户端。"""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.environ.get("AGENT_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")

    def _post(self, path: str, json: dict) -> dict:
        with httpx.Client(timeout=60.0) as c:
            r = c.post(f"{self.base_url}{path}", json=json)
            r.raise_for_status()
            return r.json()

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(f"{self.base_url}{path}")
            r.raise_for_status()
            return r.json()

    def health(self) -> dict:
        """GET /health 健康检查。"""
        return self._get("/health")

    def chat(
        self,
        session_id: str,
        message: str,
        history: List[dict] | None = None,
    ) -> dict:
        """
        POST /chat 多轮对话。
        history: [{"role":"user"|"assistant","content":"..."}, ...]
        返回: {"reply": "...", "session_id": "..."}
        """
        return self._post("/chat", {
            "session_id": session_id,
            "message": message,
            "history": history or [],
        })

    def reload_plugins(self) -> dict:
        """POST /reload_plugins 插件热重载。"""
        return self._post("/reload_plugins", {})

    def reload_model(self) -> dict:
        """POST /reload_model 模型热更新。"""
        return self._post("/reload_model", {})


def main():
    """命令行 demo：健康检查 + 推断昨天 + 查订单 + 退货 多轮对话。"""
    client = AgentClient()
    print("健康检查:", client.health())
    session = "cli-session-1"

    # 推断昨天 demo（结合当前时间理解「昨天」）
    print("\n--- 推断昨天 ---")
    t1 = client.chat("cli-session-date", "我昨天下的单，想查一下", [])
    print("用户: 我昨天下的单，想查一下")
    print("助手:", t1["reply"])

    # 查订单 demo
    print("\n--- 查订单 ---")
    r1 = client.chat(session, "查订单", [])
    print("用户: 查订单")
    print("助手:", r1["reply"])
    history = [
        {"role": "user", "content": "查订单"},
        {"role": "assistant", "content": r1["reply"]},
    ]
    r2 = client.chat(session, "1234567890", history)
    print("用户: 1234567890")
    print("助手:", r2["reply"])

    # 退货/退款 demo
    print("\n--- 退货/退款 ---")
    session_refund = "cli-session-refund"
    h1 = client.chat(session_refund, "我要退货", [])
    print("用户: 我要退货")
    print("助手:", h1["reply"])
    history_refund = [
        {"role": "user", "content": "我要退货"},
        {"role": "assistant", "content": h1["reply"]},
    ]
    h2 = client.chat(session_refund, "订单号 9876543210，不想要了", history_refund)
    print("用户: 订单号 9876543210，不想要了")
    print("助手:", h2["reply"])

    # 开发票 demo：订单号不合法 / 发票类型不支持 / 正常开发票
    print("\n--- 开发票 ---")
    session_inv = "cli-session-invoice"
    # 1) 订单号不合法（少于 10 位）
    inv1 = client.chat(session_inv, "给订单号 123 开电子发票", [])
    print("用户: 给订单号 123 开电子发票（订单号不合法）")
    print("助手:", inv1["reply"])
    # 2) 发票类型不支持
    inv2 = client.chat(session_inv + "-2", "订单号 1234567890 开专票", [])
    print("用户: 订单号 1234567890 开专票（类型不支持）")
    print("助手:", inv2["reply"])
    # 3) 正常开发票
    inv3 = client.chat(session_inv + "-3", "订单号 1234567890 开电子发票", [])
    print("用户: 订单号 1234567890 开电子发票（正常）")
    print("助手:", inv3["reply"])


if __name__ == "__main__":
    main()
