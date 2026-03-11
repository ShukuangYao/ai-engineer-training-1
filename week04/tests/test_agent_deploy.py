"""
自动化测试：发票开具插件、健康检查、热更新后旧会话不受影响

运行：pytest week04/tests/test_agent_deploy.py -v
或：python -m pytest week04/tests/test_agent_deploy.py -v
"""

import os
import sys

import pytest

# 保证可导入 week04 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- 1. 发票开具插件功能正确性 ----------
def test_invoice_plugin_loaded():
    """插件目录下应能加载到 issue_invoice 工具。"""
    from plugin_loader import load_plugins
    tools = load_plugins()
    names = [t.name for t in tools]
    assert "issue_invoice" in names, f"期望加载到 issue_invoice，当前工具: {names}"


def test_invoice_plugin_invoke_success():
    """发票开具：合法订单号应返回包含「发票已开具」和订单号。"""
    from plugin_loader import load_plugins
    tools = load_plugins()
    inv = next((t for t in tools if t.name == "issue_invoice"), None)
    assert inv is not None
    out = inv.invoke({"order_id": "1234567890", "invoice_type": "电子"})
    assert "发票已开具" in out
    assert "1234567890" in out


def test_invoice_plugin_invoke_invalid_order():
    """发票开具：非法订单号应返回明确错误。"""
    from plugin_loader import load_plugins
    tools = load_plugins()
    inv = next((t for t in tools if t.name == "issue_invoice"), None)
    assert inv is not None
    out = inv.invoke({"order_id": "abc", "invoice_type": "电子"})
    assert "不合法" in out or "10" in out


# ---------- 2. 健康检查接口 ----------
def test_health_endpoint():
    """/health 应返回 200 且 status ok（通过 TestClient 调 FastAPI）。"""
    from fastapi.testclient import TestClient
    from agent_server import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


# ---------- 3. 热更新后旧会话不受影响 ----------
def test_session_after_plugin_reload():
    """热重载插件后，用同一会话（带历史）再发一条消息，应正常返回且无异常。"""
    from fastapi.testclient import TestClient
    from agent_server import app, _build_agent
    client = TestClient(app)
    session_id = "test-session-reload"

    # 第一轮：查订单
    r1 = client.post("/chat", json={
        "session_id": session_id,
        "message": "查订单",
        "history": [],
    })
    assert r1.status_code == 200
    reply1 = r1.json().get("reply", "")
    assert "订单号" in reply1 or "提供" in reply1

    # 触发插件热重载
    r_reload = client.post("/reload_plugins")
    assert r_reload.status_code == 200

    # 第二轮：同一会话，带历史，发订单号
    r2 = client.post("/chat", json={
        "session_id": session_id,
        "message": "1234567890",
        "history": [
            {"role": "user", "content": "查订单"},
            {"role": "assistant", "content": reply1},
        ],
    })
    assert r2.status_code == 200
    reply2 = r2.json().get("reply", "")
    # 应返回订单/物流相关，证明旧会话上下文有效且新 agent 正常
    assert len(reply2) > 0
    assert "1234567890" in reply2 or "订单" in reply2 or "物流" in reply2
