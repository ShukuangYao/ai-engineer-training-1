"""
订单 Agent 服务：健康检查、多轮对话、模型与插件热更新

- GET  /health        健康检查
- POST /chat          多轮对话（body: session_id, message；可选 history）
- POST /reload_plugins 插件热重载
- POST /reload_model  模型热更新（从环境变量重新读取 MODEL_NAME）
"""

import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent

# 内置工具与插件加载
from plugin_loader import load_plugins


# ---------- 配置（模型热更新时重读）----------
def get_model_name() -> str:
    return os.environ.get("MODEL_NAME", "qwen-turbo")


def get_date_context() -> dict:
    t = datetime.now()
    y = t - timedelta(days=1)
    return {"today": t.strftime("%Y年%m月%d日"), "yesterday": y.strftime("%Y年%m月%d日")}


def _validate_order_id(order_id: str) -> tuple[bool, str]:
    order_id = (order_id or "").strip()
    if not order_id:
        return False, "订单号不能为空，请输入10到12位数字。"
    if not re.fullmatch(r"\d{10,12}", order_id):
        return False, "订单号不合法，请输入10到12位数字。"
    return True, ""


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态与物流信息。订单号10到12位数字。"""
    ok, err = _validate_order_id(order_id)
    if not ok:
        return err
    return f"订单号：{order_id}\n状态：已发货\n物流：顺丰速运 SF1234567890，已到达【北京转运中心】。"


@tool
def refund_request(order_id: str, reason: str = "") -> str:
    """提交退款申请。需要订单号（10～12位数字）和可选原因。"""
    ok, err = _validate_order_id(order_id)
    if not ok:
        return err
    return f"订单 {order_id} 退款申请已提交，原因：{reason or '未填写'}。预计1～3个工作日处理。"


# ---------- 全局 Agent（热更新时替换）----------
_agent: Optional[Any] = None
_agent_model_name: Optional[str] = None
_agent_tools_hash: Optional[str] = None


def _build_tools() -> List:
    builtin = [query_order, refund_request]
    plugins = load_plugins()
    return builtin + plugins


def _build_agent(force_rebuild: bool = False) -> Any:
    global _agent, _agent_model_name, _agent_tools_hash
    model_name = get_model_name()
    tools = _build_tools()
    tools_hash = str(sorted(t.name for t in tools))
    if not force_rebuild and _agent is not None and _agent_model_name == model_name and _agent_tools_hash == tools_hash:
        return _agent
    date_ctx = get_date_context()
    system = (
        "你是订单客服助手。当前日期 {today}，昨天 {yesterday}。"
        "用户说「查订单」且未提供订单号时，只回复「请提供订单号（10到12位数字）」；"
        "提供订单号后调用 query_order。用户说「退款」时先追问订单号再调用 refund_request。"
        "用户说「开发票」「开具发票」时调用 issue_invoice（若可用）。"
    ).format(**date_ctx)
    llm = ChatTongyi(model_name=model_name, temperature=0)
    _agent = create_agent(model=llm, tools=tools, system_prompt=system)
    _agent_model_name = model_name
    _agent_tools_hash = tools_hash
    return _agent


# ---------- FastAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _build_agent()
    yield


app = FastAPI(title="Order Agent API", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    message: str = Field(..., description="用户本条消息")
    history: List[dict] = Field(default_factory=list, description="历史消息 [{role, content}]，可选")


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.get("/health")
def health():
    """健康检查，供部署与监控探测。"""
    try:
        _build_agent()
        return {"status": "ok", "model": get_model_name()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """多轮对话。热更新后旧 session_id 仍可用，历史由调用方传入。"""
    agent = _build_agent()
    messages = []
    for h in req.history:
        if h.get("role") == "user":
            messages.append(HumanMessage(content=h.get("content", "")))
        else:
            messages.append(AIMessage(content=h.get("content", "")))
    messages.append(HumanMessage(content=req.message))
    result = agent.invoke({"messages": messages})
    reply = ""
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            reply = m.content
            break
    return ChatResponse(reply=reply or "未获取到回复。", session_id=req.session_id)


@app.post("/reload_plugins")
def reload_plugins_endpoint():
    """插件热重载：重新扫描 plugins 目录并重建 Agent。旧会话不受影响（历史由客户端维护）。"""
    _build_agent(force_rebuild=True)
    return {"status": "ok", "message": "plugins reloaded"}


@app.post("/reload_model")
def reload_model_endpoint():
    """模型热更新：从环境变量重新读取 MODEL_NAME 并重建 Agent。"""
    _build_agent(force_rebuild=True)
    return {"status": "ok", "model": get_model_name()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
