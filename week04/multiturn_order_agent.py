"""
多轮对话与工具调用

在基础对话（时间推断）基础上，实现「订单查询」「退款申请」等多轮交互，
支持工具自动调用。使用 LangGraph（create_agent）构建：

- 用户说「查订单」→ 追问「请提供订单号」
- 收到订单号后 → 调用 query_order(order_id) 工具
- 返回订单状态与物流信息

依赖：basic_chat_chain 的 get_date_context；通义需 DASHSCOPE_API_KEY。
"""

import re
from datetime import datetime, timedelta

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent


# ---------- 时间上下文（与 basic_chat_chain 一致）----------
def get_date_context() -> dict:
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return {
        "today": today.strftime("%Y年%m月%d日"),
        "yesterday": yesterday.strftime("%Y年%m月%d日"),
    }


# ---------- 订单号校验 ----------
def _validate_order_id(order_id: str) -> tuple[bool, str]:
    order_id = (order_id or "").strip()
    if not order_id:
        return False, "订单号不能为空，请输入10到12位数字。"
    if not re.fullmatch(r"\d{10,12}", order_id):
        return False, "订单号不合法，请输入10到12位数字。"
    return True, ""


# ---------- 工具：query_order ----------
@tool
def query_order(order_id: str) -> str:
    """
    根据订单号查询订单状态与物流信息。
    仅支持10到12位数字的订单号。
    """
    ok, err = _validate_order_id(order_id)
    if not ok:
        return err
    # 模拟后端
    return (
        f"订单号：{order_id}\n"
        "状态：已发货\n"
        "物流：顺丰速运 SF1234567890，当前已到达【北京转运中心】。"
    )


# ---------- 工具：退款申请（占位，多轮可扩展）----------
@tool
def refund_request(order_id: str, reason: str = "") -> str:
    """
    提交退款申请。需要订单号（10～12位数字）和可选退款原因。
    """
    ok, err = _validate_order_id(order_id)
    if not ok:
        return err
    return f"订单 {order_id} 的退款申请已提交，原因：{reason or '未填写'}。预计1～3个工作日处理。"


# ---------- 构建 LangGraph Agent（多轮 + 工具）----------
def build_agent():
    date_ctx = get_date_context()
    system = (
        "你是订单客服助手。当前日期是 {today}，昨天是 {yesterday}；"
        "用户说「今天」「昨天」等时请结合上述日期理解。\n\n"
        "能力与流程：\n"
        "1）订单查询：用户说「查订单」「查一下订单」但未提供订单号时，你必须只回复「请提供订单号（10到12位数字）」；"
        "当用户提供了订单号（本轮或上一轮对话中）时，调用 query_order(order_id) 获取状态与物流，并用自然语言回复。\n"
        "2）退款申请：用户说「退款」「申请退款」时，若未提供订单号则先追问订单号；提供后调用 refund_request(order_id, reason) 并告知已提交。\n"
        "3）其他咨询：礼貌回复并引导到查订单或退款。"
    ).format(**date_ctx)
    llm = ChatTongyi(model_name="qwen-turbo", temperature=0)
    return create_agent(
        model=llm,
        tools=[query_order, refund_request],
        system_prompt=system,
    )


# ---------- 多轮对话：传入历史 messages，返回本轮回复 ----------
def chat(messages: list) -> str:
    """
    多轮对话。messages 为历史消息列表，每项为 {"role": "user"|"assistant", "content": "..."}。
    返回本轮 assistant 的回复文本。
    """
    agent = build_agent()
    lc_messages = []
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    result = agent.invoke({"messages": lc_messages})
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "未获取到回复，请重试。"


# ---------- 演示：查订单 → 追问订单号 → 收到订单号 → 调工具返回 ----------
if __name__ == "__main__":
    agent = build_agent()

    # 第一轮：用户说「查订单」→ 应追问订单号
    print("=== 第 1 轮 ===")
    r1 = agent.invoke({"messages": [HumanMessage(content="查订单")]})
    reply1 = next((m.content for m in reversed(r1["messages"]) if isinstance(m, AIMessage) and m.content), "")
    print("用户: 查订单")
    print("助手:", reply1)
    print()

    # 第二轮：用户提供订单号 → 应调 query_order 并返回状态与物流
    print("=== 第 2 轮 ===")
    r2 = agent.invoke({
        "messages": [
            HumanMessage(content="查订单"),
            AIMessage(content=reply1),
            HumanMessage(content="1234567890"),
        ]
    })
    reply2 = next((m.content for m in reversed(r2["messages"]) if isinstance(m, AIMessage) and m.content), "")
    print("用户: 1234567890")
    print("助手:", reply2)
    print()

    # 可选：用 chat() 封装跑同样流程
    print("=== 使用 chat() 封装 ===")
    history = [
        {"role": "user", "content": "我昨天下的单，想查一下"},
        {"role": "assistant", "content": "好的，请问您的订单号是多少？（10到12位数字）"},
        {"role": "user", "content": "9876543210"},
    ]
    out = chat(history)
    print("历史:", [h["content"][:20] + "..." for h in history])
    print("助手:", out)
