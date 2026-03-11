"""
LangGraph 订单查询 Agent

用户说「查一下订单1234567890」时，Agent 识别意图并调用 query_order 工具，
返回订单状态与物流信息的自然语言回复。

要求实现：
- 工具支持参数格式校验：订单号须为 10～12 位数字，否则返回明确错误提示
- 调用失败时最多重试 3 次（在 query_order 内部对后端请求重试）
- 查询结果缓存 5 分钟（同订单号 5 分钟内直接返回缓存）
- 输入非法时返回明确错误提示（如「订单号不合法，请输入10到12位数字。」）

依赖：langchain, langchain-community, langchain-core；通义千问需配置 DASHSCOPE_API_KEY。
"""

import re
import time
from typing import Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent


# ---------- 缓存：5 分钟 TTL ----------
_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 5 * 60  # 5 分钟


def _get_cached(order_id: str) -> Optional[str]:
    if order_id not in _CACHE:
        return None
    value, expiry = _CACHE[order_id]
    if time.time() > expiry:
        del _CACHE[order_id]
        return None
    return value


def _set_cached(order_id: str, result: str) -> None:
    _CACHE[order_id] = (result, time.time() + _CACHE_TTL)


# ---------- 订单号校验 ----------
def _validate_order_id(order_id: str) -> tuple[bool, str]:
    """校验订单号：10～12 位数字。返回 (是否合法, 错误提示)。"""
    order_id = (order_id or "").strip()
    if not order_id:
        return False, "订单号不能为空，请输入10到12位数字。"
    if not re.fullmatch(r"\d{10,12}", order_id):
        return False, "订单号不合法，请输入10到12位数字。"
    return True, ""


# ---------- 模拟后端查询（可替换为真实 API）----------
# 用于验证「最多重试 3 次」：订单 1111111111 前 2 次调用失败，第 3 次成功
_fail_until_success_count: dict[str, int] = {}


def _fetch_order_backend(order_id: str) -> str:
    """模拟：根据订单号查询状态与物流。真实环境可改为 HTTP/DB 调用。"""
    # 订单 0000000000：始终失败，用于验证「重试 3 次后返回失败提示」
    if order_id == "0000000000":
        raise RuntimeError("模拟后端超时")
    # 订单 1111111111：前 2 次失败、第 3 次成功，用于验证「确实重试到第 3 次才成功」
    if order_id == "1111111111":
        n = _fail_until_success_count.get(order_id, 0)
        _fail_until_success_count[order_id] = n + 1
        if n < 2:
            raise RuntimeError(f"模拟后端暂时不可用（第 {n + 1} 次调用失败）")
    return (
        f"订单号：{order_id}\n"
        "状态：已发货\n"
        "物流：顺丰速运 SF1234567890，当前已到达【北京转运中心】。"
    )


# ---------- 工具：带校验、缓存、重试的 query_order ----------
@tool
def query_order(order_id: str) -> str:
    """
    根据订单号查询订单状态与物流信息。
    仅支持10到12位数字的订单号，否则返回错误提示。
    """
    # 1. 参数校验，非法直接返回明确错误（不重试）
    ok, err = _validate_order_id(order_id)
    if not ok:
        return err

    # 2. 查缓存
    cached = _get_cached(order_id)
    if cached is not None:
        return "[缓存] " + cached

    # 3. 最多重试 3 次
    last_error: Optional[str] = None
    for attempt in range(3):
        try:
            result = _fetch_order_backend(order_id)
            _set_cached(order_id, result)
            return result
        except Exception as e:
            last_error = str(e)
            if attempt < 2:
                time.sleep(0.2)  # 简单退避

    return f"查询失败（已重试3次）：{last_error}。请稍后再试或联系客服。"


# ---------- 构建 LangGraph Agent ----------
def build_order_agent():
    """创建订单查询 Agent（底层为 LangGraph）。"""
    llm = ChatTongyi(model_name="qwen-turbo", temperature=0)
    agent = create_agent(
        model=llm,
        tools=[query_order],
        system_prompt=(
            "你是订单查询助手。当用户要「查订单」「查一下订单xxx」时，"
            "你必须调用 query_order 工具，传入用户提到的订单号（仅数字部分），"
            "然后将工具返回的订单状态和物流信息用自然语言整理后回复用户。"
            "若工具返回错误提示（如订单号不合法），直接如实告诉用户。"
        ),
    )
    return agent


# ---------- 主流程：识别意图 → 调工具 → 自然语言回复 ----------
def run_query(user_input: str) -> str:
    """
    用户输入一句话（如「查一下订单1234567890」），返回自然语言回复。
    """
    agent = build_order_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)],
    })
    # 取最后一条 AI 回复内容
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "未获取到回复，请重试。"


# ---------- 命令行与示例 ----------
if __name__ == "__main__":
    # 正常查询
    print("【正常】查一下订单1234567890")
    print(run_query("查一下订单1234567890"))
    print()

    # 非法订单号：非数字
    print("【非法】查一下订单 abc")
    print(run_query("查一下订单 abc"))
    print()

    # 非法订单号：长度不符
    print("【非法】查一下订单 123")
    print(run_query("查一下订单 123"))
    print()

    # 二次查询同一订单（走缓存）
    print("【缓存】再次查订单1234567890")
    print(run_query("查一下订单1234567890"))
    print()

    # ---------- 验证「最多重试 3 次」---------
    # 直接调工具看原始返回值，便于核对「已重试3次」「第3次成功」
    print("【工具直调】query_order('0000000000') 原始返回（应含「已重试3次」）：")
    print("  ", query_order.invoke({"order_id": "0000000000"}))
    print()

    print("【工具直调】query_order('1111111111') 第1次（前2次失败，第3次才成功）：")
    print("  ", query_order.invoke({"order_id": "1111111111"}))
    print()

    # 再通过 Agent 走一遍，确认端到端行为
    print("【重试上限】查订单 0000000000（Agent 回复，预期体现「失败/重试」）")
    print(run_query("查一下订单0000000000"))
    print()

    print("【重试成功】查订单 1111111111（前2次失败第3次成功，预期：返回物流信息）")
    print(run_query("查一下订单1111111111"))
