"""
发票开具插件（供热加载）

根据订单号与发票类型开具发票，订单号须为 10～12 位数字。
"""

import re
from langchain_core.tools import tool


def _validate_order_id(order_id: str) -> tuple[bool, str]:
    order_id = (order_id or "").strip()
    if not order_id:
        return False, "订单号不能为空，请输入10到12位数字。"
    if not re.fullmatch(r"\d{10,12}", order_id):
        return False, "订单号不合法，请输入10到12位数字。"
    return True, ""


@tool
def issue_invoice(order_id: str, invoice_type: str = "电子") -> str:
    """
    为指定订单开具发票。支持类型：电子、纸质。
    订单号须为10到12位数字。
    """
    ok, err = _validate_order_id(order_id)
    if not ok:
        return err
    if invoice_type not in ("电子", "纸质"):
        return f"不支持的发票类型：{invoice_type}，请使用「电子」或「纸质」。"
    return (
        f"订单 {order_id} 的{invoice_type}发票已开具。"
        "发票号：INV20260310001，可在「我的-发票」中查看或下载。"
    )
