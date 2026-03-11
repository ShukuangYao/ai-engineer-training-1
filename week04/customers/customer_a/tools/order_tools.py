# customer_a/tools/order_tools.py - 客户 A 专属：订单与报表
def tool(name: str, description: str = ""):
    def decorator(func):
        func.is_tool = True
        func.tool_name = name
        func.description = description
        return func
    return decorator


@tool("query_order", "根据订单号查询订单状态（客户A）")
def query_order(order_id: str) -> str:
    # 模拟：实际可接数据库或 API
    return f"[客户A] 订单 {order_id} 状态: 已发货"


@tool("generate_report", "生成客户A的日报摘要")
def generate_report(date: str) -> str:
    return f"[客户A] {date} 日报: 订单 12 笔，金额 ￥8,600"
