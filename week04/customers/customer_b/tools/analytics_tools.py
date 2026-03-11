# customer_b/tools/analytics_tools.py - 客户 B 专属：简单分析
def tool(name: str, description: str = ""):
    def decorator(func):
        func.is_tool = True
        func.tool_name = name
        func.description = description
        return func
    return decorator


@tool("sum_list", "对数字列表求和（客户B）")
def sum_list(numbers: str) -> str:
    # 输入如 "1,2,3" 或 "1 2 3"
    parts = numbers.replace(",", " ").split()
    total = sum(float(x) for x in parts if x.strip())
    return f"[客户B] 合计: {total}"


@tool("avg_list", "对数字列表求平均（客户B）")
def avg_list(numbers: str) -> str:
    parts = numbers.replace(",", " ").split()
    vals = [float(x) for x in parts if x.strip()]
    avg = sum(vals) / len(vals) if vals else 0
    return f"[客户B] 平均值: {avg}"
