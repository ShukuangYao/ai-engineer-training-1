import json

def search_tool(args):
    """模拟搜索工具。"""
    query = args.get("query", "")
    print(f"  [工具] 正在执行搜索工具，查询词: '{query}'")
    # 模拟结果
    if "langgraph" in query.lower():
        return {"results": ["LangGraph 是一个用于构建有状态、多角色 LLM 应用程序的库。"]}
    elif "weather" in query.lower():
        return {"results": ["搜索中没有天气预报，请使用天气工具。"]}
    else:
        return {"results": [f"关于 {query} 的结果"]}

def calculator_tool(args):
    """模拟计算器工具。"""
    expression = args.get("expression", "")
    print(f"  [工具] 正在执行计算器工具，表达式: '{expression}'")
    try:
        # 在生产环境中这样做很危险，但对于这个 DSL 演示是可以的
        # 禁用内置函数以增加安全性
        result = eval(expression, {"__builtins__": None}, {})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

def weather_tool(args):
    """模拟天气工具。"""
    location = args.get("location", "")
    print(f"  [工具] 正在执行天气工具，地点: '{location}'")
    return {"temperature": "25°C", "condition": "晴朗", "location": location}

def user_input_tool(args):
    """模拟获取用户输入（针对非交互式执行进行了模拟）。"""
    prompt = args.get("prompt", "")
    print(f"  [工具] 请求用户输入: '{prompt}'")
    # 在真实场景中，这可能会暂停执行或等待输入。
    # 这里我们根据提示返回一个模拟值或默认值。
    return {"input": "模拟用户输入"}

TOOL_REGISTRY = {
    "search_tool": search_tool,
    "calculator_tool": calculator_tool,
    "weather_tool": weather_tool,
    "user_input_tool": user_input_tool
}

def get_tool(name):
    """根据名称获取工具函数。"""
    return TOOL_REGISTRY.get(name)
