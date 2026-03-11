# math_tools.py
def tool(name: str, description: str = ""):
    def decorator(func):
        func.is_tool = True
        func.tool_name = name
        func.description = description
        return func
    return decorator

@tool("add", "加法")
def add(a: int, b: int) -> int:
    return a + b

@tool("multiply", "乘法")
def multiply(a: int, b: int) -> int:
    return a * b

MAX_INPUT = 100  # 普通变量，不是工具