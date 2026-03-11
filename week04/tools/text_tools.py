# text_tools.py - 文本处理工具
def tool(name: str, description: str = ""):
    def decorator(func):
        func.is_tool = True
        func.tool_name = name
        func.description = description
        return func
    return decorator


@tool("reverse", "将字符串反转")
def reverse(text: str) -> str:
    return text[::-1]


@tool("uppercase", "将文本转为大写")
def uppercase(text: str) -> str:
    return text.upper()


@tool("word_count", "统计文本字数（按空格分词）")
def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


@tool("strip_spaces", "去除首尾空白")
def strip_spaces(text: str) -> str:
    return text.strip()
