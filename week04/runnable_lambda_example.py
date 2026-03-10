from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# 1. 定义预处理函数
def preprocess_input(input_data):
    """预处理输入数据

    Args:
        input_data: 输入数据，可以是字符串或字典

    Returns:
        预处理后的数据
    """
    if isinstance(input_data, dict):
        # 处理字典类型输入
        topic = input_data.get("topic", "")
        # 去除首尾空格
        topic = topic.strip()
        # 转换为小写
        topic = topic.lower()
        # 添加前缀
        processed_topic = f"关于{topic}的详细信息"
        return {"processed_topic": processed_topic}
    elif isinstance(input_data, str):
        # 处理字符串类型输入
        input_data = input_data.strip()
        input_data = input_data.lower()
        return input_data
    else:
        return input_data

# 2. 定义后处理函数
def postprocess_output(output):
    """后处理输出数据

    Args:
        output: 模型的输出

    Returns:
        后处理后的数据
    """
    if isinstance(output, list):
        # 处理列表类型输出
        return [item.strip() for item in output if item.strip()]
    elif isinstance(output, str):
        # 处理字符串类型输出
        return output.strip()
    else:
        return output

# 3. 初始化组件
parser = CommaSeparatedListOutputParser()
llm = Tongyi(temperature=0)

# 4. 创建提示模板
prompt = PromptTemplate(
    template="请列出5个{processed_topic}的关键点。\n{format_instructions}",
    input_variables=["processed_topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 5. 创建 LCEL 链
chain = (
    # 使用 RunnableLambda 集成预处理函数
    RunnableLambda(preprocess_input)
    # 连接提示模板
    | prompt
    # 连接语言模型
    | llm
    # 连接输出解析器
    | parser
    # 使用 RunnableLambda 集成后处理函数
    | RunnableLambda(postprocess_output)
)

# 6. 测试链
print("=== 测试 LCEL 链 ===")

# 测试1：使用字典输入
test_input1 = {"topic": "  人工智能  "}
result1 = chain.invoke(test_input1)
print(f"输入: {test_input1}")
print(f"输出: {result1}")
print()

# 测试2：使用字符串输入（需要调整链的结构）
simple_chain = (
    RunnableLambda(lambda x: x["input"].strip().lower())
    | PromptTemplate.from_template("请简要介绍{input}")
    | llm
    | RunnableLambda(lambda x: x.strip())
)

test_input2 = "  机器学习  "
result2 = simple_chain.invoke({"input": test_input2})
print(f"输入: '{test_input2}'")
print(f"输出: {result2}")
print()

# 7. 更复杂的预处理示例
def complex_preprocess(input_data):
    """更复杂的预处理函数

    Args:
        input_data: 输入数据

    Returns:
        处理后的提示词和参数
    """
    # 提取主题和要求
    topic = input_data.get("topic", "")
    requirements = input_data.get("requirements", [])

    # 处理主题
    topic = topic.strip().lower()

    # 处理要求
    processed_requirements = []
    for req in requirements:
        if req.strip():
            processed_requirements.append(req.strip())

    # 构建提示词
    prompt_text = f"请介绍{topic}"
    if processed_requirements:
        prompt_text += "，要求：" + "、".join(processed_requirements)

    return {"prompt_text": prompt_text, "topic": topic}

# 创建使用复杂预处理的链
complex_chain = (
    RunnableLambda(complex_preprocess)
    | PromptTemplate.from_template("{prompt_text}")
    | llm
    | RunnableLambda(lambda x: x.strip())
)

# 测试复杂预处理
test_input3 = {
    "topic": "  深度学习  ",
    "requirements": ["通俗易懂", "包含应用场景", "介绍主要算法"]
}
result3 = complex_chain.invoke(test_input3)
print("=== 测试复杂预处理 ===")
print(f"输入: {test_input3}")
print(f"输出: {result3}")
