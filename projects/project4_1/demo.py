# -*- coding: utf-8 -*-
"""
订单查询客服系统 - 多轮对话智能客服
功能：支持文本输入、语音输入、图像上传、订单查询、RAG知识检索、对话持久化
作者：AI助手
版本：1.0
"""

# 标准库导入 - 操作系统、数据库、JSON、时间等基础功能
import os
import sqlite3
import json
import time
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from datetime import datetime
import operator

# 第三方库导入 - Pydantic用于数据验证，DashScope用于阿里云AI服务
from pydantic import BaseModel, Field
import dashscope
from dashscope.audio.asr import Recognition
from http import HTTPStatus

# LangChain 核心库导入 - 用于构建对话系统、工具、提示词模板等
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# LangGraph 导入 - 用于构建状态图和工作流，实现多轮对话
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# ==============================================
# 一、配置部分
# ==============================================

# 数据库文件路径配置
DB_PATH = "orders.db"

# API密钥配置说明
# 说明：实际使用时请在环境变量中设置 DASHSCOPE_API_KEY。
# os.environ["DASHSCOPE_API_KEY"] = "sk-..."
# 如果未检测到密钥，本演示会使用简易规则引擎（Mock LLM）。

# ==============================================
# 二、数据库初始化 (SQLite)
# ==============================================

def setup_database():
    """
    初始化 SQLite 数据库并写入示例订单数据。

    功能说明：
    1. 创建订单表，包含订单号、用户ID、状态、商品、物流信息、创建时间等字段
    2. 检查数据库是否为空，如果为空则插入示例数据
    3. 提交事务并关闭数据库连接

    技术要点：
    - 使用 sqlite3 标准库操作 SQLite 数据库
    - 使用 CREATE TABLE IF NOT EXISTS 避免重复创建表
    - 使用 executemany 批量插入数据提高效率
    - 支持的订单状态：shipped(已发货)、pending_payment(待付款)、delivered(已送达)
    """
    # 连接到 SQLite 数据库，如果文件不存在会自动创建
    conn = sqlite3.connect(DB_PATH)
    # 创建游标对象，用于执行SQL语句
    cursor = conn.cursor()

    # 创建订单表结构
    # order_id: 订单号（主键）
    # user_id: 用户ID
    # status: 订单状态
    # items: 商品信息（文本存储）
    # logistics_info: 物流信息
    # created_at: 创建时间（ISO格式）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        user_id TEXT,
        status TEXT,
        items TEXT,
        logistics_info TEXT,
        created_at TEXT
    )
    ''')

    # 检查数据库中是否已有数据
    cursor.execute('SELECT count(*) FROM orders')
    # 如果订单数为0，说明数据库为空，需要插入示例数据
    if cursor.fetchone()[0] == 0:
        print("正在写入示例订单数据...")
        # 准备示例订单数据，每条数据对应一个订单
        sample_orders = [
            ("12345", "user_001", "shipped", "Wireless Headphones", "Arrived at Beijing Sorting Center", datetime.now().isoformat()),
            ("67890", "user_001", "pending_payment", "Smart Watch", "Waiting for payment", datetime.now().isoformat()),
            ("11223", "user_002", "delivered", "Laptop Stand", "Delivered to locker", datetime.now().isoformat()),
        ]
        # 批量插入示例订单数据
        cursor.executemany('INSERT INTO orders VALUES (?,?,?,?,?,?)', sample_orders)
        # 提交事务，确保数据写入数据库
        conn.commit()

    # LangGraph 对话检查点持久化说明：
    # SqliteSaver 会自动创建所需的检查点表，本演示直接依赖其默认行为。
    # 检查点用于保存对话状态，支持对话中断后恢复和多轮对话上下文管理

    # 关闭数据库连接
    conn.close()

# ==============================================
# 三、RAG 初始化 (知识库)
# ==============================================

def setup_rag_retriever():
    """
    初始化一个用于检索政策知识的简易 RAG 检索器。

    RAG (Retrieval-Augmented Generation) 是检索增强生成技术
    功能说明：
    1. 定义客服政策知识库（退款政策、物流政策等）
    2. 将知识库转换为 Document 对象
    3. 使用向量存储（Vector Store）存储文档
    4. 返回检索器，用于根据用户查询检索相关知识

    技术要点：
    - 使用 FakeEmbeddings 作为演示（生产环境应使用真实的嵌入模型）
    - 使用 InMemoryVectorStore 作为内存向量数据库
    - 支持语义搜索，根据查询的语义相似度检索相关文档
    """
    # 客服政策知识库 - 包含退款、物流、质保、支付等政策
    policies = [
        "退款政策：自签收之日起 7 天内，且商品未拆封，可发起退款申请。",
        "物流政策：满 50 美元免邮，标准配送一般为 3-5 个工作日。",
        "质保政策：电子类商品享受 1 年制造商质保服务。",
        "支付政策：支持信用卡、PayPal 和支付宝。",
        "订单修改：订单状态变为\"已发货\"后不可再修改订单信息。"
    ]

    # 将政策文本转换为 Document 对象
    # Document 是 LangChain 中表示文档的标准格式，包含 page_content 和 metadata
    documents = [Document(page_content=p, metadata={"source": "policy_doc"}) for p in policies]

    # 说明：演示环境使用 FakeEmbeddings，无需外部 API。
    # 生产环境可替换为 OpenAIEmbeddings 或 DashScopeEmbeddings。
    # FakeEmbeddings 生成随机向量，仅用于演示，不具备真实的语义理解能力
    embeddings = FakeEmbeddings(size=768)

    # 使用 InMemoryVectorStore 创建向量存储
    # from_documents 方法会自动将文档转换为向量并存储
    vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)

    # 返回检索器，retriever 可以根据用户查询检索相关文档
    # as_retriever() 返回一个 Retriever 对象，支持 invoke 方法进行检索
    return vectorstore.as_retriever()

# ==============================================
# 四、工具定义 (Tools)
# ==============================================

@tool
def check_order(order_id: str) -> str:
    """
    根据订单号查询订单状态与物流信息。

    这是一个 LangChain 工具（Tool），使用 @tool 装饰器定义
    工具是 Agent 可以调用的外部功能，用于获取实时数据或执行操作

    参数说明：
    - order_id: 订单号，字符串类型

    返回值：
    - 包含订单状态和物流信息的字符串，或者未找到订单的提示信息

    技术要点：
    - 使用 sqlite3 查询数据库
    - 参数化查询，防止 SQL 注入
    - 格式化返回结果，便于 LLM 理解
    """
    # 连接到订单数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 使用参数化查询，防止 SQL 注入攻击
    # ? 是占位符，后面传入的元组会替换这些占位符
    cursor.execute('SELECT status, items, logistics_info FROM orders WHERE order_id = ?', (order_id,))

    # 获取查询结果，fetchone() 返回第一条记录
    result = cursor.fetchone()

    # 关闭数据库连接
    conn.close()

    # 格式化返回结果
    if result:
        # 如果找到订单，提取订单状态、商品信息和物流信息
        status, items, logistics = result
        # 格式化输出，包含订单号、商品、状态和物流信息
        return f"订单 {order_id}（{items}）：当前状态为『{status}』。物流信息：{logistics}。"
    else:
        # 如果未找到订单，返回提示信息
        return f"未查询到订单 {order_id}，请检查订单号是否正确。"

@tool
def search_policy(query: str) -> str:
    """
    查询与客服政策相关的知识（退款、物流等）。

    这是一个 RAG 检索工具，用于从知识库中检索相关政策信息

    参数说明：
    - query: 用户的查询文本，字符串类型

    返回值：
    - 相关政策文档的拼接字符串

    技术要点：
    - 重新初始化检索器（演示用，生产环境应复用全局检索器）
    - 使用 RAG 检索技术，根据查询的语义相似度检索相关文档
    - 将检索到的多个文档内容拼接成一个字符串返回给 LLM
    """
    # 说明：真实环境可复用全局检索器；为保证工具无状态，这里简单重新初始化。
    # 初始化 RAG 检索器
    retriever = setup_rag_retriever()

    # 调用检索器，获取与查询相关的文档
    # invoke 方法接收查询字符串，返回 Document 列表
    docs = retriever.invoke(query)

    # 将检索到的文档内容拼接成一个字符串
    # doc.page_content 是文档的实际内容
    return "\n".join([doc.page_content for doc in docs])

# ==============================================
# 五、输入处理 (ASR/OCR)
# ==============================================

def process_audio_input(file_path: str) -> str:
    """
    使用 Qwen/Dashscope 进行语音转写（演示中为模拟）。

    ASR (Automatic Speech Recognition) 是自动语音识别技术
    功能说明：
    1. 检查是否配置了 API 密钥
    2. 如果没有密钥，返回模拟结果用于演示
    3. 如果有密钥，调用真实的 ASR 服务进行语音转写

    参数说明：
    - file_path: 音频文件路径，支持 .wav, .mp3 等格式

    返回值：
    - 转写后的文本字符串

    技术要点：
    - 使用 DashScope ASR 服务（阿里云通义千问的语音识别服务）
    - 支持多种音频格式
    - 包含错误处理逻辑
    """
    print(f"[系统] 正在处理音频文件：{file_path}")

    # 从环境变量获取 API 密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        # 如果没有检测到 API Key，返回模拟 ASR 结果用于演示
        print("[系统] 未检测到 API Key，返回模拟 ASR 结果。")
        return "查订单 12345" # 模拟结果

    try:
        # 真实实现：使用 DashScope ASR
        # task = dashscope.audio.asr.Recognition.call(...)
        # 演示保持为模拟，返回固定文本。
        # 注意：实际使用时需要取消注释上面的代码，并配置正确的参数
        return "查订单 12345"
    except Exception as e:
        # 捕获异常并返回错误信息
        return f"Error in ASR: {str(e)}"

def process_image_input(file_path: str) -> str:
    """
    使用 Qwen-VL 进行 OCR（演示中为模拟）。

    OCR (Optical Character Recognition) 是光学字符识别技术
    功能说明：
    1. 检查是否配置了 API 密钥
    2. 如果没有密钥，返回模拟结果用于演示
    3. 如果有密钥，调用真实的 OCR 服务识别图片中的文字

    参数说明：
    - file_path: 图片文件路径，支持 .jpg, .png 等格式

    返回值：
    - 识别后的文本字符串

    技术要点：
    - 使用 Qwen-VL 多模态模型（阿里云通义千问的视觉语言模型）
    - 支持从图片中识别文字（OCR）和理解图片内容
    - 包含错误处理逻辑
    """
    print(f"[系统] 正在处理图片文件：{file_path}")

    # 从环境变量获取 API 密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        # 如果没有检测到 API Key，返回模拟 OCR 结果用于演示
        print("[系统] 未检测到 API Key，返回模拟 OCR 结果。")
        return "图片中订单号似乎是 67890" # 模拟结果

    try:
        # 真实实现：使用 DashScope 多模态
        # messages = [{...}]
        # response = dashscope.MultiModalConversation.call(model='qwen-vl-max', messages=messages)
        # 演示保持为模拟，返回固定文本。
        # 注意：实际使用时需要取消注释上面的代码，并配置正确的参数
        return "图片中订单号似乎是 67890"
    except Exception as e:
        # 捕获异常并返回错误信息
        return f"Error in OCR: {str(e)}"

# ==============================================
# 六、LangGraph 状态与节点
# ==============================================

class AgentState(TypedDict):
    """
    定义 Agent 的状态结构。

    AgentState 是一个 TypedDict，用于描述 LangGraph 中状态的数据结构
    状态在工作流的节点之间传递，记录对话的当前状态

    字段说明：
    - messages: 对话历史消息列表，使用 operator.add 实现消息追加
    - order_id: 当前查询的订单号，可选字段（可能为空）

    技术要点：
    - 使用 TypedDict 定义类型安全的状态结构
    - 使用 Annotated 为 messages 字段添加注解 operator.add
    - operator.add 表示当多个节点更新 messages 时，将新消息追加到列表中
    """
    # messages: 对话历史消息列表
    # Annotated 用于添加元数据，operator.add 表示合并策略为追加
    messages: Annotated[List[BaseMessage], operator.add]
    # order_id: 当前查询的订单号，Optional 表示可以为 None
    order_id: Optional[str]

# LLM 初始化
# 如果存在密钥则使用真实模型，否则使用简易规则引擎（Mock）。
# 首先尝试从环境变量获取 API 密钥（支持 DashScope 或 OpenAI）
api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")

if api_key:
    # 使用 ChatOpenAI（如需对接 Qwen 可设置兼容的 base_url）；此处默认使用 OpenAI。
    # model="gpt-4o"：使用 GPT-4o 模型
    # temperature=0：设置温度为 0，使输出更加确定性和保守
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
else:
    # 未检测到密钥，使用规则引擎模拟。
    # MockLLM 是一个简单的模拟 LLM，用于演示目的
    print("[系统] 未检测到 LLM API Key，使用规则引擎（Mock）进行演示。")

    class MockLLM:
        """
        模拟 LLM 类，用于演示目的。

        这个类模拟了 LLM 的基本行为，包括：
        1. invoke 方法：处理消息并返回响应
        2. bind_tools 方法：绑定工具（返回 self，不做实际绑定）

        功能说明：
        - 根据用户输入的关键词进行简单的规则匹配
        - 支持模拟工具调用（tool_calls）
        - 支持模拟正常对话
        """
        def invoke(self, messages):
            """
            模拟 LLM 的 invoke 方法。

            参数说明：
            - messages: 消息列表

            返回值：
            - AIMessage 对象，包含响应内容和工具调用信息
            """
            # 获取最后一条用户消息的内容，并转换为小写
            last_msg = messages[-1].content.lower()

            # 检查是否包含订单查询相关的关键词
            if "查订单" in last_msg or "订单" in last_msg or "12345" in last_msg or "67890" in last_msg:
                # 需要调用工具时返回 tool_calls
                if "12345" in last_msg:
                    # 模拟调用 check_order 工具查询订单 12345
                    return AIMessage(content="", tool_calls=[{"name": "check_order", "args": {"order_id": "12345"}, "id": "call_1"}])
                elif "67890" in last_msg:
                    # 模拟调用 check_order 工具查询订单 67890
                    return AIMessage(content="", tool_calls=[{"name": "check_order", "args": {"order_id": "67890"}, "id": "call_2"}])
                else:
                    # 如果没有具体的订单号，提示用户提供订单号
                    return AIMessage(content="请提供订单号。")
            elif "政策" in last_msg or "退款" in last_msg:
                # 检查是否包含政策查询相关的关键词
                # 模拟调用 search_policy 工具查询政策
                 return AIMessage(content="", tool_calls=[{"name": "search_policy", "args": {"query": last_msg}, "id": "call_3"}])
            else:
                # 其他情况，返回默认的问候语
                return AIMessage(content="我可以帮您查询订单或解答政策相关问题。")

        def bind_tools(self, tools):
            """
            模拟 LLM 的 bind_tools 方法。

            参数说明：
            - tools: 工具列表

            返回值：
            - self，不做实际绑定
            """
            return self

    # 创建 MockLLM 实例
    llm = MockLLM()

# 绑定工具
# 将定义好的工具（check_order 和 search_policy）绑定到 LLM
# 这样 LLM 就知道可以调用哪些工具了
tools = [check_order, search_policy]
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: AgentState):
    """
    聊天机器人节点：根据历史消息决定要采取的动作。

    这是 LangGraph 中的一个节点，负责处理用户输入并生成响应

    参数说明：
    - state: 当前的 AgentState，包含对话历史等信息

    返回值：
    - 包含更新后的状态的字典，这里只更新了 messages 字段

    技术要点：
    - 使用已经绑定工具的 LLM（llm_with_tools）
    - LLM 会根据对话历史决定是否需要调用工具
    - 如果需要调用工具，返回的 AIMessage 中会包含 tool_calls
    - 如果不需要调用工具，直接返回 AIMessage
    """
    # 调用 LLM 处理对话历史，生成响应
    response = llm_with_tools.invoke(state["messages"])
    # 返回更新后的状态，将新消息追加到 messages 列表中
    return {"messages": [response]}

def input_processing_node(state: AgentState):
    """
    输入预处理节点：对输入进行预处理（文本/音频/图片）。

    这是 LangGraph 中的一个节点，负责在聊天机器人处理前对输入进行预处理

    功能说明：
    1. 检查最后一条消息是否是用户消息
    2. 如果是文件路径（音频或图片），进行相应的处理
    3. 将处理结果转换为新的用户消息
    4. 返回更新后的状态

    参数说明：
    - state: 当前的 AgentState，包含对话历史等信息

    返回值：
    - 包含更新后的状态的字典，可能更新了 messages 字段

    技术要点：
    - 检查消息类型（HumanMessage）
    - 根据文件扩展名判断文件类型（.wav/.mp3 是音频，.jpg/.png 是图片）
    - 调用相应的处理函数（process_audio_input 或 process_image_input）
    - 返回空更新（{"messages": []}）以避免 InvalidUpdateError
    """
    # 说明：真实系统可拆分更细；此处假定用户消息已加入状态，若是文件路径则进行相应处理。

    # 获取最后一条消息
    last_message = state["messages"][-1]

    # 检查最后一条消息是否是用户消息
    if isinstance(last_message, HumanMessage):
        # 获取消息内容
        content = last_message.content

        # 检查是否是音频文件（以 .wav 或 .mp3 结尾）
        if content.endswith(".wav") or content.endswith(".mp3"):
            # 调用音频处理函数进行语音转写
            text = process_audio_input(content)
            # 返回更新后的状态，将转写结果作为新的用户消息
            return {"messages": [HumanMessage(content=f"音频转写：{text}")]}
        # 检查是否是图片文件（以 .jpg 或 .png 结尾）
        elif content.endswith(".jpg") or content.endswith(".png"):
            # 调用图片处理函数进行 OCR 识别
            text = process_image_input(content)
            # 返回更新后的状态，将识别结果作为新的用户消息
            return {"messages": [HumanMessage(content=f"图片识别：{text}")]}

    # 返回空更新，明确避免 InvalidUpdateError
    # 如果没有需要处理的内容，返回空消息列表
    return {"messages": []}

# 定义图工作流
# 使用 StateGraph 创建状态图，指定状态类型为 AgentState
workflow = StateGraph(AgentState)

# 添加节点到工作流
# 节点是工作流中的处理单元
workflow.add_node("input_proc", input_processing_node)  # 输入处理节点
workflow.add_node("chatbot", chatbot_node)              # 聊天机器人节点
workflow.add_node("tools", ToolNode(tools))              # 工具执行节点（使用 LangGraph 预构建的 ToolNode）

# 添加边连接节点
# 边定义了节点之间的执行顺序
workflow.add_edge(START, "input_proc")  # 从 START 节点开始，先执行 input_proc
workflow.add_edge("input_proc", "chatbot")  # 执行完 input_proc 后，执行 chatbot

def route_tools(state: AgentState):
    """
    条件路由函数：判断是否需要调用工具。

    这是 LangGraph 中的条件路由函数，根据当前状态决定下一步执行哪个节点

    功能说明：
    1. 从状态中获取最后一条 AI 消息
    2. 检查消息中是否包含工具调用（tool_calls）
    3. 如果包含工具调用，路由到 tools 节点
    4. 如果不包含工具调用，路由到 END 节点（结束）

    参数说明：
    - state: 当前的 AgentState，包含对话历史等信息

    返回值：
    - 字符串，表示要路由到的节点名称（"tools" 或 END）

    技术要点：
    - 支持多种状态类型（列表、字典、BaseMessage）
    - 使用 hasattr 检查对象是否有 tool_calls 属性
    - 检查 tool_calls 的长度是否大于 0
    """
    # 从状态中获取最后一条 AI 消息
    if isinstance(state, list):
        # 如果状态是列表，直接取最后一条
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get("messages", [])):
        # 如果状态是字典，从 messages 字段中取最后一条
        ai_message = messages[-1]
    elif isinstance(state, BaseMessage):
        # 如果状态本身就是 BaseMessage，直接使用
        ai_message = state
    else:
        # 其他情况，抛出异常
        raise ValueError(f"未在状态中找到消息，无法进行工具路由：{state}")

    # 检查是否有工具调用
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        # 如果有工具调用，路由到 tools 节点
        return "tools"
    # 如果没有工具调用，路由到 END 节点（结束）
    return END

# 添加条件边
# 条件边根据条件路由函数的返回值决定执行哪个节点
workflow.add_conditional_edges("chatbot", route_tools)

# 添加工具执行后的边
# 执行完 tools 节点后，回到 chatbot 节点继续处理
workflow.add_edge("tools", "chatbot")

# 配置检查点持久化
# 使用 SqliteSaver 将对话状态保存到 SQLite 数据库
# 这样可以实现对话中断后恢复和多轮对话上下文管理
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# 编译工作流
# 将工作流编译为可执行的应用
# checkpointer=memory 表示使用 SqliteSaver 作为检查点持久化
app = workflow.compile(checkpointer=memory)

# ==============================================
# 七、主流程执行与测试
# ==============================================

def run_demo():
    """
    运行订单查询客服系统的演示。

    功能说明：
    1. 初始化数据库
    2. 配置对话会话 ID（用于状态持久化）
    3. 模拟用户输入场景（文本、音频、图片）
    4. 执行工作流并输出结果

    技术要点：
    - 使用 config 参数指定 thread_id，实现对话隔离
    - 使用 stream 方法以流式方式运行工作流
    - 使用 stream_mode="values" 获取状态更新
    """
    print("--- 订单查询客服演示 ---")

    # 初始化数据库
    setup_database()

    # 对话会话 ID（用于状态持久化）
    # 不同的 thread_id 对应不同的对话会话
    config = {"configurable": {"thread_id": "session_1"}}

    # 模拟用户输入场景
    scenarios = [
        "你好，我要查订单。",           # 文本输入 - 问候
        "订单号是 12345。",             # 文本输入 - 查询订单
        "如果不喜欢可以退货吗？",          # 文本输入 - RAG 检索政策
        "audio_sample.wav",             # 模拟音频输入
        "order_image.jpg"               # 模拟图片输入
    ]

    # 遍历每个场景，模拟用户输入
    for user_input in scenarios:
        print(f"\n用户：{user_input}")

        # 判断是文件路径还是文本
        # 这里直接使用 user_input，input_processing_node 会处理文件路径
        msg_content = user_input

        # 以流式方式运行图
        # stream 方法返回一个生成器，逐个产生状态更新事件
        events = app.stream(
            {"messages": [HumanMessage(content=msg_content)]},  # 输入状态
            config,                                                # 配置（包含 thread_id）
            stream_mode="values"                                   # 流模式：values 表示获取状态更新
        )

        # 处理流式输出的事件
        for event in events:
            if "messages" in event:
                # 获取最后一条消息
                last_msg = event["messages"][-1]
                # 检查是否是 AI 消息
                if isinstance(last_msg, AIMessage):
                    # 输出客服回复
                    print(f"客服：{last_msg.content}")
                    # 检查是否有工具调用
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        # 输出工具调用信息
                         print(f"  [工具调用]：{last_msg.tool_calls[0]['name']}")

# 主程序入口
if __name__ == "__main__":
    # 运行演示
    run_demo()
