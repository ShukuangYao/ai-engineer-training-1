"""
问答智能代理模块

基于 LangChain 框架构建的多任务问答助手，支持：
1. 自然语言对话
2. 天气查询（通过高德地图API）
3. 信息搜索（通过Tavily搜索API）

核心设计：
- 使用 LangChain 的 LCEL (LangChain Expression Language) 语法
- 采用 prompt | llm | output 管道模式
- 支持工具自动调用（Function Calling）
- 维护对话历史记录

技术栈：
- LangChain: 用于构建对话链和工具集成
- DeepSeek API: 作为底层大语言模型（兼容 OpenAI API 格式）
- Pydantic: 用于工具参数验证和类型检查
"""

import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# 禁用 LangChain 追踪（避免不必要的日志输出）
# 如果需要在 LangSmith 中追踪，可以将这些设置为 "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

# LangChain 核心组件导入
from langchain_openai import ChatOpenAI  # 注意：虽然类名是 ChatOpenAI，但实际配置为使用 DeepSeek API
from langchain_core.prompts import ChatPromptTemplate  # 提示词模板，用于构建对话提示
from langchain_core.output_parsers import StrOutputParser  # 输出解析器，将 LLM 输出转换为字符串
from langchain_core.runnables import RunnablePassthrough  # 可运行组件，用于构建链式调用
from langchain_core.tools import tool  # 工具装饰器，用于将函数转换为 LangChain 工具
import uuid  # 用于生成唯一的会话ID

# 项目内部模块导入
from config.settings import settings  # 配置管理模块，包含 API 密钥等配置
from core.logger import app_logger  # 日志记录模块
from tools.amap_weather_tool import AmapWeatherTool  # 高德地图天气查询工具
from tools.tavily_search_tool import TavilySearchTool  # Tavily 搜索工具
from tools.tool_schemas import WeatherQuery, NewsSearch  # 工具参数模式定义


class QAAgent:
    """
    问答智能代理类

    这是系统的核心组件，负责处理用户输入、调用工具、生成回复。
    使用 LangChain 的 LCEL (LangChain Expression Language) 语法构建对话链。

    主要功能：
    1. 理解用户意图
    2. 自动决定是否需要调用工具（天气查询、信息搜索）
    3. 执行工具调用并格式化结果
    4. 生成自然语言回复
    5. 维护对话历史

    设计模式：
    - 使用工具绑定（bind_tools）实现 Function Calling
    - 使用链式组合（prompt | llm | output）实现对话流程
    - 使用单例模式管理会话状态
    """

    def __init__(self, session_id: str = None):
        """
        初始化问答代理

        Args:
            session_id: 可选的会话ID，如果不提供则自动生成UUID
                       用于标识不同的对话会话，便于日志追踪和状态管理
        """
        # 生成或使用提供的会话ID，用于追踪和日志记录
        self.session_id = session_id or str(uuid.uuid4())

        # 对话历史记录，存储用户和助手的交互历史
        # 格式: [{"user": "用户输入", "assistant": "助手回复", "timestamp": "...", "tools_used": [...]}, ...]
        self.conversation_history = []

        # 初始化工具实例
        # 天气查询工具：使用高德地图API查询城市天气
        self.weather_tool = AmapWeatherTool()
        # 信息搜索工具：使用Tavily API搜索网络信息
        # 从配置中读取 Tavily API 密钥
        self.search_tool = TavilySearchTool(api_key=settings.api.tavily_api_key)

        # 创建工具函数列表，这些工具将被绑定到LLM，供LLM自动调用
        self.tools = self._create_tools()

        # 初始化语言模型（LLM）
        # 注意：虽然使用 ChatOpenAI 类，但实际配置为使用 DeepSeek API
        self.llm = self._initialize_llm()

        # 将工具绑定到LLM，使LLM能够自动决定何时调用哪些工具
        # bind_tools 方法会将工具的函数签名传递给LLM，LLM可以根据用户输入自动调用
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 创建通用对话链，用于处理不需要工具调用的普通对话
        # 链式结构：prompt（提示词模板）| llm（语言模型）| output（输出解析器）
        self.general_chain = self._create_general_chain()

        # 记录初始化完成的日志
        app_logger.info(f"QA代理初始化完成，会话ID: {self.session_id}")

    def _create_tools(self):
        """
        创建工具函数列表

        使用 LangChain 的 @tool 装饰器将普通函数转换为工具。
        这些工具会被绑定到LLM，LLM可以根据用户输入自动决定是否调用。

        Returns:
            List: 工具函数列表，每个工具都是一个可调用的函数对象

        工具说明：
        1. weather_query: 天气查询工具，调用高德地图API获取城市天气信息
        2. news_search: 信息搜索工具，调用Tavily API搜索网络信息
        """

        @tool("weather_query", args_schema=WeatherQuery)
        def weather_query(city_name: str) -> str:
            """
            查询指定城市的天气信息

            这是一个 LangChain 工具函数，使用 @tool 装饰器标记。
            LLM 会自动识别这个工具，并在用户询问天气时调用它。

            Args:
                city_name: 城市名称，例如 "北京"、"上海" 等

            Returns:
                str: 天气信息的字符串描述，如果查询失败则返回错误信息

            工作流程：
            1. 接收城市名称参数
            2. 调用 AmapWeatherTool 获取天气数据
            3. 格式化并返回结果
            4. 如果出错，记录日志并返回错误信息
            """
            try:
                # 调用天气工具获取数据
                result = self.weather_tool.get_weather(city_name)
                if result.get("success"):
                    # 查询成功，返回天气数据
                    return result.get("data", "获取天气信息失败")
                else:
                    # 查询失败，返回错误信息
                    return f"获取{city_name}天气信息失败: {result.get('error', '未知错误')}"
            except Exception as e:
                # 捕获异常，记录日志并返回友好的错误信息
                app_logger.error(f"天气查询工具调用失败: {str(e)}")
                return f"天气查询失败: {str(e)}"

        @tool("news_search", args_schema=NewsSearch)
        def news_search(query: str, max_results: int = 5) -> str:
            """
            搜索新闻和信息

            这是一个 LangChain 工具函数，用于搜索网络上的最新信息。
            LLM 会在用户询问需要实时信息的问题时自动调用此工具。

            Args:
                query: 搜索查询字符串，例如 "最新财经新闻"、"人工智能发展"
                max_results: 最大返回结果数量，默认5条

            Returns:
                str: 格式化后的搜索结果，如果搜索失败则返回错误信息

            工作流程：
            1. 接收搜索查询和结果数量参数
            2. 调用 TavilySearchTool 执行搜索
            3. 格式化搜索结果并返回
            4. 如果出错，记录日志并返回错误信息
            """
            try:
                # 调用搜索工具获取结果
                result = self.search_tool.search_news(query, max_results)
                if result.get("success"):
                    # 搜索成功，格式化并返回结果
                    return self.search_tool.format_search_results(result)
                else:
                    # 搜索失败，返回错误信息
                    return f"搜索失败: {result.get('error', '未知错误')}"
            except Exception as e:
                # 捕获异常，记录日志并返回友好的错误信息
                app_logger.error(f"新闻搜索工具调用失败: {str(e)}")
                return f"搜索失败: {str(e)}"

        # 返回工具列表，这些工具将被绑定到LLM
        return [weather_query, news_search]

    def _initialize_llm(self) -> ChatOpenAI:
        """
        初始化语言模型（LLM）

        使用 LangChain 的 ChatOpenAI 类，但配置为使用 DeepSeek API。
        虽然类名是 ChatOpenAI，但由于 DeepSeek API 兼容 OpenAI API 格式，
        可以通过 base_url 参数指向 DeepSeek API 端点。

        Returns:
            ChatOpenAI: 配置好的语言模型实例，支持工具调用和对话生成

        配置说明：
        - model: 使用 "deepseek-chat" 模型（DeepSeek 的对话模型）
        - api_key: DeepSeek API 密钥（从配置中读取）
        - base_url: DeepSeek API 的基础 URL（从配置中读取，默认 https://api.deepseek.com/v1）
        - temperature: 0.3，控制输出的随机性（较低值使输出更确定性和一致）
        - max_tokens: 1000，限制单次响应的最大 token 数
        """
        return ChatOpenAI(
            model="deepseek-chat",  # DeepSeek 的对话模型名称
            api_key=settings.api.openai_api_key,  # 实际是 DeepSeek API 密钥（通过兼容性属性访问）
            base_url=settings.api.openai_base_url,  # 实际是 DeepSeek API 基础 URL（通过兼容性属性访问）
            temperature=0.3,  # 温度参数：0.0-2.0，值越低输出越确定，值越高输出越随机
            max_tokens=1000  # 最大 token 数，限制响应长度
        )

    def _create_general_chain(self):
        """
        创建通用对话链

        使用 LangChain 的 LCEL (LangChain Expression Language) 语法构建对话链。
        链式结构：prompt（提示词模板）| llm（语言模型）| output（输出解析器）

        Returns:
            Chain: LangChain 链对象，可以直接调用 invoke 方法处理用户输入

        链式流程说明：
        1. prompt: 将用户输入填充到提示词模板中
        2. llm: 将提示词发送给语言模型（DeepSeek）生成回复
        3. StrOutputParser: 将模型输出解析为纯文本字符串

        提示词设计：
        - 定义助手角色：友好的助手
        - 指导回答风格：简洁友好
        - 提供工具使用建议：引导用户使用正确的工具调用格式
        """
        # 创建提示词模板
        # 使用 from_template 方法创建模板，{query} 是占位符，会被用户输入替换
        prompt = ChatPromptTemplate.from_template("""
        你是一个友好的助手。用户说: {query}

        请简洁友好地回答用户的问题。如果用户询问天气，建议他们说"查询XX城市天气"。
        如果用户想搜索信息，建议他们说"搜索XX"。
        """)

        # 使用管道操作符 | 连接三个组件，形成处理链
        # prompt | llm: 将提示词发送给LLM
        # | StrOutputParser(): 将LLM的输出解析为字符串
        return prompt | self.llm | StrOutputParser()

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        处理用户输入并生成回复

        这是代理的核心方法，负责：
        1. 接收用户输入
        2. 调用LLM分析用户意图
        3. 自动决定是否需要调用工具
        4. 执行工具调用（如果需要）
        5. 格式化工具结果并生成最终回复
        6. 记录对话历史

        Args:
            user_input: 用户的输入文本

        Returns:
            Dict[str, Any]: 包含以下字段的字典：
                - response: 助手的回复文本
                - session_id: 会话ID
                - processing_time_ms: 处理时间（毫秒）
                - tools_used: 使用的工具列表
                - timestamp: 时间戳
                - error: 错误信息（如果有）

        工作流程：
        1. 记录开始时间，用于计算处理耗时
        2. 调用绑定工具的LLM，让LLM决定是否需要调用工具
        3. 如果LLM决定调用工具：
           a. 解析工具调用信息（工具名称和参数）
           b. 执行相应的工具函数
           c. 使用LLM格式化工具结果，生成自然语言回复
        4. 如果不需要工具调用：
           a. 使用通用对话链直接生成回复
        5. 记录对话历史
        6. 返回处理结果
        """
        # 记录开始时间，用于计算处理耗时
        start_time = time.time()

        try:
            # 步骤1：调用绑定工具的LLM处理用户输入
            # LLM会自动分析用户意图，决定是否需要调用工具
            # 如果决定调用工具，response.tool_calls 会包含工具调用信息
            response = self.llm_with_tools.invoke(user_input)

            # 初始化工具使用列表和最终回复
            tools_used = []  # 记录本次对话中使用的工具
            final_response = ""  # 最终的回复文本

            # 步骤2：检查LLM是否决定调用工具
            # tool_calls 是一个列表，包含所有需要调用的工具信息
            if response.tool_calls:
                # LLM决定需要调用工具
                print(f"🔧 检测到工具调用: {len(response.tool_calls)}个")

                # 步骤3：遍历所有工具调用并执行
                for tool_call in response.tool_calls:
                    # 提取工具名称和参数
                    tool_name = tool_call['name']  # 工具名称，例如 "weather_query" 或 "news_search"
                    tool_args = tool_call['args']  # 工具参数，是一个字典

                    print(f"📞 调用工具: {tool_name}, 参数: {tool_args}")

                    # 步骤4：根据工具名称执行相应的工具函数
                    if tool_name == "weather_query":
                        # 天气查询工具
                        city_name = tool_args.get('city_name', '')  # 从参数中提取城市名称
                        tool_result = self._execute_weather_tool(city_name)  # 执行天气查询
                        tools_used.append("amap_weather_tool")  # 记录使用的工具
                    elif tool_name == "news_search":
                        # 信息搜索工具
                        query = tool_args.get('query', '')  # 从参数中提取搜索查询
                        max_results = tool_args.get('max_results', 5)  # 从参数中提取最大结果数
                        tool_result = self._execute_search_tool(query, max_results)  # 执行搜索
                        tools_used.append("tavily_search_tool")  # 记录使用的工具
                    else:
                        # 未知工具，返回错误信息
                        tool_result = f"未知工具: {tool_name}"

                    # 步骤5：使用LLM格式化工具结果
                    # 工具返回的是原始数据，需要LLM将其转换为自然语言回复
                    format_prompt = ChatPromptTemplate.from_template("""
                    用户问题: {user_input}
                    工具结果: {tool_result}

                    请根据工具结果，用自然、友好的语言回答用户的问题。
                    """)

                    # 创建格式化链：提示词 | LLM | 输出解析器
                    format_chain = format_prompt | self.llm | StrOutputParser()
                    # 调用格式化链，生成最终回复
                    final_response = format_chain.invoke({
                        "user_input": user_input,  # 用户原始问题
                        "tool_result": tool_result  # 工具执行结果
                    })
            else:
                # 步骤6：不需要工具调用，使用通用对话链直接生成回复
                # 这种情况适用于普通对话，不需要调用外部工具
                final_response = self.general_chain.invoke({"query": user_input})

            # 步骤7：记录对话历史
            # 将本次对话保存到历史记录中，用于上下文理解（虽然当前版本未使用）
            self.conversation_history.append({
                "user": user_input,  # 用户输入
                "assistant": final_response,  # 助手回复
                "timestamp": datetime.now().isoformat(),  # 时间戳（ISO格式）
                "tools_used": tools_used  # 使用的工具列表
            })

            # 步骤8：限制历史记录长度，避免内存占用过大
            # 只保留最近10轮对话
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # 计算处理耗时（毫秒）
            processing_time = (time.time() - start_time) * 1000

            # 返回处理结果
            return {
                "response": final_response,  # 助手的回复
                "session_id": self.session_id,  # 会话ID
                "processing_time_ms": processing_time,  # 处理时间（毫秒）
                "tools_used": tools_used,  # 使用的工具列表
                "timestamp": datetime.now().isoformat()  # 时间戳
            }

        except Exception as e:
            # 异常处理：如果处理过程中出现错误，记录日志并返回错误信息
            app_logger.error(f"对话处理失败: {e}")
            processing_time = (time.time() - start_time) * 1000

            # 返回错误响应
            return {
                "response": f"抱歉，处理您的请求时出现了错误: {str(e)}",  # 友好的错误提示
                "session_id": self.session_id,
                "processing_time_ms": processing_time,
                "tools_used": [],  # 没有使用工具
                "timestamp": datetime.now().isoformat(),
                "error": str(e)  # 错误详情
            }

    def _execute_weather_tool(self, city_name: str) -> str:
        """
        执行天气查询工具

        这是一个内部方法，用于执行天气查询工具的实际调用。
        当LLM决定需要查询天气时，会调用此方法。

        Args:
            city_name: 城市名称，例如 "北京"、"上海"

        Returns:
            str: 天气信息的字符串描述，如果查询失败则返回错误信息
        """
        try:
            # 调用天气工具获取数据
            result = self.weather_tool.get_weather(city_name)
            if result.get("success"):
                # 查询成功，返回天气数据
                return result.get("data", "获取天气信息失败")
            else:
                # 查询失败，返回错误信息
                return f"获取{city_name}天气信息失败: {result.get('error', '未知错误')}"
        except Exception as e:
            # 捕获异常，记录日志并返回友好的错误信息
            app_logger.error(f"天气查询工具调用失败: {str(e)}")
            return f"天气查询失败: {str(e)}"

    def _execute_search_tool(self, query: str, max_results: int = 5) -> str:
        """
        执行新闻搜索工具

        这是一个内部方法，用于执行信息搜索工具的实际调用。
        当LLM决定需要搜索信息时，会调用此方法。

        Args:
            query: 搜索查询字符串
            max_results: 最大返回结果数量，默认5条

        Returns:
            str: 格式化后的搜索结果，如果搜索失败则返回错误信息
        """
        try:
            # 调用搜索工具获取结果
            result = self.search_tool.search_news(query, max_results)
            if result.get("success"):
                # 搜索成功，格式化并返回结果
                return self.search_tool.format_search_results(result['data'])
            else:
                # 搜索失败，返回错误信息
                return f"搜索失败: {result.get('error', '未知错误')}"
        except Exception as e:
            # 捕获异常，记录日志并返回友好的错误信息
            app_logger.error(f"新闻搜索工具调用失败: {str(e)}")
            return f"搜索失败: {str(e)}"

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        获取对话历史记录

        返回当前会话的所有对话历史，用于上下文理解或历史回顾。
        返回的是历史记录的副本，避免外部修改影响内部状态。

        Returns:
            List[Dict[str, Any]]: 对话历史记录列表，每个元素包含：
                - user: 用户输入
                - assistant: 助手回复
                - timestamp: 时间戳
                - tools_used: 使用的工具列表
        """
        return self.conversation_history.copy()  # 返回副本，保护内部数据

    def clear_conversation(self) -> None:
        """
        清空对话历史

        清除当前会话的所有对话历史记录。
        用于开始新的对话上下文，或释放内存。
        """
        self.conversation_history = []  # 清空历史记录
        app_logger.info(f"对话历史已清空: {self.session_id}")  # 记录日志

    def end_session(self) -> None:
        """
        结束会话

        标记当前会话结束，记录日志。
        可以在这里添加清理资源、保存历史等操作。
        """
        app_logger.info(f"会话已结束: {self.session_id}")  # 记录会话结束日志


def create_qa_agent(session_id: Optional[str] = None) -> QAAgent:
    """
    创建QA代理实例的工厂函数

    这是一个便捷函数，用于创建 QAAgent 实例。
    使用工厂函数的好处是可以统一管理代理的创建逻辑，
    未来如果需要添加初始化参数或配置，只需修改此函数。

    Args:
        session_id: 可选的会话ID，如果不提供则自动生成

    Returns:
        QAAgent: 初始化完成的问答代理实例

    使用示例:
        agent = create_qa_agent()
        result = agent.chat("查询北京天气")
    """
    return QAAgent(session_id=session_id)