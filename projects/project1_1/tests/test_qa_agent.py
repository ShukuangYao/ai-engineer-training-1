"""
QAAgent 测试模块

测试问答代理的核心功能，包括：
1. 代理初始化
2. 对话处理
3. 工具调用
4. 对话历史管理
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# 在导入前设置环境变量，避免配置验证失败
os.environ.setdefault('DEEPSEEK_API_KEY', 'test-deepseek-key')
os.environ.setdefault('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')
os.environ.setdefault('AMAP_API_KEY', 'test-amap-key')
os.environ.setdefault('TAVILY_API_KEY', 'test-tavily-key')

# 在导入前 mock dotenv，避免权限问题
with patch('dotenv.load_dotenv'):
    # 添加项目根目录到 Python 路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agents.qa_agent import QAAgent, create_qa_agent


class TestQAAgent:
    """QAAgent 测试类"""

    @pytest.fixture
    def agent(self):
        """创建测试用的 QAAgent 实例"""
        # Mock 所有需要的依赖
        with patch('agents.qa_agent.settings') as mock_settings, \
             patch('agents.qa_agent.ChatOpenAI') as mock_llm, \
             patch('agents.qa_agent.AmapWeatherTool') as mock_weather, \
             patch('agents.qa_agent.TavilySearchTool') as mock_search:

            # 模拟配置
            mock_settings.api.deepseek_api_key = "test-api-key"
            mock_settings.api.deepseek_base_url = "https://api.deepseek.com/v1"
            mock_settings.api.amap_api_key = "test-amap-key"
            mock_settings.api.tavily_api_key = "test-tavily-key"

            # 模拟工具
            mock_weather_instance = Mock()
            mock_weather.return_value = mock_weather_instance

            mock_search_instance = Mock()
            mock_search.return_value = mock_search_instance

            # 模拟 LLM
            mock_llm_instance = Mock()
            mock_llm_instance.bind_tools.return_value = mock_llm_instance
            mock_llm.return_value = mock_llm_instance

            # 创建代理实例
            agent = QAAgent(session_id="test-session-123")
            return agent

    def test_agent_initialization(self, agent):
        """测试代理初始化"""
        assert agent is not None
        assert agent.session_id == "test-session-123"
        assert agent.conversation_history == []
        assert agent.weather_tool is not None
        assert agent.search_tool is not None
        assert agent.llm is not None
        assert agent.tools is not None
        assert len(agent.tools) > 0

    def test_agent_has_required_methods(self, agent):
        """测试代理是否包含必需的方法"""
        assert hasattr(agent, 'chat')
        assert hasattr(agent, 'get_conversation_history')
        assert hasattr(agent, 'clear_conversation')
        assert hasattr(agent, 'end_session')
        assert callable(agent.chat)
        assert callable(agent.get_conversation_history)
        assert callable(agent.clear_conversation)
        assert callable(agent.end_session)

    @patch('agents.qa_agent.ChatOpenAI')
    def test_llm_initialization(self, mock_chat_openai, agent):
        """测试 LLM 初始化"""
        # 验证 LLM 被正确初始化
        assert agent.llm is not None

    def test_get_conversation_history_empty(self, agent):
        """测试获取空对话历史"""
        history = agent.get_conversation_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_clear_conversation(self, agent):
        """测试清空对话历史"""
        # 先添加一些历史记录
        agent.conversation_history = [
            {"user": "test", "assistant": "response"}
        ]

        # 清空历史
        agent.clear_conversation()

        # 验证历史已清空
        assert len(agent.conversation_history) == 0

    def test_end_session(self, agent):
        """测试结束会话"""
        # 添加一些历史记录
        agent.conversation_history = [
            {"user": "test", "assistant": "response"}
        ]

        # 结束会话（返回 None）
        result = agent.end_session()

        # 验证返回结果为 None（根据实际实现）
        assert result is None
        # 验证会话ID仍然存在
        assert agent.session_id is not None

    @patch('agents.qa_agent.QAAgent._execute_weather_tool')
    def test_chat_with_weather_query(self, mock_weather, agent):
        """测试天气查询对话"""
        # 模拟天气工具返回
        mock_weather.return_value = "北京今天天气晴朗，温度15°C"

        # 模拟 LLM 响应（工具调用）
        mock_llm_response = MagicMock()
        mock_llm_response.tool_calls = [
            {
                'name': 'weather_query',
                'args': {'city_name': '北京'},
                'id': 'call_123'
            }
        ]
        mock_llm_response.content = ""

        # 模拟格式化链的响应（最终回复）
        mock_format_response = "根据查询，北京今天天气晴朗，温度15°C"

        # 使用 MagicMock 替换链对象
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_llm_response

        # Mock 格式化链 - 确保返回字符串
        mock_format_chain = MagicMock()
        # 直接设置 invoke 方法的返回值
        def format_chain_invoke(*args, **kwargs):
            return mock_format_response
        mock_format_chain.invoke = format_chain_invoke

        # 替换链对象
        agent.llm_with_tools = mock_llm_with_tools

        # Mock ChatPromptTemplate 和格式化链的创建
        # 直接创建一个返回字符串的链对象
        mock_format_chain = MagicMock()
        mock_format_chain.invoke = MagicMock(return_value=mock_format_response)

        # Mock ChatPromptTemplate.from_template 返回的对象，支持链式调用
        with patch('agents.qa_agent.ChatPromptTemplate') as mock_template:
            # 创建模板实例，支持 | 操作符
            mock_template_instance = MagicMock()
            # 第一次 | 操作（与 llm）
            mock_intermediate = MagicMock()
            # 第二次 | 操作（与 StrOutputParser）
            mock_intermediate.__or__ = MagicMock(return_value=mock_format_chain)
            mock_template_instance.__or__ = MagicMock(return_value=mock_intermediate)
            mock_template.from_template.return_value = mock_template_instance

            result = agent.chat("查询北京天气")

            # 验证结果
            assert isinstance(result, dict)
            assert 'response' in result
            assert 'tools_used' in result
            assert 'processing_time_ms' in result
            assert isinstance(result['response'], str)
            assert result['response'] == mock_format_response
            assert isinstance(result['tools_used'], list)
            assert isinstance(result['processing_time_ms'], (int, float))
            # 验证工具被调用
            mock_weather.assert_called_once_with("北京")
            # 验证使用了工具
            assert len(result['tools_used']) > 0

    @patch('agents.qa_agent.QAAgent._execute_search_tool')
    def test_chat_with_search_query(self, mock_search, agent):
        """测试信息搜索对话"""
        # 模拟搜索工具返回
        mock_search.return_value = "找到了关于AI技术的最新信息..."

        # 模拟 LLM 响应（工具调用）
        mock_llm_response = MagicMock()
        mock_llm_response.tool_calls = [
            {
                'name': 'news_search',
                'args': {'query': 'AI技术', 'max_results': 5},
                'id': 'call_456'
            }
        ]
        mock_llm_response.content = ""

        # 模拟格式化链的响应（最终回复）
        mock_format_response = "根据搜索结果，以下是关于AI技术的最新信息..."

        # 使用 MagicMock 替换链对象
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_llm_response

        # Mock 格式化链 - 确保返回字符串
        def format_chain_invoke(*args, **kwargs):
            return mock_format_response

        # 替换链对象
        agent.llm_with_tools = mock_llm_with_tools

        # Mock ChatPromptTemplate 和格式化链的创建
        # 直接创建一个返回字符串的链对象
        mock_format_chain = MagicMock()
        mock_format_chain.invoke = MagicMock(return_value=mock_format_response)

        # Mock ChatPromptTemplate.from_template 返回的对象，支持链式调用
        with patch('agents.qa_agent.ChatPromptTemplate') as mock_template:
            # 创建模板实例，支持 | 操作符
            mock_template_instance = MagicMock()
            # 第一次 | 操作（与 llm）
            mock_intermediate = MagicMock()
            # 第二次 | 操作（与 StrOutputParser）
            mock_intermediate.__or__ = MagicMock(return_value=mock_format_chain)
            mock_template_instance.__or__ = MagicMock(return_value=mock_intermediate)
            mock_template.from_template.return_value = mock_template_instance

            result = agent.chat("搜索AI技术")

            # 验证结果
            assert isinstance(result, dict)
            assert 'response' in result
            assert 'tools_used' in result
            assert isinstance(result['response'], str)
            assert result['response'] == mock_format_response
            assert len(result['tools_used']) > 0
            # 验证工具被调用
            mock_search.assert_called_once_with("AI技术", 5)

    def test_chat_with_general_conversation(self, agent):
        """测试普通对话（不需要工具）"""
        # 模拟 LLM 响应（无工具调用）
        mock_llm_response = MagicMock()
        mock_llm_response.content = "你好！我是AI助手，很高兴为你服务。"
        mock_llm_response.tool_calls = []  # 空列表表示不调用工具

        # 模拟通用对话链的响应
        mock_general_response = "你好！我是AI助手，很高兴为你服务。"

        # 使用 MagicMock 替换链对象
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_llm_response

        mock_general_chain = MagicMock()
        mock_general_chain.invoke.return_value = mock_general_response

        # 替换链对象
        agent.llm_with_tools = mock_llm_with_tools
        agent.general_chain = mock_general_chain

        result = agent.chat("你好")

        # 验证结果
        assert isinstance(result, dict)
        assert 'response' in result
        assert 'tools_used' in result
        assert len(result['tools_used']) == 0  # 普通对话不使用工具
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0
        assert result['response'] == mock_general_response

    def test_conversation_history_management(self, agent):
        """测试对话历史管理"""
        # 模拟 LLM 响应（无工具调用）
        mock_llm_response = MagicMock()
        mock_llm_response.content = "测试回复"
        mock_llm_response.tool_calls = []

        # 模拟通用对话链的响应
        mock_general_response = "测试回复"

        # 使用 MagicMock 替换链对象
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_llm_response

        mock_general_chain = MagicMock()
        mock_general_chain.invoke.return_value = mock_general_response

        # 替换链对象
        agent.llm_with_tools = mock_llm_with_tools
        agent.general_chain = mock_general_chain

        # 第一次对话
        result1 = agent.chat("第一条消息")

        # 验证历史记录
        history = agent.get_conversation_history()
        assert len(history) == 1
        assert history[0]['user'] == "第一条消息"
        assert history[0]['assistant'] == "测试回复"

        # 第二次对话
        result2 = agent.chat("第二条消息")

        # 验证历史记录增加
        history = agent.get_conversation_history()
        assert len(history) == 2
        assert history[1]['user'] == "第二条消息"
        assert history[1]['assistant'] == "测试回复"

    def test_chat_handles_empty_input(self, agent):
        """测试处理空输入"""
        # 模拟 LLM 响应（无工具调用）
        mock_llm_response = MagicMock()
        mock_llm_response.content = "请提供有效的问题"
        mock_llm_response.tool_calls = []

        # 模拟通用对话链的响应
        mock_general_response = "请提供有效的问题"

        # 使用 MagicMock 替换链对象
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_llm_response

        mock_general_chain = MagicMock()
        mock_general_chain.invoke.return_value = mock_general_response

        # 替换链对象
        agent.llm_with_tools = mock_llm_with_tools
        agent.general_chain = mock_general_chain

        result = agent.chat("")

        # 验证结果
        assert isinstance(result, dict)
        assert 'response' in result
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0

    def test_chat_handles_error(self, agent):
        """测试错误处理"""
        # 模拟 LLM 抛出异常
        test_error = Exception("API错误")

        # 使用 MagicMock 替换链对象，让它抛出异常
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.side_effect = test_error

        # 替换链对象
        agent.llm_with_tools = mock_llm_with_tools

        # 应该捕获异常并返回错误信息
        result = agent.chat("测试")

        # 验证结果包含错误信息
        assert isinstance(result, dict)
        assert 'response' in result
        assert 'error' in result
        # 错误情况下应该返回错误信息
        assert len(result['response']) > 0
        assert "错误" in result['response'] or "抱歉" in result['response']
        assert result['error'] == "API错误"
        # 验证工具使用列表为空
        assert result['tools_used'] == []


class TestCreateQAAgent:
    """测试 create_qa_agent 函数"""

    @patch('agents.qa_agent.ChatOpenAI')
    @patch('agents.qa_agent.AmapWeatherTool')
    @patch('agents.qa_agent.TavilySearchTool')
    @patch('agents.qa_agent.settings')
    def test_create_qa_agent(self, mock_settings, mock_search, mock_weather, mock_llm):
        """测试创建代理函数"""
        # 模拟配置
        mock_settings.api.deepseek_api_key = "test-key"
        mock_settings.api.deepseek_base_url = "https://api.deepseek.com/v1"
        mock_settings.api.amap_api_key = "test-amap"
        mock_settings.api.tavily_api_key = "test-tavily"

        # 模拟工具和 LLM
        mock_weather.return_value = Mock()
        mock_search.return_value = Mock()
        mock_llm_instance = Mock()
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm.return_value = mock_llm_instance

        # 创建代理
        agent = create_qa_agent()

        # 验证代理已创建
        assert agent is not None
        assert isinstance(agent, QAAgent)
        assert agent.session_id is not None


class TestQAAgentIntegration:
    """QAAgent 集成测试（需要真实 API，可选）"""

    @pytest.mark.skip(reason="需要真实 API 密钥，跳过集成测试")
    @pytest.mark.skipif(
        not os.getenv('DEEPSEEK_API_KEY'),
        reason="需要配置 DEEPSEEK_API_KEY 环境变量"
    )
    def test_real_chat_with_llm(self):
        """测试真实 LLM 对话（需要 API 密钥）"""
        # 只有在配置了 API 密钥时才运行
        with patch('config.settings.settings') as mock_settings:
            if not mock_settings.api.deepseek_api_key:
                pytest.skip("未配置 API 密钥")

            agent = create_qa_agent()
            result = agent.chat("你好")

            # 验证基本结果
            assert isinstance(result, dict)
            assert 'response' in result
            assert len(result['response']) > 0


# 运行测试的辅助函数
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
