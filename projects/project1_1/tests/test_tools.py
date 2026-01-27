"""
工具模块测试

测试天气查询工具和信息搜索工具的功能
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.amap_weather_tool import AmapWeatherTool
from tools.tavily_search_tool import TavilySearchTool


class TestAmapWeatherTool:
    """高德地图天气工具测试"""

    @pytest.fixture
    def weather_tool(self):
        """创建测试用的天气工具实例"""
        with patch('tools.amap_weather_tool.settings') as mock_settings:
            mock_settings.api.amap_api_key = "test-amap-key"
            mock_settings.api.amap_base_url = "https://restapi.amap.com/v3"
            return AmapWeatherTool()

    def test_weather_tool_initialization(self, weather_tool):
        """测试天气工具初始化"""
        assert weather_tool is not None
        assert hasattr(weather_tool, 'api_key')
        assert hasattr(weather_tool, 'base_url')
        assert hasattr(weather_tool, 'get_weather')

    @patch('tools.amap_weather_tool.requests.get')
    def test_get_weather_success(self, mock_get, weather_tool):
        """测试成功获取天气"""
        # 模拟 API 响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "1",
            "count": "1",
            "info": "OK",
            "lives": [{
                "province": "北京",
                "city": "北京市",
                "weather": "晴",
                "temperature": "15",
                "winddirection": "南",
                "windpower": "3",
                "humidity": "45",
                "reporttime": "2026-01-27 10:00:00"
            }]
        }
        mock_get.return_value = mock_response

        # 调用方法
        result = weather_tool.get_weather("北京")

        # 验证结果
        assert isinstance(result, dict)
        assert result.get('success') is True
        assert 'data' in result
        assert '北京' in result['data'] or '天气' in result['data']

    @patch('tools.amap_weather_tool.requests.get')
    def test_get_weather_api_error(self, mock_get, weather_tool):
        """测试 API 错误处理"""
        # 模拟 API 错误响应
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "status": "0",
            "info": "INVALID_KEY"
        }
        mock_get.return_value = mock_response

        # 调用方法
        result = weather_tool.get_weather("北京")

        # 验证错误处理
        assert isinstance(result, dict)
        assert result.get('success') is False
        assert 'error' in result

    @patch('tools.amap_weather_tool.requests.get')
    def test_get_weather_network_error(self, mock_get, weather_tool):
        """测试网络错误处理"""
        # 模拟网络异常
        mock_get.side_effect = Exception("网络连接失败")

        # 调用方法
        result = weather_tool.get_weather("北京")

        # 验证错误处理
        assert isinstance(result, dict)
        assert result.get('success') is False
        assert 'error' in result

    def test_get_weather_empty_city(self, weather_tool):
        """测试空城市名称"""
        result = weather_tool.get_weather("")

        # 应该返回错误或空结果
        assert isinstance(result, dict)


class TestTavilySearchTool:
    """Tavily 搜索工具测试"""

    @pytest.fixture
    def search_tool(self):
        """创建测试用的搜索工具实例"""
        return TavilySearchTool(api_key="test-tavily-key")

    def test_search_tool_initialization(self, search_tool):
        """测试搜索工具初始化"""
        assert search_tool is not None
        assert hasattr(search_tool, 'api_key')
        assert hasattr(search_tool, 'client')
        assert hasattr(search_tool, 'search_news')
        assert hasattr(search_tool, 'format_search_results')

    def test_search_tool_requires_api_key(self):
        """测试搜索工具需要 API 密钥"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API 密钥未配置"):
                TavilySearchTool(api_key=None)

    @patch('tools.tavily_search_tool.TavilyClient')
    def test_search_news_success(self, mock_client_class, search_tool):
        """测试成功搜索新闻"""
        # 模拟客户端
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # 模拟搜索结果
        mock_response = {
            'results': [
                {
                    'title': '测试标题',
                    'url': 'https://example.com',
                    'content': '测试内容',
                    'published_date': '2026-01-27',
                    'score': 0.95
                }
            ],
            'answer': '这是AI生成的总结'
        }
        mock_client.search.return_value = mock_response

        # 更新工具的客户端
        search_tool.client = mock_client

        # 调用方法
        result = search_tool.search_news("测试查询")

        # 验证结果
        assert isinstance(result, dict)
        assert result.get('success') is True
        assert 'data' in result
        assert result['data']['total_results'] > 0

    @patch('tools.tavily_search_tool.TavilyClient')
    def test_search_news_api_error(self, mock_client_class, search_tool):
        """测试 API 错误处理"""
        # 模拟客户端
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # 模拟 API 错误
        from requests.exceptions import HTTPError
        mock_client.search.side_effect = HTTPError("432 Client Error")

        # 更新工具的客户端
        search_tool.client = mock_client

        # 调用方法
        result = search_tool.search_news("测试查询")

        # 验证错误处理
        assert isinstance(result, dict)
        assert result.get('success') is False
        assert 'error' in result
        assert '432' in result['error'] or '错误' in result['error']

    @patch('tools.tavily_search_tool.TavilyClient')
    def test_search_news_empty_results(self, mock_client_class, search_tool):
        """测试空搜索结果"""
        # 模拟客户端
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # 模拟空结果
        mock_client.search.return_value = {}

        # 更新工具的客户端
        search_tool.client = mock_client

        # 调用方法
        result = search_tool.search_news("测试查询")

        # 验证结果
        assert isinstance(result, dict)
        assert result.get('success') is False
        assert 'error' in result or '空' in result.get('error', '')

    def test_format_search_results(self, search_tool):
        """测试格式化搜索结果"""
        # 测试数据
        search_data = {
            'query': '测试查询',
            'answer': 'AI总结',
            'results': [
                {
                    'title': '标题1',
                    'content': '内容1' * 10,  # 长内容
                    'url': 'https://example.com/1'
                },
                {
                    'title': '标题2',
                    'content': '内容2',
                    'url': 'https://example.com/2'
                }
            ],
            'total_results': 2
        }

        # 调用方法
        formatted = search_tool.format_search_results(search_data)

        # 验证结果
        assert isinstance(formatted, str)
        assert '测试查询' in formatted
        assert '标题1' in formatted
        assert '标题2' in formatted
        assert len(formatted) > 0

    def test_format_search_results_empty(self, search_tool):
        """测试格式化空搜索结果"""
        # 测试空数据
        result = search_tool.format_search_results(None)
        assert isinstance(result, str)
        assert '未找到' in result or '空' in result.lower()

    def test_search_and_format(self, search_tool):
        """测试搜索并格式化"""
        with patch.object(search_tool, 'search_news') as mock_search:
            # 模拟搜索结果
            mock_search.return_value = {
                'success': True,
                'data': {
                    'query': '测试',
                    'results': [{'title': '标题', 'content': '内容'}],
                    'total_results': 1
                }
            }

            # 调用方法
            result = search_tool.search_and_format("测试查询")

            # 验证结果
            assert isinstance(result, str)
            assert len(result) > 0


class TestToolSchemas:
    """工具参数模式测试"""

    def test_weather_query_schema(self):
        """测试天气查询参数模式"""
        from tools.tool_schemas import WeatherQuery

        # 测试有效参数
        query = WeatherQuery(city_name="北京")
        assert query.city_name == "北京"

        # 测试必需参数
        with pytest.raises(Exception):  # Pydantic 会抛出验证错误
            WeatherQuery()

    def test_news_search_schema(self):
        """测试新闻搜索参数模式"""
        from tools.tool_schemas import NewsSearch

        # 测试有效参数
        search = NewsSearch(query="AI技术")
        assert search.query == "AI技术"
        assert search.max_results == 5  # 默认值

        # 测试自定义 max_results
        search2 = NewsSearch(query="测试", max_results=10)
        assert search2.max_results == 10

        # 测试必需参数
        with pytest.raises(Exception):  # Pydantic 会抛出验证错误
            NewsSearch()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
