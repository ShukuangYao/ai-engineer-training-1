"""
Tavily搜索工具模块
用于获取最新新闻和信息搜索
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from tavily import TavilyClient
from core.logger import app_logger


class TavilySearchTool:
    """
    Tavily搜索工具类

    用于调用 Tavily API 进行网络信息搜索，支持新闻搜索和信息检索。
    需要有效的 Tavily API 密钥才能使用。
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Tavily搜索工具

        Args:
            api_key: Tavily API密钥，如果为 None 则从环境变量或配置中读取

        Raises:
            ValueError: 如果 API 密钥未提供且无法从环境变量读取
        """
        # 优先使用传入的 API 密钥，否则从环境变量读取
        if api_key is None:
            api_key = os.getenv('TAVILY_API_KEY')

        if not api_key:
            raise ValueError(
                "Tavily API 密钥未配置。请设置环境变量 TAVILY_API_KEY 或在初始化时传入 api_key 参数。"
            )

        self.api_key = api_key
        self.client = TavilyClient(api_key)
        app_logger.info("Tavily搜索工具初始化完成")

    def search_news(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        搜索新闻信息

        调用 Tavily API 进行网络信息搜索，支持新闻、文章、网页内容等搜索。
        返回格式化的搜索结果，包括标题、URL、内容摘要等。

        Args:
            query: 搜索查询字符串，例如 "最新AI技术"、"北京天气" 等
            max_results: 最大返回结果数量，默认为 5，建议范围 1-10

        Returns:
            Dict[str, Any]: 包含搜索结果的字典，格式如下：
                {
                    'success': bool,  # 搜索是否成功
                    'data': {         # 成功时的数据
                        'query': str,           # 搜索查询
                        'answer': str,           # AI 生成的总结（如果有）
                        'results': List[Dict],   # 搜索结果列表
                        'search_time': str,     # 搜索时间（ISO 格式）
                        'total_results': int     # 结果总数
                    },
                    'error': str      # 失败时的错误信息
                }

        Raises:
            不会抛出异常，所有错误都会被捕获并返回在结果字典中
        """
        try:
            app_logger.info(f"开始搜索新闻: {query}")

            # 调用Tavily搜索API
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )

            if not response or 'results' not in response:
                return {
                    'success': False,
                    'error': '搜索结果为空',
                    'data': None
                }

            # 格式化搜索结果
            formatted_results = []
            for result in response.get('results', []):
                formatted_result = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'published_date': result.get('published_date', ''),
                    'score': result.get('score', 0)
                }
                formatted_results.append(formatted_result)

            # 构建返回数据
            search_data = {
                'query': query,
                'answer': response.get('answer', ''),
                'results': formatted_results,
                'search_time': datetime.now().isoformat(),
                'total_results': len(formatted_results)
            }

            app_logger.info(f"成功获取 {len(formatted_results)} 条搜索结果")

            return {
                'success': True,
                'data': search_data,
                'error': None
            }

        except Exception as e:
            # 解析错误信息，提供更友好的错误提示
            error_str = str(e)
            error_msg = "搜索失败"

            # 检查是否是 HTTP 错误
            if "432" in error_str or "Client Error" in error_str:
                error_msg = (
                    "Tavily API 请求失败 (432 错误)。可能的原因：\n"
                    "1. API 密钥无效或已过期\n"
                    "2. API 配额已用完\n"
                    "3. 请求参数格式不正确\n"
                    f"详细错误: {error_str}"
                )
            elif "401" in error_str or "Unauthorized" in error_str:
                error_msg = (
                    "Tavily API 认证失败。请检查 API 密钥是否正确。\n"
                    f"详细错误: {error_str}"
                )
            elif "429" in error_str or "Too Many Requests" in error_str:
                error_msg = (
                    "Tavily API 请求频率过高，请稍后再试。\n"
                    f"详细错误: {error_str}"
                )
            else:
                error_msg = f"搜索失败: {error_str}"

            app_logger.error(f"Tavily 搜索错误: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'data': None
            }

    def format_search_results(self, search_data: Dict[str, Any]) -> str:
        """
        格式化搜索结果为可读文本

        Args:
            search_data: 搜索数据

        Returns:
            格式化后的文本
        """
        if not search_data or not search_data.get('results'):
            return "未找到相关搜索结果"

        formatted_text = f"🔍 搜索查询: {search_data.get('query', '')}\n\n"

        # 添加AI总结（如果有）
        if search_data.get('answer'):
            formatted_text += f"📝 AI总结:\n{search_data['answer']}\n\n"

        # 添加搜索结果
        formatted_text += "📰 相关新闻:\n"
        for i, result in enumerate(search_data['results'][:5], 1):
            title = result.get('title', '无标题')
            content = result.get('content', '')
            url = result.get('url', '')

            # 截取内容前150个字符
            if len(content) > 150:
                content = content[:150] + "..."

            formatted_text += f"\n{i}. {title}\n"
            formatted_text += f"   {content}\n"
            if url:
                formatted_text += f"   🔗 {url}\n"

        formatted_text += f"\n⏰ 搜索时间: {search_data.get('search_time', '')}"
        formatted_text += f"\n📊 共找到 {search_data.get('total_results', 0)} 条结果"

        return formatted_text

    def search_and_format(self, query: str, max_results: int = 5) -> str:
        """
        搜索并格式化结果的便捷方法

        Args:
            query: 搜索查询
            max_results: 最大结果数量

        Returns:
            格式化后的搜索结果文本
        """
        search_result = self.search_news(query, max_results)

        if not search_result['success']:
            return f"搜索失败: {search_result.get('error', '未知错误')}"

        return self.format_search_results(search_result['data'])


# 创建全局实例（延迟初始化，避免在导入时就需要配置）
# 实际使用时应该从配置中获取 API 密钥
def get_tavily_search_tool(api_key: Optional[str] = None) -> TavilySearchTool:
    """
    获取 Tavily 搜索工具实例

    Args:
        api_key: Tavily API 密钥，如果为 None 则从环境变量读取

    Returns:
        TavilySearchTool: Tavily 搜索工具实例
    """
    return TavilySearchTool(api_key=api_key)


# 尝试创建全局实例（如果环境变量已配置）
try:
    tavily_search_tool = TavilySearchTool()
except ValueError:
    # 如果 API 密钥未配置，创建 None 占位符
    # 实际使用时应该通过 get_tavily_search_tool() 或从配置中获取
    tavily_search_tool = None
    app_logger.warning("Tavily 搜索工具未初始化，因为 API 密钥未配置")