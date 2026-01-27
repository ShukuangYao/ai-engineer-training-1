"""
Pytest 配置文件

提供测试用的 fixtures 和配置
"""

import pytest
import sys
import os
from unittest.mock import patch

# 在导入任何模块前，先 mock dotenv 避免权限问题
with patch('dotenv.load_dotenv'):
    # 添加项目根目录到 Python 路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """设置测试环境"""
    # 禁用 LangChain 追踪
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"

    # 设置测试日志级别
    os.environ["LOG_LEVEL"] = "ERROR"  # 测试时只显示错误

    # Mock dotenv 避免权限问题
    with patch('dotenv.load_dotenv'):
        yield

    # 清理（如果需要）


@pytest.fixture
def mock_settings():
    """模拟配置对象"""
    with patch('config.settings.settings') as mock:
        # 设置默认配置值
        mock.api.deepseek_api_key = "test-deepseek-key"
        mock.api.deepseek_base_url = "https://api.deepseek.com/v1"
        mock.api.amap_api_key = "test-amap-key"
        mock.api.tavily_api_key = "test-tavily-key"
        mock.app.log_level = "ERROR"
        mock.app.max_conversation_history = 50
        yield mock
