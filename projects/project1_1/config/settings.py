"""
配置管理模块
使用Pydantic进行配置验证和类型检查，符合企业级开发规范
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator, model_validator
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class APISettings(BaseSettings):
    """
    API相关配置类

    使用 Pydantic 的 BaseSettings 进行配置管理，支持从环境变量自动加载。
    所有 API 密钥和基础 URL 都通过环境变量配置，确保安全性。

    注意：为了兼容 LangChain 的代码，保留了 openai_api_key 和 openai_base_url 属性，
    但实际使用的是 DeepSeek API。环境变量应设置为 DEEPSEEK_API_KEY 和 DEEPSEEK_API_BASE。
    """

    # DeepSeek API配置
    # BaseSettings 会自动从环境变量读取，字段名 deepseek_api_key 对应环境变量 DEEPSEEK_API_KEY
    # 我们使用 model_validator 来处理向后兼容（支持从 OPENAI_API_KEY 读取）
    deepseek_api_key: str = Field(
        default="",
        description="DeepSeek API密钥，优先从环境变量 DEEPSEEK_API_KEY 读取，如果没有则从 OPENAI_API_KEY 读取（向后兼容）"
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API基础URL，优先从环境变量 DEEPSEEK_API_BASE 读取，如果没有则从 OPENAI_API_BASE 读取，默认 https://api.deepseek.com/v1"
    )

    @model_validator(mode='before')
    @classmethod
    def handle_compatibility(cls, data):
        """
        mode='before'：在字段验证之前执行，可以修改原始数据
        handle_compatibility 在每次创建 APISettings 实例时自动执行，确保环境变量兼容性处理在字段验证之前完成。
        处理环境变量兼容性，支持从 OPENAI_API_KEY 读取（向后兼容）
        这个方法在 BaseSettings 从环境变量读取之前运行，可以预先设置值

        Args:
            data: 原始输入数据（可能是字典或 Pydantic 对象）

        Returns:
            dict: 处理后的数据字典
        """
        # 如果 data 是字典，直接处理
        if isinstance(data, dict):
            # 处理 deepseek_api_key：优先从 DEEPSEEK_API_KEY 读取，如果没有则从 OPENAI_API_KEY 读取（向后兼容）
            if 'deepseek_api_key' not in data or not data.get('deepseek_api_key'):
                data['deepseek_api_key'] = os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY') or ''
            # 处理 deepseek_base_url：优先从 DEEPSEEK_API_BASE 读取，如果没有则从 OPENAI_API_BASE 读取
            if 'deepseek_base_url' not in data or not data.get('deepseek_base_url'):
                data['deepseek_base_url'] = os.getenv('DEEPSEEK_API_BASE') or os.getenv('OPENAI_API_BASE') or 'https://api.deepseek.com/v1'
            # 确保其他必需字段也能从环境变量读取（BaseSettings 会自动处理，但这里确保它们存在）
            if 'amap_api_key' not in data:
                data['amap_api_key'] = os.getenv('AMAP_API_KEY') or ''
            if 'tavily_api_key' not in data:
                data['tavily_api_key'] = os.getenv('TAVILY_API_KEY') or ''
        return data

    # 为了兼容现有代码，保留 openai_ 前缀的属性（实际指向 DeepSeek）
    @property
    def openai_api_key(self) -> str:
        """
        OpenAI API密钥的兼容性属性（实际返回 DeepSeek API密钥）

        这个属性是为了兼容使用 settings.api.openai_api_key 的代码，
        实际返回的是 DeepSeek API 密钥。

        Returns:
            str: DeepSeek API 密钥
        """
        return self.deepseek_api_key

    @property
    def openai_base_url(self) -> str:
        """
        OpenAI API基础URL的兼容性属性（实际返回 DeepSeek API基础URL）

        这个属性是为了兼容使用 settings.api.openai_base_url 的代码，
        实际返回的是 DeepSeek API 基础 URL。

        Returns:
            str: DeepSeek API 基础 URL
        """
        return self.deepseek_base_url

    # 高德地图API配置
    amap_api_key: str = Field(..., description="高德地图API密钥，用于查询天气信息")
    amap_base_url: str = Field(default="https://restapi.amap.com/v3", description="高德地图API基础URL")

    # Tavily搜索API配置
    tavily_api_key: str = Field(..., description="Tavily搜索API密钥，用于网络信息搜索")

    @validator('deepseek_api_key', 'amap_api_key', 'tavily_api_key')
    def validate_api_keys(cls, v):
        """
        验证API密钥不能为空

        Args:
            v: API密钥字符串

        Returns:
            str: 去除首尾空格的API密钥

        Raises:
            ValueError: 如果API密钥为空或只包含空白字符
        """
        if not v or v.strip() == "":
            raise ValueError("API密钥不能为空")
        return v.strip()

    class Config:
        env_prefix = ""
        case_sensitive = False


class RedisSettings(BaseSettings):
    """Redis缓存配置"""

    redis_host: str = Field(default="localhost", description="Redis主机地址")
    redis_port: int = Field(default=6379, description="Redis端口")
    redis_db: int = Field(default=0, description="Redis数据库编号")
    redis_password: Optional[str] = Field(default=None, description="Redis密码")

    @validator('redis_port')
    def validate_port(cls, v):
        """验证端口范围"""
        if not 1 <= v <= 65535:
            raise ValueError("端口号必须在1-65535范围内")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class AppSettings(BaseSettings):
    """应用程序配置"""

    app_name: str = Field(default="MultiTaskQAAssistant", description="应用名称")
    app_version: str = Field(default="1.0.0", description="应用版本")
    log_level: str = Field(default="INFO", description="日志级别")
    max_conversation_history: int = Field(default=50, description="最大对话历史记录数")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")

    @validator('log_level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是以下之一: {valid_levels}")
        return v.upper()

    @validator('max_conversation_history', 'cache_ttl')
    def validate_positive_int(cls, v):
        """验证正整数"""
        if v <= 0:
            raise ValueError("值必须大于0")
        return v

    class Config:
        env_prefix = ""
        case_sensitive = False


class Settings:
    """
    全局配置管理器

    采用单例模式，确保配置的一致性和性能
    提供统一的配置访问接口
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                self.api = APISettings()
                self.redis = RedisSettings()
                self.app = AppSettings()
                self._initialized = True
            except Exception as e:
                # 提供更友好的错误提示
                error_msg = str(e)
                if "amap_api_key" in error_msg or "tavily_api_key" in error_msg:
                    missing_keys = []
                    if not os.getenv('AMAP_API_KEY'):
                        missing_keys.append("AMAP_API_KEY (高德地图API密钥)")
                    if not os.getenv('TAVILY_API_KEY'):
                        missing_keys.append("TAVILY_API_KEY (Tavily搜索API密钥)")
                    if missing_keys:
                        keys_list = "\n".join(f"  - {key}" for key in missing_keys)
                        raise RuntimeError(
                            f"配置初始化失败: 缺少必需的环境变量:\n{keys_list}\n"
                            f"请在 .env 文件中设置这些环境变量，或使用 export 命令设置。"
                        )
                raise RuntimeError(f"配置初始化失败: {error_msg}")

    def get_city_data_path(self) -> str:
        """获取城市数据文件路径"""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "China-City-List-latest.csv")

    def validate_all(self) -> bool:
        """
        验证所有配置是否完整和有效

        检查所有必需的API密钥是否已配置，确保系统可以正常运行。

        Returns:
            bool: 如果所有配置验证通过返回 True，否则返回 False
        """
        try:
            # 检查必需的API密钥
            # 注意：这里检查的是 DeepSeek API 密钥（通过兼容性属性访问）
            if not self.api.deepseek_api_key:
                print("❌ DeepSeek API密钥未配置")
                print("💡 请在 .env 文件中设置 DEEPSEEK_API_KEY 环境变量")
                return False

            if not self.api.amap_api_key:
                print("❌ 高德地图API密钥未配置")
                print("💡 请在 .env 文件中设置 AMAP_API_KEY 环境变量")
                return False

            if not self.api.tavily_api_key:
                print("❌ Tavily搜索API密钥未配置")
                print("💡 请在 .env 文件中设置 TAVILY_API_KEY 环境变量")
                return False

            print("✅ 配置验证通过")
            return True

        except Exception as e:
            print(f"❌ 配置验证失败: {str(e)}")
            return False


# 全局配置实例
settings = Settings()