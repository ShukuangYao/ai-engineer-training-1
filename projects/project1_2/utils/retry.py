"""
重试机制工具模块

模块简介：
    提供网络请求的自动重试机制，使用 tenacity 库实现指数级退避策略。
    当网络请求失败时，自动进行重试，提高系统的稳定性和可靠性。

功能特点：
    1. 指数退避重试
       - 每次重试的等待时间按指数增长
       - 公式：wait_time = min_wait * (multiplier ^ attempt_number)
       - 例如：第1次重试等待1秒，第2次等待2秒，第3次等待4秒

    2. 智能重试条件
       - 404错误不重试（资源不存在）
       - 500错误重试（服务暂时不可用）
       - 网络错误重试（连接超时、连接失败等）

    3. 可配置参数
       - max_attempts: 最大重试次数（默认：3次）
       - min_wait: 最小等待时间（默认：1.0秒）
       - max_wait: 最大等待时间（默认：10.0秒）
       - multiplier: 指数退避倍数（默认：2.0）

    4. 详细日志记录
       - 记录每次重试的尝试次数
       - 记录等待时间
       - 记录异常信息

使用场景：
    - API调用失败时自动重试
    - 网络连接不稳定时的容错处理
    - 临时服务不可用时的自动恢复

技术实现：
    - 使用 tenacity 库实现重试装饰器
    - 自定义重试条件（should_retry_http_error）
    - 集成日志记录（before_sleep_log, after_log）

作者: AutoGen 多智能体客服系统开发团队
版本: 1.0.0
"""
import asyncio
import logging
from typing import Callable, Any, Optional, Type, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import httpx
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

def should_retry_http_error(exception):
    """
    判断HTTP错误是否应该重试

    功能说明：
        根据HTTP错误的状态码判断是否应该重试。
        404错误（资源不存在）不应该重试，因为重试也不会成功。
        其他HTTP错误（如500、503等）可以重试，可能是临时性问题。

    参数:
        exception: HTTP异常对象（httpx.HTTPStatusError）

    Returns:
        bool: True表示应该重试，False表示不应该重试

    重试策略：
        - 404 Not Found: 不重试（资源不存在，重试无意义）
        - 500 Internal Server Error: 重试（服务器临时错误）
        - 503 Service Unavailable: 重试（服务暂时不可用）
        - 其他HTTP错误: 重试（可能是临时性问题）
        - 网络错误: 重试（连接超时、连接失败等）

    示例:
        >>> # 404错误不重试
        >>> should_retry_http_error(httpx.HTTPStatusError(404, ...))
        False

        >>> # 500错误重试
        >>> should_retry_http_error(httpx.HTTPStatusError(500, ...))
        True
    """
    if isinstance(exception, httpx.HTTPStatusError):
        # 404 Not Found 不应该重试
        # 原因：资源不存在，重试也不会成功，只会浪费时间和资源
        if exception.response.status_code == 404:
            return False
        # 其他HTTP错误可以重试
        # 原因：可能是临时性问题（如服务器过载、网络波动等）
        return True
    # 非HTTP错误（如网络连接错误、超时等）应该重试
    return True

# 定义需要重试的异常类型
RETRIABLE_EXCEPTIONS = (
    httpx.RequestError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.HTTPStatusError,
    RequestException,
    ConnectionError,
    TimeoutError
)

def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0,
    exception_types: tuple = RETRIABLE_EXCEPTIONS
):
    """
    创建重试装饰器

    功能说明：
        创建一个使用指数退避策略的重试装饰器。
        当函数抛出可重试的异常时，自动进行重试，每次重试的等待时间按指数增长。

    参数:
        max_attempts (int): 最大重试次数（包括首次尝试）
            - 默认值：3次（首次尝试 + 2次重试）
            - 例如：max_attempts=3 表示最多尝试3次

        min_wait (float): 最小等待时间（秒）
            - 默认值：1.0秒
            - 第一次重试的等待时间

        max_wait (float): 最大等待时间（秒）
            - 默认值：10.0秒
            - 防止等待时间过长，超过此值则不再增长

        multiplier (float): 指数退避倍数
            - 默认值：2.0
            - 每次重试的等待时间 = 上次等待时间 * multiplier
            - 例如：第1次等待1秒，第2次等待2秒，第3次等待4秒

        exception_types (tuple): 需要重试的异常类型
            - 默认值：RETRIABLE_EXCEPTIONS（包含网络错误、HTTP错误等）
            - 只有这些异常类型才会触发重试

    Returns:
        retry装饰器: 可以用于装饰需要重试的函数

    重试策略示例：
        假设 max_attempts=3, min_wait=1.0, multiplier=2.0:
        - 第1次尝试：立即执行
        - 第1次失败：等待 1.0 秒后重试
        - 第2次失败：等待 2.0 秒后重试
        - 第3次失败：不再重试，抛出异常

    使用示例:
        >>> @create_retry_decorator(max_attempts=3, min_wait=1.0)
        ... async def fetch_data():
        ...     response = await httpx.get("https://api.example.com/data")
        ...     return response.json()
        >>>
        >>> # 如果请求失败，会自动重试最多3次
        >>> data = await fetch_data()
    """
    def custom_retry_condition(exception):
        """自定义重试条件"""
        # 首先检查是否是需要重试的异常类型
        if not isinstance(exception, exception_types):
            return False
        # 然后检查HTTP错误的具体条件
        return should_retry_http_error(exception)

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=custom_retry_condition,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )

# 默认重试装饰器
default_retry = create_retry_decorator()

# API调用专用重试装饰器
api_retry = create_retry_decorator(
    max_attempts=5,
    min_wait=0.5,
    max_wait=30.0,
    multiplier=2.0
)

class RetryableHTTPClient:
    """
    带重试机制的HTTP客户端
    """

    def __init__(self, base_url: str = "", timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout
        )

    @api_retry
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """GET请求with重试"""
        logger.info(f"发起GET请求: {url}")
        try:
            response = await self.client.get(url, **kwargs)
            response.raise_for_status()
            logger.info(f"GET请求成功: {url} -> {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"GET请求失败: {url} -> {str(e)}")
            raise

    @api_retry
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """POST请求with重试"""
        logger.info(f"发起POST请求: {url}")
        try:
            response = await self.client.post(url, **kwargs)
            response.raise_for_status()
            logger.info(f"POST请求成功: {url} -> {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"POST请求失败: {url} -> {str(e)}")
            raise

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# 全局HTTP客户端实例
http_client = RetryableHTTPClient()

async def retry_async_call(
    func: Callable,
    *args,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    **kwargs
) -> Any:
    """
    异步函数重试调用

    Args:
        func: 要重试的异步函数
        max_attempts: 最大重试次数
        min_wait: 最小等待时间
        max_wait: 最大等待时间
        *args, **kwargs: 传递给函数的参数
    """
    retry_decorator = create_retry_decorator(
        max_attempts=max_attempts,
        min_wait=min_wait,
        max_wait=max_wait
    )

    @retry_decorator
    async def _wrapped_func():
        return await func(*args, **kwargs)

    return await _wrapped_func()

def log_retry_attempt(retry_state):
    """记录重试尝试"""
    logger.warning(
        f"重试第 {retry_state.attempt_number} 次调用 "
        f"{retry_state.outcome.exception()} "
        f"等待 {retry_state.next_action.sleep} 秒后重试"
    )