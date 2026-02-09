"""
API客户端工具模块

模块简介：
    提供与 FastAPI 模拟服务通信的客户端工具。
    封装了订单状态查询和物流信息查询的API调用，并集成了自动重试机制。

功能特点：
    1. 订单状态查询（Agent A 使用）
       - 调用 /api/orders/{order_id} 接口
       - 返回订单详细信息（状态、金额、商品等）
       - 支持自动重试（指数退避）

    2. 物流信息查询（Agent B 使用）
       - 调用 /api/logistics/{order_id} 接口
       - 返回物流跟踪信息（位置、状态、轨迹等）
       - 支持自动重试（指数退避）

    3. 健康检查
       - 调用 /health 接口
       - 检查服务是否正常运行

    错误处理：
        - 404错误：订单或物流信息不存在（不重试）
        - 500错误：服务暂时不可用（自动重试）
        - 网络错误：连接超时、连接失败等（自动重试）

    重试机制：
        - 使用 tenacity 库实现指数退避重试
        - 最大重试次数：3次
        - 最小等待时间：1秒
        - 最大等待时间：10秒
        - 退避倍数：2.0（每次重试等待时间翻倍）

技术实现：
    - 使用 httpx 异步HTTP客户端
    - 集成 RetryableHTTPClient 实现自动重试
    - 使用 Rich 库提供美观的错误提示

作者: AutoGen 多智能体客服系统开发团队
版本: 1.0.0
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from utils.retry import create_retry_decorator, RetryableHTTPClient
import httpx
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()

class APIClient:
    """
    API客户端类

    功能说明：
        负责与 FastAPI 模拟服务通信的客户端类。
        封装了所有API调用方法，并提供统一的错误处理和重试机制。

    属性说明：
        - base_url: API服务的基础URL（默认：http://127.0.0.1:8000）
        - client: 带重试机制的HTTP客户端（RetryableHTTPClient）

    使用场景：
        - Agent A（订单查询专员）调用 get_order_status() 查询订单
        - Agent B（物流跟踪专员）调用 get_logistics_info() 查询物流
        - 系统启动时调用 health_check() 检查服务状态

    示例：
        >>> client = APIClient()
        >>> order_info = await client.get_order_status("ORD001")
        >>> logistics_info = await client.get_logistics_info("ORD001")
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        初始化API客户端

        参数:
            base_url (str): API服务的基础URL
                - 默认值：http://127.0.0.1:8000（本地FastAPI服务）
                - 可以修改为其他服务地址（如生产环境）
        """
        self.base_url = base_url  # API服务基础URL
        # 创建带重试机制的HTTP客户端
        # timeout=30.0: 请求超时时间30秒
        self.client = RetryableHTTPClient(
            base_url=base_url,
            timeout=30.0
        )

    @create_retry_decorator(max_attempts=3, min_wait=1.0, max_wait=10.0)
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        获取订单状态信息
        Agent A 使用此方法查询订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态信息字典
        """
        try:
            logger.info(f"🔍 查询订单状态: {order_id}")
            console.print(Panel(f"[bold blue]正在查询订单[/bold blue]: {order_id}", border_style="blue"))

            response = await self.client.get(f"/api/orders/{order_id}")
            order_data = response.json()

            logger.info(f"✅ 订单查询成功: {order_id} -> {order_data['status']}")
            console.print(Panel(f"[bold green]订单查询成功[/bold green]: {order_id}", border_style="green"))
            return order_data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"❌ 订单不存在: {order_id}")
                console.print(Panel(f"[bold yellow]订单不存在[/bold yellow]: {order_id}", border_style="yellow"))
                return {"error": f"订单 {order_id} 不存在"}
            elif e.response.status_code == 500:
                logger.warning(f"⚠️ 订单服务暂时不可用: {order_id} -> HTTP {e.response.status_code}")
                console.print(Panel(f"[bold yellow]订单服务暂时不可用[/bold yellow]: {order_id} -> HTTP {e.response.status_code}", border_style="yellow"))
                return {"error": f"订单服务暂时不可用，请稍后再试", "order_id": order_id, "status": "service_unavailable"}
            else:
                logger.error(f"❌ 订单查询失败: {order_id} -> HTTP {e.response.status_code}")
                console.print(Panel(f"[bold red]订单查询失败[/bold red]: {order_id} -> HTTP {e.response.status_code}", border_style="red"))
                return {"error": f"订单查询失败: HTTP {e.response.status_code}", "order_id": order_id, "status": "query_failed"}
        except Exception as e:
            logger.error(f"❌ 订单查询异常: {order_id} -> {str(e)}")
            console.print(Panel(f"[bold red]订单查询异常[/bold red]: {order_id} -> {str(e)}", border_style="red"))
            return {"error": f"订单查询异常: {str(e)}", "order_id": order_id, "status": "exception"}

    @create_retry_decorator(max_attempts=3, min_wait=1.0, max_wait=10.0)
    async def get_logistics_info(self, order_id: str) -> Dict[str, Any]:
        """
        获取物流信息
        Agent B 使用此方法查询物流信息

        Args:
            order_id: 订单ID

        Returns:
            物流信息字典
        """
        try:
            logger.info(f"🚚 查询物流信息: {order_id}")
            console.print(Panel(f"[bold blue]正在查询物流[/bold blue]: {order_id}", border_style="blue"))

            response = await self.client.get(f"/api/logistics/{order_id}")
            logistics_data = response.json()

            logger.info(f"✅ 物流查询成功: {order_id}")
            console.print(Panel(f"[bold green]物流查询成功[/bold green]: {order_id}", border_style="green"))
            return logistics_data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"❌ 物流信息不存在: {order_id}")
                console.print(Panel(f"[bold yellow]物流信息不存在[/bold yellow]: {order_id}", border_style="yellow"))
                return {"error": f"订单 {order_id} 的物流信息不存在"}
            elif e.response.status_code == 500:
                logger.warning(f"⚠️ 物流服务暂时不可用: {order_id} -> HTTP {e.response.status_code}")
                console.print(Panel(f"[bold yellow]物流服务暂时不可用[/bold yellow]: {order_id} -> HTTP {e.response.status_code}", border_style="yellow"))
                return {"error": f"物流服务暂时不可用，请稍后再试", "order_id": order_id, "status": "service_unavailable"}
            else:
                logger.error(f"❌ 物流查询失败: {order_id} -> HTTP {e.response.status_code}")
                console.print(Panel(f"[bold red]物流查询失败[/bold red]: {order_id} -> HTTP {e.response.status_code}", border_style="red"))
                return {"error": f"物流查询失败: HTTP {e.response.status_code}", "order_id": order_id, "status": "query_failed"}
        except Exception as e:
            logger.error(f"❌ 物流查询异常: {order_id} -> {str(e)}")
            console.print(Panel(f"[bold red]物流查询异常[/bold red]: {order_id} -> {str(e)}", border_style="red"))
            return {"error": f"物流查询异常: {str(e)}", "order_id": order_id, "status": "exception"}

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            服务是否健康
        """
        try:
            logger.info("🔍 执行健康检查")

            response = await self.client.get("/health")
            health_data = response.json()

            is_healthy = health_data.get("status") == "healthy"
            if is_healthy:
                logger.info("✅ 服务健康检查通过")
            else:
                logger.warning("⚠️  服务健康检查失败")
            return is_healthy

        except Exception as e:
            logger.error(f"❌ 健康检查失败: {str(e)}")
            return False

    async def close(self):
        """关闭客户端连接"""
        await self.client.close()
        logger.info("API客户端关闭")

    async def __aenter__(self):
        """异步上下文管理器进入"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()


# 全局API客户端实例
api_client = APIClient()