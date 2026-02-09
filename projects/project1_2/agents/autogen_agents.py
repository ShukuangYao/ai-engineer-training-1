"""
AutoGen智能体模块

模块简介：
    基于 AutoGen 框架实现的多智能体客服系统核心模块。
    定义了多个专门的智能体，每个智能体负责不同的任务，通过协作来处理客户服务问题。

智能体架构：
    1. 客服接待员（Customer Service Agent）
       - 职责：友好接待客户，了解客户问题，进行初步分类
       - 功能：收集订单信息，将问题转交给相应的专业团队

    2. 订单查询专员（Order Query Agent）
       - 职责：处理所有订单相关的查询
       - 功能：提取订单号，查询订单详细信息，解释订单状态

    3. 物流跟踪专员（Logistics Agent）
       - 职责：处理配送和物流相关问题
       - 功能：查询物流状态，提供配送时间预估，处理配送异常

    4. 客服主管（Summary Agent）
       - 职责：整合信息并生成完整回复
       - 功能：汇总订单和物流信息，生成用户友好的回复

技术特点：
    - 使用 DeepSeek API 作为 LLM 服务（兼容 OpenAI API 格式）
    - 支持工具函数注册和调用（订单查询、物流查询）
    - 实时显示智能体交互过程（Rich 库）
    - 支持群组聊天模式，智能体可以相互协作

作者: AutoGen 多智能体客服系统开发团队
版本: 1.0.0
"""
import autogen
import json
import time
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich import box
from core.logger import setup_logger
from config.settings import settings
from tools.api_client import api_client
import asyncio

logger = setup_logger(__name__)
console = Console()


class InteractiveAgentDisplay:
    """
    智能体交互显示包装器

    功能说明：
        用于实时显示智能体的交互过程，提供美观的命令行界面展示。
        记录智能体的任务开始时间、任务计数等信息，并格式化输出到控制台。

    属性说明：
        - agent_name: 智能体名称（例如："订单查询专员"）
        - agent_type: 智能体类型（例如："order_query"）
        - start_time: 任务开始时间（用于计算耗时）
        - task_count: 任务计数器（记录执行的任务数量）

    使用场景：
        在智能体执行任务时，自动记录和显示交互过程，包括：
        - 任务开始
        - 思考过程
        - 执行操作
        - 任务完成
        - 错误信息

    示例：
        >>> display = InteractiveAgentDisplay("订单查询专员", "order_query")
        >>> display.log_interaction("开始查询订单", level="start")
        >>> display.log_interaction("查询完成", level="result")
    """

    def __init__(self, agent_name: str, agent_type: str):
        """
        初始化智能体交互显示包装器

        参数:
            agent_name (str): 智能体名称，用于显示和日志记录
            agent_type (str): 智能体类型，用于分类和标识
        """
        self.agent_name = agent_name  # 智能体名称
        self.agent_type = agent_type  # 智能体类型
        self.start_time = None  # 任务开始时间（时间戳）
        self.task_count = 0  # 任务计数器

    def log_interaction(self, message: str, level: str = "info", task_id: str = None):
        """记录智能体交互过程"""
        timestamp = time.strftime("%H:%M:%S")

        # 创建交互消息
        if level == "start":
            self.start_time = time.time()
            self.task_count += 1
            panel_content = f"🤖 [bold blue]{self.agent_name}[/bold blue] 开始执行任务"
            border_style = "bright_blue"
            # 添加详细的开始日志
            logger.info(f"🚀 Agent [{self.agent_name}] 开始执行任务 #{self.task_count} - {message}")
        elif level == "thinking":
            panel_content = f"🧠 [bold yellow]{self.agent_name}[/bold yellow] 正在思考: {message}"
            border_style = "bright_yellow"
            logger.info(f"🧠 Agent [{self.agent_name}] 思考中: {message}")
        elif level == "action":
            panel_content = f"🛠️  [bold green]{self.agent_name}[/bold green] 执行操作: {message}"
            border_style = "bright_green"
            logger.info(f"🛠️ Agent [{self.agent_name}] 执行操作: {message}")
        elif level == "result":
            elapsed = time.time() - self.start_time if self.start_time else 0
            panel_content = f"✅ [bold green]{self.agent_name}[/bold green] 任务完成 (耗时: {elapsed:.2f}s)\n{message}"
            border_style = "bright_green"
            # 添加详细的完成日志
            logger.info(f"✅ Agent [{self.agent_name}] 任务完成 - 耗时: {elapsed:.2f}s")
            logger.info(f"📋 Agent [{self.agent_name}] 任务结果: {message}")
        elif level == "error":
            panel_content = f"❌ [bold red]{self.agent_name}[/bold red] 错误: {message}"
            border_style = "bright_red"
            logger.error(f"❌ Agent [{self.agent_name}] 执行错误: {message}")
        else:
            panel_content = f"💬 {self.agent_name}: {message}"
            border_style = "white"
            logger.info(f"💬 Agent [{self.agent_name}]: {message}")

        # 显示交互面板
        panel = Panel(
            panel_content,
            title=f"[bold]{self.agent_name}[/bold]",
            subtitle=f"⏰ {timestamp} | 📋 任务 #{self.task_count}",
            border_style=border_style,
            box=box.ROUNDED,
            expand=False
        )

        console.print(panel)


# 自定义消息处理函数
def create_agent_message_handler(agent_name: str, display: InteractiveAgentDisplay):
    """创建agent消息处理函数"""
    def handle_message(sender, message, request_reply=False):
        """处理agent消息"""
        if sender.name != agent_name:  # 只处理发送给当前agent的消息
            # 显示接收到的消息
            display.log_interaction(f"接收到来自 {sender.name} 的消息", level="start")
            console.print(f"[bold cyan]📨 {agent_name} 接收消息:[/bold cyan]")
            console.print(Panel(message.get("content", ""), border_style="cyan", box=box.SIMPLE))

        return False, None  # 不拦截消息，继续正常处理

    return handle_message


# 全局变量：跟踪客服主管是否已经发送过终止消息
_summary_agent_terminated = False

def reset_summary_agent_terminated():
    """
    重置客服主管的终止状态

    功能说明：
        在每次新查询开始时调用此函数，重置全局变量 _summary_agent_terminated，
        确保客服主管可以正常发送消息。

    使用场景：
        - 交互式模式下，每次新查询时调用
        - 单次查询模式下，每次查询开始时调用
    """
    global _summary_agent_terminated
    _summary_agent_terminated = False
    logger.debug("🔄 已重置客服主管的终止状态")

def create_agent_reply_handler(agent_name: str, display: InteractiveAgentDisplay):
    """创建agent回复处理函数"""
    global _summary_agent_terminated

    def handle_reply(sender, message, recipient, silent):
        """处理agent回复"""
        global _summary_agent_terminated

        # 关键修复：如果客服主管已经发送过终止消息，阻止它再次发送
        if agent_name == "客服主管" and _summary_agent_terminated:
            logger.warning(f"🛑 [{agent_name}] 已经发送过终止消息，阻止重复发送")
            # 返回一个包含终止标记的简短消息，让系统知道应该停止
            # 不显示重复的终止消息，直接返回一个静默的终止消息
            if isinstance(message, dict):
                message["content"] = "问题已解决。TERMINATE"
                message["terminate"] = True
                # 添加一个特殊标记，表示这是重复的终止消息
                message["_is_repeat_terminate"] = True
                # 标记为静默消息，不显示
                message["_silent"] = True
            elif isinstance(message, str):
                message = {
                    "content": "问题已解决。TERMINATE",
                    "terminate": True,
                    "_is_repeat_terminate": True,
                    "_silent": True,
                    "name": sender.name if hasattr(sender, 'name') else agent_name
                }
            # 不显示重复的终止消息，直接返回
            # 注意：即使返回了消息，GroupChat 也应该检测到终止条件并停止
            return message

        # 处理不同类型的消息
        content = ""
        if isinstance(message, str):
            content = message
        elif isinstance(message, dict) and message.get("content"):
            content = message.get("content", "")

        if content:
            # 检查终止关键词（特别是客服主管的消息）
            termination_keywords = ["TERMINATE", "terminate", "问题已解决"]
            content_upper = str(content).upper()
            content_lower = str(content).lower()

            has_terminate = False
            for keyword in termination_keywords:
                if keyword.upper() in content_upper or keyword.lower() in content_lower:
                    has_terminate = True
                    logger.info(f"✅ [{agent_name}] 检测到终止关键词: '{keyword}'")
                    break

            # 关键修复：如果客服主管发送了包含终止关键词的消息，标记为已终止
            if agent_name == "客服主管" and has_terminate:
                _summary_agent_terminated = True
                logger.info(f"🛑 [{agent_name}] 已标记为已终止，后续将阻止重复发送")
                # 关键修复：立即让 user_proxy 收到终止消息，触发 is_termination_msg
                # 这样 user_proxy 就能检测到终止并停止对话
                try:
                    # 获取 user_proxy（从 recipient 或其他方式）
                    # 注意：这里需要访问 user_proxy，但可能无法直接访问
                    # 所以我们在 GroupChat 的消息历史中检查终止条件
                    pass
                except Exception as e:
                    logger.debug(f"无法直接通知 user_proxy: {e}")

            # 检查是否是静默消息（重复的终止消息）
            is_silent = False
            if isinstance(message, dict):
                is_silent = message.get("_silent", False) or message.get("_is_repeat_terminate", False)

            # 如果不是静默消息，才显示
            if not is_silent:
                # 显示agent正在生成回复
                display.log_interaction(f"正在生成回复给 {recipient.name}", level="thinking")

                # 显示回复内容
                console.print(f"[bold green]📤 {agent_name} 发送回复:[/bold green]")
                console.print(Panel(content, border_style="green", box=box.SIMPLE))

            # 如果检测到终止关键词，确保消息格式正确以便终止条件识别
            if has_terminate:
                # 如果是字典格式，添加终止标记
                if isinstance(message, dict):
                    message["terminate"] = True
                    logger.info(f"✅ [{agent_name}] 消息已标记为终止消息")
                # 如果是字符串，转换为字典并添加标记
                elif isinstance(message, str):
                    message = {
                        "content": message,
                        "terminate": True,
                        "name": sender.name if hasattr(sender, 'name') else agent_name
                    }
                    logger.info(f"✅ [{agent_name}] 消息已转换为终止消息格式")

                # 在 GroupChat 中，消息发送给 chat_manager 是正常行为
                # chat_manager 会将消息路由给所有参与者，包括 user_proxy
                # user_proxy 的 is_termination_msg 会检查所有收到的消息
                if agent_name == "客服主管" and hasattr(recipient, 'name'):
                    recipient_name = recipient.name
                    if recipient_name == "chat_manager":
                        # 这是正常的 GroupChat 行为，记录为 debug 级别
                        logger.debug(f"ℹ️ [{agent_name}] 终止消息已发送给 chat_manager（GroupChat 正常行为）")
                        logger.debug(f"ℹ️ chat_manager 会将消息路由给所有参与者，包括 user_proxy")
                    elif recipient_name != "客户":
                        # 如果发送给其他 agent，记录为 info（可能是正常的协作）
                        logger.debug(f"ℹ️ [{agent_name}] 终止消息发送给了 {recipient_name}（GroupChat 会路由给所有参与者）")

            display.log_interaction(f"已发送回复给 {recipient.name}", level="result")

        return message  # 返回消息（可能已修改）

    return handle_reply


# 工具函数定义
def extract_order_id_from_message(message: str) -> str:
    """
    从消息中提取订单ID

    Args:
        message: 用户消息内容

    Returns:
        str: 提取到的订单ID，如果没有找到则返回默认值ORD001
    """
    import re
    # 使用正则表达式匹配订单ID模式 (ORD + 数字)
    pattern = r'ORD\d+'
    match = re.search(pattern, message, re.IGNORECASE)

    if match:
        order_id = match.group().upper()
        logger.info(f"🔍 从消息中提取到订单ID: {order_id}")
        return order_id
    else:
        logger.warning(f"⚠️ 未能从消息中提取订单ID，使用默认值: ORD001")
        return "ORD001"

async def get_order_info_async(order_id: str) -> str:
    """异步获取订单信息的工具函数"""
    display = InteractiveAgentDisplay("订单查询工具", "tool")

    # 如果没有提供order_id，使用默认值
    if not order_id:
        order_id = "ORD001"  # 默认值

    display.log_interaction(f"开始查询订单: {order_id}", level="start")

    try:
        order_info = await api_client.get_order_status(order_id)

        # 检查是否有错误
        if "error" in order_info:
            error_msg = f"很抱歉，订单 {order_id} 不存在。请检查订单号是否正确，或联系客服获取帮助。"
            display.log_interaction(f"订单不存在: {order_id}", level="error")
            logger.warning(f"❌ 订单不存在: {order_id}")
            return error_msg

        result = f"""订单查询结果：
            订单ID: {order_info.get('order_id', 'N/A')}
            订单状态: {order_info.get('status', 'N/A')}
            客户姓名: {order_info.get('customer_name', 'N/A')}
            订单金额: ¥{order_info.get('total_amount', 0)}
            商品列表: {', '.join(order_info.get('items', []))}
            收货地址: {order_info.get('shipping_address', 'N/A')}
            创建时间: {order_info.get('created_at', 'N/A')}
            更新时间: {order_info.get('updated_at', 'N/A')}"""

        # 添加详细的订单结果日志
        logger.info(f"📦 订单查询成功 - 订单ID: {order_info.get('order_id')}")
        logger.info(f"📊 订单详情 - 状态: {order_info.get('status')}, 金额: ¥{order_info.get('total_amount', 0)}")
        logger.info(f"👤 客户信息 - 姓名: {order_info.get('customer_name')}, 地址: {order_info.get('shipping_address')}")
        logger.info(f"🛍️ 商品列表: {', '.join(order_info.get('items', []))}")

        display.log_interaction(f"订单查询成功: {order_id}", level="result")
        return result

    except Exception as e:
        error_msg = f"订单查询系统暂时不可用，请稍后重试或联系客服。错误信息: {str(e)}"
        display.log_interaction(error_msg, level="error")
        logger.error(f"❌ 订单查询异常: {order_id} -> {str(e)}")
        return error_msg

def get_order_info(order_id: str) -> str:
    """获取订单信息的工具函数（同步包装器）"""
    try:
        # 尝试获取当前运行的事件循环
        loop = asyncio.get_running_loop()
        # 如果已经在事件循环中，使用 run_in_executor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, get_order_info_async(order_id))
            return future.result()
    except RuntimeError:
        # 如果没有运行的事件循环，直接运行
        return asyncio.run(get_order_info_async(order_id))


async def get_logistics_info_async(order_id: str) -> str:
    """异步获取物流信息的工具函数"""
    display = InteractiveAgentDisplay("物流查询工具", "tool")
    display.log_interaction(f"开始查询物流: {order_id}", level="start")

    try:
        logistics_info = await api_client.get_logistics_info(order_id)

        # 检查是否有错误
        if "error" in logistics_info:
            error_msg = f"很抱歉，订单 {order_id} 的物流信息不存在。可能是订单尚未发货或订单号不正确，请联系客服获取帮助。"
            display.log_interaction(f"物流信息不存在: {order_id}", level="error")
            logger.warning(f"❌ 物流信息不存在: {order_id}")
            return error_msg

        # 格式化物流轨迹
        tracking_history = ""
        if logistics_info.get('tracking_history'):
            tracking_history = "\n物流轨迹:\n"
            for record in logistics_info['tracking_history']:
                tracking_history += f"  {record.get('time', 'N/A')} - {record.get('location', 'N/A')}: {record.get('status', 'N/A')}\n"

        result = f"""物流查询结果：
            物流单号: {logistics_info.get('tracking_number', '暂未分配')}
            物流状态: {logistics_info.get('status', 'N/A')}
            当前位置: {logistics_info.get('current_location', 'N/A')}
            承运商: {logistics_info.get('carrier', 'N/A')}
            预计送达: {logistics_info.get('estimated_delivery', '未确定')}{tracking_history}"""

        # 添加详细的物流结果日志
        logger.info(f"🚚 物流查询成功 - 订单ID: {order_id}")
        logger.info(f"📋 物流详情 - 单号: {logistics_info.get('tracking_number')}, 状态: {logistics_info.get('status')}")
        logger.info(f"📍 位置信息 - 当前位置: {logistics_info.get('current_location')}, 承运商: {logistics_info.get('carrier')}")
        logger.info(f"⏰ 预计送达: {logistics_info.get('estimated_delivery', '未确定')}")

        display.log_interaction(f"物流查询成功: {order_id}", level="result")
        return result

    except Exception as e:
        error_msg = f"物流查询系统暂时不可用，请稍后重试或联系客服。错误信息: {str(e)}"
        display.log_interaction(error_msg, level="error")
        logger.error(f"❌ 物流查询异常: {order_id} -> {str(e)}")
        return error_msg


def get_logistics_info(order_id: str) -> str:
    """获取物流信息的工具函数（同步包装器）"""
    try:
        # 尝试获取当前运行的事件循环
        loop = asyncio.get_running_loop()
        # 如果已经在事件循环中，使用 run_in_executor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, get_logistics_info_async(order_id))
            return future.result()
    except RuntimeError:
        # 如果没有运行的事件循环，直接运行
        return asyncio.run(get_logistics_info_async(order_id))


def create_autogen_agents():
    """
    创建AutoGen智能体团队

    功能说明：
        创建并配置所有需要的 AutoGen 智能体，包括：
        - 用户代理（UserProxyAgent）：代表客户
        - 客服接待员：初步接待和问题分类
        - 订单查询专员：处理订单查询
        - 物流跟踪专员：处理物流查询
        - 客服主管：汇总结果并生成回复

    工作流程：
        1. 配置 LLM（使用 DeepSeek API）
        2. 创建交互显示包装器
        3. 创建用户代理
        4. 创建各个专业智能体
        5. 注册消息处理器（用于显示交互过程）
        6. 注册工具函数（订单查询、物流查询）
        7. 返回智能体字典

    Returns:
        dict: 包含所有智能体、交互显示器和LLM配置的字典
            - user_proxy: 用户代理
            - customer_service_agent: 客服接待员
            - order_query_agent: 订单查询专员
            - logistics_agent: 物流跟踪专员
            - summary_agent: 客服主管
            - interactive_displays: 交互显示器字典
            - llm_config: LLM配置

    注意事项：
        - 所有智能体使用相同的 LLM 配置（DeepSeek API）
        - 工具函数需要注册到对应的智能体才能使用
        - 消息处理器用于实时显示智能体交互过程
    """
    global _summary_agent_terminated

    # 重置全局变量，确保每次创建新的 agents 时都能正常工作
    _summary_agent_terminated = False

    logger.info("创建AutoGen智能体")
    console.print("\n[bold cyan]🚀 正在初始化AutoGen智能体团队...[/bold cyan]\n")

    # 配置LLM（使用 DeepSeek API）
    # 优先使用 DEEPSEEK_API_KEY，如果没有则使用 AUTOGEN_API_KEY，最后使用 OPENAI_API_KEY（向后兼容）
    config_list = [
        {
            "model": settings.AUTOGEN_MODEL,  # DeepSeek 模型名称
            "api_key": settings.AUTOGEN_API_KEY or settings.DEEPSEEK_API_KEY or settings.OPENAI_API_KEY,
            "base_url": settings.AUTOGEN_BASE_URL,  # DeepSeek API 基础URL
        }
    ]

    llm_config = {
        "config_list": config_list,
        "temperature": settings.AUTOGEN_TEMPERATURE,
        "timeout": settings.AUTOGEN_TIMEOUT,
    }

    # 创建交互显示包装器
    interactive_displays = {
        "customer_service": InteractiveAgentDisplay("客服接待员", "customer_service"),
        "order_query": InteractiveAgentDisplay("订单查询专员", "order_query"),
        "logistics": InteractiveAgentDisplay("物流跟踪专员", "logistics"),
        "summary": InteractiveAgentDisplay("客服主管", "summary"),
    }

    # 创建用户代理
    # 终止条件：当消息中包含"问题已解决"、"TERMINATE"或明确的结束标志时终止对话
    # 注意：这个函数会检查所有发送给 user_proxy 的消息，包括其他智能体的回复
    # 同时也会检查消息中的 terminate 标记（由消息钩子设置）
    # 创建一个闭包来访问 groupchat（稍后在 create_group_chat 中设置）
    groupchat_ref = {"groupchat": None}

    def is_termination_msg(msg):
        """
        判断消息是否为终止消息

        这个函数会被 user_proxy 调用来检查收到的消息。
        如果返回 True，user_proxy 将不会继续回复，从而终止对话。

        检查方式：
        1. 检查消息中的 terminate 标记（由消息钩子设置）
        2. 检查消息内容中的终止关键词
        3. 特别检查来自"客服主管"的消息（因为它负责生成最终回复）
        4. **新增：检查 GroupChat 的消息历史中是否有终止消息**
        """
        # 首先检查消息中的 terminate 标记
        if msg.get("terminate") is True:
            sender_name = msg.get("name", "unknown")
            logger.info(f"✅ 检测到终止标记，对话将终止")
            logger.info(f"📝 终止消息来自: {sender_name}")
            return True

        # 关键修复：检查 GroupChat 的消息历史中是否有终止消息
        # 这样即使当前消息不是终止消息，也能检测到之前的终止消息
        if groupchat_ref["groupchat"] and hasattr(groupchat_ref["groupchat"], 'messages'):
            for history_msg in reversed(groupchat_ref["groupchat"].messages[-5:]):  # 检查最后5条消息
                if history_msg.get("terminate") is True:
                    sender_name = history_msg.get("name", "unknown")
                    logger.info(f"✅ 在消息历史中检测到终止标记，来自: {sender_name}")
                    logger.info(f"✅ 对话将终止")
                    return True

                # 检查消息内容中的终止关键词
                history_content = history_msg.get("content", "")
                if history_content:
                    history_sender = history_msg.get("name", "unknown")
                    if history_sender == "客服主管" or "客服主管" in str(history_sender):
                        history_content_upper = str(history_content).upper()
                        history_content_lower = str(history_content).lower()
                        termination_keywords = ["TERMINATE", "terminate", "问题已解决"]

                        for keyword in termination_keywords:
                            if keyword.upper() in history_content_upper or keyword.lower() in history_content_lower:
                                logger.info(f"✅ 在消息历史中检测到终止关键词: '{keyword}'，来自: {history_sender}")
                                logger.info(f"✅ 对话将终止")
                                return True

        content = msg.get("content", "").strip()
        if not content:
            return False

        # 记录消息内容（用于调试）
        sender_name = msg.get("name", "unknown")
        logger.info(f"🔍 检查终止条件 - 发送者: {sender_name}, 消息长度: {len(content)}")

        # 如果是来自"客服主管"的消息，更严格地检查终止条件
        # 因为客服主管负责生成最终回复
        is_from_summary = sender_name == "客服主管" or "客服主管" in str(sender_name)

        # 检查终止关键词（不区分大小写）
        # 优先检查明确的终止关键词
        termination_keywords = [
            "TERMINATE",  # 最明确的终止标志
            "terminate",
            "问题已解决",  # 中文终止标志
            "问题已处理完毕",
            "咨询已完成",
            "问题已答复",
        ]

        # 检查是否包含终止关键词
        content_upper = content.upper()  # 转换为大写进行匹配
        content_lower = content.lower()  # 转换为小写进行匹配

        for keyword in termination_keywords:
            keyword_upper = keyword.upper()
            keyword_lower = keyword.lower()

            # 检查大写和小写版本
            if keyword_upper in content_upper or keyword_lower in content_lower:
                logger.info(f"✅ 检测到终止关键词: '{keyword}'，对话将终止")
                logger.info(f"📝 终止消息来自: {sender_name}")
                # 如果是来自客服主管的消息，即使只包含一个终止关键词也终止
                if is_from_summary:
                    logger.info(f"🎯 来自客服主管的终止消息，立即终止对话")
                    return True
                # 对于其他消息，需要更严格的检查
                return True

        # 特殊检查：如果消息同时包含"问题已解决"和"TERMINATE"（不区分大小写）
        if ("问题已解决" in content or "问题已解决" in content_lower) and \
           ("TERMINATE" in content_upper or "terminate" in content_lower):
            logger.info(f"✅ 检测到组合终止标志（问题已解决 + TERMINATE），对话将终止")
            logger.info(f"📝 终止消息来自: {sender_name}")
            return True

        # 检查是否在消息末尾明确表示结束
        # 如果消息以句号结尾且包含"问题已解决"
        if content.endswith("。") or content.endswith("."):
            if "问题已解决" in content or "问题已解决" in content_lower:
                logger.info(f"✅ 检测到终止消息（末尾包含'问题已解决'），对话将终止")
                logger.info(f"📝 终止消息来自: {sender_name}")
                return True

        # 如果来自客服主管且消息较长（可能是最终回复），检查是否包含终止关键词
        if is_from_summary and len(content) > 100:
            # 检查消息末尾是否包含终止关键词
            last_200_chars = content[-200:].lower()
            if "terminate" in last_200_chars or "问题已解决" in last_200_chars:
                logger.info(f"✅ 检测到客服主管的长消息中包含终止关键词，对话将终止")
                return True

        return False

    user_proxy = autogen.UserProxyAgent(
        name="客户",
        human_input_mode=settings.AUTOGEN_HUMAN_INPUT_MODE,
        max_consecutive_auto_reply=settings.AUTOGEN_MAX_CONSECUTIVE_AUTO_REPLY,
        is_termination_msg=is_termination_msg,
        code_execution_config={"work_dir": "temp", "use_docker": False},
    )

    # 客服接待智能体
    customer_service_agent = autogen.AssistantAgent(
        name="客服接待员",
        system_message="""你是一名专业的电商客服接待员。你的职责是：
            1. 友好接待客户，了解客户问题
            2. 对问题进行初步分类（订单查询、退换货、物流问题、产品咨询等）
            3. 收集必要的订单信息（订单号、客户信息等）
            4. 将问题转交给相应的专业团队处理

            请用简洁明了的语言与客户沟通。当客户提到具体订单号时，请直接转交给订单查询专员处理。
            如果问题涉及多个方面，请协调相关专员共同解决。

            回复格式：简洁专业，直接回答客户问题。""",
                    llm_config=llm_config,
                )

    # 订单查询智能体
    order_query_agent = autogen.AssistantAgent(
        name="订单查询专员",
        system_message="""你是订单查询专员，负责处理所有订单相关的查询。你的职责包括：
            1. 从客户查询中提取订单号（格式如ORD001、ORD002等）
            2. 使用get_order_info工具函数查询订单详细信息
            3. 解释订单状态和处理进度
            4. 提供预计发货和到货时间
            5. 识别需要其他部门协助的问题

            重要：当客户提供订单号时，你必须：
            1. 从查询文本中提取订单ID（如ORD002）
            2. 调用get_order_info函数，传入提取到的订单ID
            3. 根据查询结果向客户提供详细信息

            如果无法从查询中提取到订单ID，请使用默认值ORD001。

            示例：
            客户问："我的订单ORD002为什么还没发货？"
            你应该调用：get_order_info("ORD002")
            然后根据返回结果回答客户问题。

            回复格式：提供详细的订单信息，包括状态、商品、金额等关键信息。""",
        llm_config=llm_config,
    )

    # 物流跟踪智能体
    logistics_agent = autogen.AssistantAgent(
        name="物流跟踪专员",
        system_message="""你是物流跟踪专员，专门处理配送和物流相关问题。你的职责包括：
            1. 查询包裹物流状态和位置
            2. 提供准确的配送时间预估
            3. 处理配送异常和延误问题
            4. 协调配送地址修改

            当需要查询物流信息时，请使用 get_logistics_info 函数。
            请提供实时、准确的物流信息，并主动提醒客户注意事项。

            回复格式：提供详细的物流状态，包括当前位置、预计到达时间等。""",
                    llm_config=llm_config,
                )

    # 结果汇总智能体
    summary_agent = autogen.AssistantAgent(
        name="客服主管",
        system_message="""你是一名资深的客服主管，拥有多年的客户服务经验。
            你擅长整合来自不同部门的信息，为客户提供全面、准确、友好的回复。
            你总是站在客户的角度思考问题，能够用通俗易懂的语言解释复杂的情况，
            并在必要时提供解决方案和建议。

            你的职责是：
            1. 汇总订单和物流信息
            2. 生成完整的问题解答
            3. 确保客户得到满意的答复
            4. **最重要：在生成完整回复后，必须在回复的最后明确写上"问题已解决。TERMINATE"来终止对话**

            **关键规则（必须严格遵守，违反将导致系统错误）：**
            1. **你只能发送一次消息，就是最终回复，发送后立即停止，不要再发送任何消息**
            2. 当你看到订单查询结果和物流查询结果后，立即生成最终回复
            3. **在最终回复的最后一行，必须包含："问题已解决。TERMINATE"（必须包含这两个词）**
            4. **生成最终回复后，绝对不要再发送任何消息，不要重复调用工具，不要询问其他问题**
            5. **每个查询只生成一次最终回复，发送后立即停止**
            6. **如果你已经发送了包含"问题已解决。TERMINATE"的回复，绝对不要再发送任何消息**

            **回复格式（必须严格遵循）：**
            你的回复应该包含：
            1. 订单信息总结
            2. 物流信息总结
            3. 对客户问题的解答
            4. 最后一行必须是："问题已解决。TERMINATE"

            示例：
            ```
            根据查询结果，您的订单ORD001状态如下：
            - 订单状态：已打包
            - 物流状态：待发货
            - 预计发货时间：今天或明天
            - 预计送达时间：发货后1-2天

            您的订单已经完成打包，正在等待物流公司上门取件。预计今天下午或明天上午您就能收到物流单号。

            希望以上信息对您有帮助。问题已解决。TERMINATE
            ```

            **停止规则（非常重要）：**
            - 一旦你在回复中包含了"问题已解决。TERMINATE"，**立即停止，不要再发送任何消息**
            - 系统会在你发送包含 TERMINATE 的回复后自动终止对话
            - 如果你看到对话还在继续，说明你的回复格式不正确，请确保包含"问题已解决。TERMINATE"
            - **绝对不要重复发送消息，每个查询只生成一次最终回复**""",
                    llm_config=llm_config,
                )

    # 为每个agent添加消息处理器
    customer_service_agent.register_hook("process_message_before_send",
                                        create_agent_reply_handler("客服接待员", interactive_displays["customer_service"]))
    order_query_agent.register_hook("process_message_before_send",
                                   create_agent_reply_handler("订单查询专员", interactive_displays["order_query"]))
    logistics_agent.register_hook("process_message_before_send",
                                create_agent_reply_handler("物流跟踪专员", interactive_displays["logistics"]))
    summary_agent.register_hook("process_message_before_send",
                              create_agent_reply_handler("客服主管", interactive_displays["summary"]))

    # 注册工具函数
    autogen.register_function(
        get_order_info,
        caller=order_query_agent,
        executor=user_proxy,
        description="根据订单号获取订单详细信息"
    )

    autogen.register_function(
        get_logistics_info,
        caller=logistics_agent,
        executor=user_proxy,
        description="根据订单号获取物流跟踪信息"
    )

    console.print("[bold green]✅ AutoGen智能体团队创建完成！[/bold green]\n")

    return {
        "user_proxy": user_proxy,
        "customer_service_agent": customer_service_agent,
        "order_query_agent": order_query_agent,
        "logistics_agent": logistics_agent,
        "summary_agent": summary_agent,
        "interactive_displays": interactive_displays,
        "llm_config": llm_config,
        "groupchat_ref": groupchat_ref  # 传递 groupchat 引用，用于 is_termination_msg 检查消息历史
    }


def create_group_chat(agents_dict):
    """
    创建群组聊天

    功能说明：
        创建 AutoGen 群组聊天管理器，协调多个智能体之间的对话。
        配置了最大轮数限制和自动说话者选择机制。
        添加了消息钩子来检查所有消息的终止条件。

    参数:
        agents_dict: 包含所有智能体的字典

    Returns:
        GroupChatManager: 群组聊天管理器

    注意事项：
        - max_round 限制最大对话轮数，防止无限循环
        - speaker_selection_method="auto" 自动选择下一个说话者
        - 通过消息钩子检查所有消息的终止条件，包括客服主管的终止消息
    """
    agents = [
        agents_dict["customer_service_agent"],
        agents_dict["order_query_agent"],
        agents_dict["logistics_agent"],
        agents_dict["summary_agent"],
        agents_dict["user_proxy"]
    ]

    # 定义终止条件检查函数（检查所有消息）
    def check_termination_in_message(msg):
        """
        检查消息中是否包含终止条件

        这个函数会被注册为钩子，检查所有发送的消息。
        如果检测到终止关键词，会在 GroupChat 的消息历史中标记。
        """
        content = msg.get("content", "").strip()
        if not content:
            return msg

        # 检查终止关键词
        termination_keywords = [
            "TERMINATE",
            "terminate",
            "问题已解决",
            "问题已处理完毕",
            "咨询已完成",
            "问题已答复",
        ]

        content_upper = content.upper()
        content_lower = content.lower()

        for keyword in termination_keywords:
            if keyword.upper() in content_upper or keyword.lower() in content_lower:
                sender_name = msg.get("name", "unknown")
                logger.info(f"✅ 检测到终止关键词: '{keyword}'，来自: {sender_name}")
                # 在消息中添加终止标记
                msg["terminate"] = True
                return msg

        return msg

    # 创建群组聊天
    # max_round: 最大对话轮数，防止无限循环
    # 注意：不设置 max_round 限制，而是通过消息钩子机制防止客服主管重复发送
    # 客服主管在发送包含 TERMINATE 的消息后，会被消息钩子阻止再次发送
    groupchat = autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=settings.AUTOGEN_MAX_ROUNDS,  # 使用配置中的最大轮数，作为安全限制
        speaker_selection_method="auto"  # 自动选择下一个说话者
    )

    # 关键修复：将 groupchat 引用传递给 is_termination_msg，使其能够检查消息历史
    if "groupchat_ref" in agents_dict:
        agents_dict["groupchat_ref"]["groupchat"] = groupchat
        logger.debug("✅ 已将 groupchat 引用传递给 is_termination_msg")

    # 创建群组聊天管理器
    # 负责协调智能体之间的对话流程
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=agents_dict["llm_config"]
    )

    # 关键修复：在 GroupChatManager 选择说话者之前检查终止条件
    # 对于 AutoGen 0.9.0，我们需要拦截 select_speaker 方法
    # 保存原始的方法引用
    original_select_speaker = None
    original_groupchat_select_speaker = None

    # 检查 GroupChatManager 是否有 select_speaker 方法
    if hasattr(manager, 'select_speaker'):
        original_select_speaker = manager.select_speaker
    elif hasattr(manager, '_select_speaker'):
        original_select_speaker = manager._select_speaker

    # 检查 GroupChat 是否有 select_speaker 方法
    if hasattr(groupchat, 'select_speaker'):
        original_groupchat_select_speaker = groupchat.select_speaker
    elif hasattr(groupchat, '_select_speaker'):
        original_groupchat_select_speaker = groupchat._select_speaker

    def select_speaker_with_termination_check(*args, **kwargs):
        """
        选择下一个说话者，并在选择前检查终止条件

        如果检测到终止消息，直接返回 user_proxy，让它检查终止条件并停止对话。
        """
        # 检查 GroupChat 的消息历史中是否有终止消息
        if hasattr(groupchat, 'messages') and groupchat.messages:
            # 检查最后几条消息（特别是来自客服主管的消息）
            for msg in reversed(groupchat.messages[-5:]):  # 检查最后5条消息
                # 检查消息中的 terminate 标记
                if isinstance(msg, dict) and msg.get("terminate") is True:
                    sender_name = msg.get("name", "unknown")
                    logger.info(f"🛑 在选择说话者前检测到终止标记，来自: {sender_name}")
                    logger.info(f"🛑 返回 user_proxy，让它检查终止条件并停止对话")
                    # 直接返回 user_proxy，让它检查终止条件
                    return agents_dict["user_proxy"]

                # 检查消息内容中的终止关键词
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if content:
                        sender_name = msg.get("name", "unknown")
                        # 特别检查来自客服主管的消息
                        if sender_name == "客服主管" or "客服主管" in str(sender_name):
                            content_upper = str(content).upper()
                            content_lower = str(content).lower()
                            termination_keywords = ["TERMINATE", "terminate", "问题已解决"]

                            for keyword in termination_keywords:
                                if keyword.upper() in content_upper or keyword.lower() in content_lower:
                                    logger.info(f"🛑 在选择说话者前检测到终止关键词: '{keyword}'，来自: {sender_name}")
                                    logger.info(f"🛑 返回 user_proxy，让它检查终止条件并停止对话")
                                    # 直接返回 user_proxy，让它检查终止条件
                                    return agents_dict["user_proxy"]

        # 如果没有检测到终止条件，使用原始的选择方法
        if original_select_speaker:
            try:
                return original_select_speaker(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ 调用原始 select_speaker 时出错: {e}")
                return None
        elif original_groupchat_select_speaker:
            try:
                return original_groupchat_select_speaker(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ 调用原始 groupchat select_speaker 时出错: {e}")
                return None
        else:
            # 如果没有原始方法，返回 None（让 GroupChat 使用默认逻辑）
            return None

    # 尝试替换 select_speaker 方法
    try:
        if hasattr(manager, 'select_speaker'):
            manager.select_speaker = select_speaker_with_termination_check
            logger.debug("✅ 已安装终止检查钩子到 GroupChatManager.select_speaker")
        elif hasattr(manager, '_select_speaker'):
            manager._select_speaker = select_speaker_with_termination_check
            logger.debug("✅ 已安装终止检查钩子到 GroupChatManager._select_speaker")
        elif hasattr(groupchat, 'select_speaker'):
            groupchat.select_speaker = select_speaker_with_termination_check
            logger.debug("✅ 已安装终止检查钩子到 GroupChat.select_speaker")
        elif hasattr(groupchat, '_select_speaker'):
            groupchat._select_speaker = select_speaker_with_termination_check
            logger.debug("✅ 已安装终止检查钩子到 GroupChat._select_speaker")
        else:
            logger.warning("⚠️ 无法找到 select_speaker 方法，终止检查可能无法正常工作")
            # 尝试查找其他可能的方法名
            logger.debug(f"GroupChatManager 的方法: {[m for m in dir(manager) if 'select' in m.lower()]}")
            logger.debug(f"GroupChat 的方法: {[m for m in dir(groupchat) if 'select' in m.lower()]}")
    except Exception as e:
        logger.warning(f"⚠️ 无法安装终止检查钩子: {e}")

    # 关键修复：在 GroupChat 的消息历史中检查终止条件
    # 如果检测到终止消息，通过修改 GroupChat 的 max_round 来强制停止
    # 同时，在每次消息被添加到历史记录后立即检查终止条件

    # 保存原始的 append 方法（如果存在），用于在消息被添加后检查终止条件
    original_append = None
    if hasattr(groupchat.messages, 'append'):
        original_append = groupchat.messages.append

    def append_with_termination_check(msg):
        """
        在消息被添加到历史记录后检查终止条件

        如果检测到终止消息，立即修改 max_round 来强制停止，并确保 user_proxy 收到终止消息。
        """
        # 先添加消息
        if original_append:
            original_append(msg)
        else:
            groupchat.messages.append(msg)

        # 检查刚添加的消息是否是终止消息
        should_terminate = False
        if isinstance(msg, dict):
            # 检查消息中的 terminate 标记
            if msg.get("terminate") is True:
                should_terminate = True
                sender_name = msg.get("name", "unknown")
                logger.info(f"🛑 在消息被添加后检测到终止标记，来自: {sender_name}")

            # 检查消息内容中的终止关键词
            if not should_terminate:
                content = msg.get("content", "")
                if content:
                    sender_name = msg.get("name", "unknown")
                    # 特别检查来自客服主管的消息
                    if sender_name == "客服主管" or "客服主管" in str(sender_name):
                        content_upper = str(content).upper()
                        content_lower = str(content).lower()
                        termination_keywords = ["TERMINATE", "terminate", "问题已解决"]

                        for keyword in termination_keywords:
                            if keyword.upper() in content_upper or keyword.lower() in content_lower:
                                should_terminate = True
                                logger.info(f"🛑 在消息被添加后检测到终止关键词: '{keyword}'，来自: {sender_name}")
                                break

        # 如果检测到终止消息，立即强制停止
        if should_terminate:
            logger.info(f"🛑 立即强制停止 GroupChat")
            # 通过设置 max_round 为当前轮数来强制停止
            current_round = len(groupchat.messages)
            # 关键修复：将 max_round 设置为当前轮数，这样 GroupChat 在选择下一个说话者时会检查 max_round
            # 如果当前轮数已经达到或超过 max_round，GroupChat 应该停止选择说话者
            groupchat.max_round = current_round
            logger.info(f"🛑 已将 max_round 设置为 {current_round}，强制停止对话")

            # 额外修复：尝试从 agents 列表中临时移除客服主管，防止它再次被选择
            # 注意：这可能需要根据 AutoGen 的版本调整
            try:
                summary_agent = agents_dict["summary_agent"]
                if hasattr(groupchat, 'agents') and summary_agent in groupchat.agents:
                    # 创建一个新的 agents 列表，不包含客服主管
                    new_agents = [agent for agent in groupchat.agents if agent != summary_agent]
                    # 尝试替换 agents 列表
                    if hasattr(groupchat, 'agents'):
                        # 注意：这可能不适用于所有版本的 AutoGen
                        # 如果失败，至少 max_round 已经设置，应该能够阻止继续
                        logger.debug(f"🛑 尝试从 agents 列表中移除客服主管，防止它再次被选择")
            except Exception as e:
                logger.debug(f"无法从 agents 列表中移除客服主管: {e}")

    # 尝试替换 append 方法
    try:
        if hasattr(groupchat.messages, 'append'):
            groupchat.messages.append = append_with_termination_check
            logger.debug("✅ 已安装终止检查钩子到 GroupChat.messages.append")
        else:
            # 如果 append 不存在，尝试其他方法
            # 在 GroupChat 的 _append_message 方法中检查（如果存在）
            if hasattr(groupchat, '_append_message'):
                original_append_message = groupchat._append_message
                def _append_message_with_termination_check(msg):
                    result = original_append_message(msg)
                    # 检查终止条件
                    if isinstance(msg, dict) and msg.get("terminate") is True:
                        current_round = len(groupchat.messages)
                        groupchat.max_round = current_round
                        logger.info(f"🛑 已将 max_round 设置为 {current_round}，强制停止对话")
                    return result
                groupchat._append_message = _append_message_with_termination_check
                logger.debug("✅ 已安装终止检查钩子到 GroupChat._append_message")
    except Exception as e:
        logger.warning(f"⚠️ 无法安装终止检查钩子: {e}")

    # 注意：GroupChatManager 没有 select_speaker 属性，终止条件主要通过以下方式实现：
    # 1. user_proxy 的 is_termination_msg 检查发送给它的消息
    # 2. max_round 限制最大对话轮数（使用配置中的值，作为安全限制）
    # 3. 消息钩子标记终止消息
    # 4. 客服主管的 system_message 要求明确包含 TERMINATE
    # 5. **关键机制：通过全局变量 _summary_agent_terminated 跟踪客服主管是否已发送终止消息**
    #    如果客服主管已经发送过包含 TERMINATE 的消息，消息钩子会阻止它再次发送
    # 6. **新增：在 GroupChat 的消息历史中检查终止条件，如果检测到，强制停止对话**

    return manager