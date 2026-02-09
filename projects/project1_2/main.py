#!/usr/bin/env python3
"""
AutoGen 多智能体客服系统 - 主入口文件

项目简介：
    基于 AutoGen 框架实现的多智能体协同客服系统，通过多个智能体分工协作
    来处理客户服务问题。系统能够自动处理客户关于订单状态和物流信息的查询，
    并提供智能化的回复服务。

功能特点：
    1. 订单状态查询 (Agent A - 订单查询专员)
       - 从客户查询中提取订单号
       - 调用 FastAPI 模拟服务查询订单详细信息
       - 解释订单状态和处理进度

    2. 物流信息检查 (Agent B - 物流跟踪专员)
       - 查询包裹物流状态和位置
       - 提供准确的配送时间预估
       - 处理配送异常和延误问题

    3. 结果汇总回复 (Agent C - 客服主管)
       - 整合订单和物流信息
       - 生成完整的问题解答
       - 确保客户得到满意的答复

    4. 自动重试机制
       - 支持指数级退避的网络请求重试
       - 自动处理网络失败和临时错误
       - 提高系统稳定性和可靠性

    5. 详细交互展示
       - 实时显示智能体之间的交互过程
       - 使用 Rich 库提供美观的命令行界面
       - 记录详细的日志信息

技术架构：
    - 前端：命令行界面（Rich）
    - 智能体框架：AutoGen
    - LLM服务：DeepSeek API（兼容 OpenAI API 格式）
    - 后端服务：FastAPI（模拟内部系统）
    - 重试机制：Tenacity（指数退避）

使用方法:
    基础查询（不使用 AutoGen）:
        python main.py --query "我的订单ORD001为什么还没发货？"

    使用 AutoGen 智能体处理:
        python main.py --query "我的订单ORD001为什么还没发货？" --use_autogen

    指定订单ID:
        python main.py --order_id ORD002 --use_autogen

    交互式模式（持续接收用户输入）:
        python main.py --interactive
        python main.py --interactive --use_autogen  # 使用AutoGen智能体

环境要求:
    - Python 3.8+
    - 已安装所有依赖包（requirements.txt）
    - 配置了 DEEPSEEK_API_KEY 环境变量（或 OPENAI_API_KEY 用于向后兼容）

作者: AutoGen 多智能体客服系统开发团队
版本: 1.0.0
"""

import sys
import os
import asyncio
from pathlib import Path
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
import re

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from api.fastapi_server import start_server
from config.settings import settings
from core.logger import setup_logger
from tools.api_client import APIClient
from agents.autogen_agents import create_autogen_agents, create_group_chat

# 设置日志
logger = setup_logger(__name__)
console = Console()

# 加载环境变量
load_dotenv()

def parse_arguments():
    """
    解析命令行参数

    功能说明：
        解析用户从命令行传入的参数，包括查询内容、订单ID和是否使用AutoGen智能体。
        如果没有提供查询内容，会使用默认查询模板。

    Returns:
        argparse.Namespace: 包含解析后的命令行参数的命名空间对象

    参数说明：
        --query: 客户查询内容（可选，如果不提供则使用默认查询）
        --order_id: 订单ID（默认值：ORD001）
        --use_autogen: 是否使用AutoGen智能体处理查询（标志参数，不需要值）

    示例：
        >>> args = parse_arguments()
        >>> args.query  # "我的订单ORD001为什么还没发货？"
        >>> args.order_id  # "ORD001"
        >>> args.use_autogen  # True 或 False
    """
    parser = argparse.ArgumentParser(
        description="AutoGen 多智能体客服系统",
        epilog="示例: python main.py --query '我的订单ORD001为什么还没发货？' --use_autogen"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="客户查询内容（例如：'我的订单ORD001为什么还没发货？'）"
    )
    parser.add_argument(
        "--order_id",
        type=str,
        default="ORD001",
        help="订单ID（默认值：ORD001，格式：ORD + 数字）"
    )
    parser.add_argument(
        "--use_autogen",
        action="store_true",
        help="使用AutoGen智能体处理查询（如果未指定，则使用基础查询测试）"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式：持续接收用户输入并处理查询（输入 'quit' 或 'exit' 退出）"
    )
    return parser.parse_args()

async def start_services():
    """
    启动模拟服务（FastAPI服务器）

    功能说明：
        在后台启动 FastAPI 模拟服务器，提供订单状态和物流信息查询接口。
        服务器运行在本地 127.0.0.1:8000，用于模拟内部系统的API服务。

    工作流程：
        1. 使用 subprocess 在子进程中启动 uvicorn 服务器
        2. 等待 3 秒确保服务完全启动
        3. 记录启动日志
        4. 返回进程对象，用于后续关闭服务

    Returns:
        subprocess.Popen: FastAPI 服务器的进程对象，用于后续关闭服务

    注意事项：
        - 服务器运行在独立进程中，不会阻塞主程序
        - 程序退出前需要调用 process.terminate() 关闭服务
        - 如果服务启动失败，会在日志中记录错误信息

    示例：
        >>> server_process = await start_services()
        >>> # ... 使用服务 ...
        >>> server_process.terminate()  # 关闭服务
    """
    # 使用子进程启动FastAPI服务器
    # 这样可以避免阻塞主程序，同时保持服务独立运行
    import subprocess
    import sys

    # 构建启动命令
    # -m uvicorn: 使用 uvicorn 模块运行 FastAPI 应用
    # api.fastapi_server:app: 指定 FastAPI 应用的位置
    # --host 127.0.0.1: 绑定到本地回环地址
    # --port 8001: 监听 8001 端口
    cmd = [
        sys.executable,  # Python 解释器路径
        "-m", "uvicorn",  # 使用 uvicorn 模块
        "api.fastapi_server:app",  # FastAPI 应用路径
        "--host", "127.0.0.1",  # 服务器地址
        "--port", "8001"  # 服务器端口
    ]

    # 在项目根目录下启动服务器进程
    process = subprocess.Popen(cmd, cwd=str(project_root))

    # 等待服务启动（给服务器足够的时间完成初始化）
    # 实际生产环境中可以使用健康检查接口来确认服务就绪
    await asyncio.sleep(3)
    logger.info("FastAPI模拟服务已启动 - http://127.0.0.1:8001")
    return process

def display_query_results(order_info: dict, logistics_info: dict):
    """展示查询成功的结果"""
    console.print("\n" + "="*80)
    console.print("[bold green]🎉 查询结果展示 🎉[/bold green]", justify="center")
    console.print("="*80)

    # 创建订单信息表格
    if order_info and "error" not in order_info:
        order_table = Table(title="📦 订单信息", box=box.ROUNDED, border_style="blue")
        order_table.add_column("项目", style="cyan", no_wrap=True)
        order_table.add_column("详情", style="white")

        order_table.add_row("订单ID", order_info.get('order_id', 'N/A'))
        order_table.add_row("订单状态", f"[bold green]{order_info.get('status', 'N/A')}[/bold green]")
        order_table.add_row("客户姓名", order_info.get('customer_name', 'N/A'))
        order_table.add_row("订单金额", f"[bold yellow]¥{order_info.get('total_amount', 0)}[/bold yellow]")
        order_table.add_row("商品列表", ', '.join(order_info.get('items', [])))
        order_table.add_row("收货地址", order_info.get('shipping_address', 'N/A'))
        order_table.add_row("创建时间", order_info.get('created_at', 'N/A'))
        order_table.add_row("更新时间", order_info.get('updated_at', 'N/A'))

        console.print(order_table)
        console.print()

        # 记录订单查询成功日志
        logger.info(f"✅ 订单查询成功展示 - 订单ID: {order_info.get('order_id')}, 状态: {order_info.get('status')}")
    else:
        error_msg = order_info.get('error', '未知错误') if order_info else '订单查询失败'
        console.print(Panel(f"[bold red]❌ 订单查询失败: {error_msg}[/bold red]", border_style="red"))
        logger.warning(f"订单查询失败: {error_msg}")

    # 创建物流信息表格
    if logistics_info and "error" not in logistics_info:
        logistics_table = Table(title="🚚 物流信息", box=box.ROUNDED, border_style="green")
        logistics_table.add_column("项目", style="cyan", no_wrap=True)
        logistics_table.add_column("详情", style="white")

        logistics_table.add_row("物流单号", logistics_info.get('tracking_number', '暂未分配'))
        logistics_table.add_row("物流状态", f"[bold green]{logistics_info.get('status', 'N/A')}[/bold green]")
        logistics_table.add_row("当前位置", logistics_info.get('current_location', 'N/A'))
        logistics_table.add_row("承运商", logistics_info.get('carrier', 'N/A'))
        logistics_table.add_row("预计送达", f"[bold yellow]{logistics_info.get('estimated_delivery', '未确定')}[/bold yellow]")

        # 显示物流轨迹
        if logistics_info.get('tracking_history'):
            tracking_text = ""
            for record in logistics_info['tracking_history']:
                tracking_text += f"{record.get('time', 'N/A')} - {record.get('location', 'N/A')}: {record.get('status', 'N/A')}\n"
            logistics_table.add_row("物流轨迹", tracking_text.strip())

        console.print(logistics_table)
        console.print()

        # 记录物流查询成功日志
        logger.info(f"✅ 物流查询成功展示 - 单号: {logistics_info.get('tracking_number')}, 状态: {logistics_info.get('status')}")
    else:
        error_msg = logistics_info.get('error', '未知错误') if logistics_info else '物流查询失败'
        console.print(Panel(f"[bold yellow]⚠️ 物流查询失败: {error_msg}[/bold yellow]", border_style="yellow"))
        logger.warning(f"物流查询失败: {error_msg}")

    console.print("="*80)
    console.print("[bold green]✨ 查询结果展示完成 ✨[/bold green]", justify="center")
    console.print("="*80 + "\n")

async def run_autogen_query(query: str, agents_dict=None, manager=None):
    """
    使用AutoGen智能体处理查询

    功能说明：
        使用 AutoGen 智能体系统处理用户查询。
        支持传入已创建的智能体和管理器，避免重复初始化。

    参数:
        query (str): 用户查询内容
        agents_dict (dict, optional): 已创建的智能体字典，如果为None则创建新的
        manager (GroupChatManager, optional): 已创建的群组聊天管理器，如果为None则创建新的

    Returns:
        result: AutoGen 处理结果，如果失败则返回 None
    """
    # 导入重置函数，用于重置客服主管的终止状态
    from agents.autogen_agents import reset_summary_agent_terminated

    # 在每次新查询开始时重置全局变量，确保客服主管可以正常发送消息
    reset_summary_agent_terminated()

    console.print(Panel(f"[bold cyan]🤖 启动AutoGen智能体处理查询[/bold cyan]", border_style="cyan"))
    console.print(Panel(f"[bold green]客户查询:[/bold green] {query}", border_style="green"))

    try:
        # 如果没有传入智能体，则创建新的
        if agents_dict is None or manager is None:
            agents_dict = create_autogen_agents()
            manager = create_group_chat(agents_dict)

        # 启动群组聊天
        console.print(Panel("[bold yellow]🚀 开始智能体协作处理...[/bold yellow]", border_style="yellow"))

        # 发起对话
        # max_turns: 最大对话轮数，防止无限循环
        # 注意：终止条件由 user_proxy 的 is_termination_msg 控制
        # 通过消息钩子机制防止客服主管重复发送，而不是依赖 max_turns 限制
        result = agents_dict["user_proxy"].initiate_chat(
            manager,
            message=query,
            max_turns=settings.AUTOGEN_MAX_ROUNDS  # 使用配置中的最大轮数，作为安全限制
        )

        console.print(Panel("[bold green]✅ AutoGen智能体处理完成[/bold green]", border_style="green"))
        return result

    except Exception as e:
        error_msg = f"AutoGen智能体处理失败: {str(e)}"
        console.print(Panel(f"[bold red]❌ {error_msg}[/bold red]", border_style="red"))
        logger.error(error_msg, exc_info=True)
        return None

async def run_query_test(query: str, order_id: str = "ORD001"):
    """运行查询测试"""
    console.print(Panel(f"[bold green]客户查询:[/bold green] {query}", border_style="green"))

    # 创建API客户端
    client = APIClient()

    # 测试订单查询
    console.print(Panel("[bold blue]测试订单查询API[/bold blue]", border_style="blue"))
    order_info = await client.get_order_status(order_id)

    # 测试物流查询
    console.print(Panel("[bold blue]测试物流查询API[/bold blue]", border_style="blue"))
    logistics_info = await client.get_logistics_info(order_id)

    # 展示查询成功的结果
    display_query_results(order_info, logistics_info)

    console.print(Panel("[bold green]✅ 测试完成[/bold green]", border_style="green"))

    return order_info, logistics_info

async def run_interactive_mode(use_autogen: bool = False):
    """
    运行交互式模式

    功能说明：
        持续接收用户输入并处理查询，直到用户输入退出命令。
        支持多次查询，每次查询都会调用相应的处理函数。

    参数:
        use_autogen (bool): 是否使用AutoGen智能体处理查询
            - True: 使用AutoGen智能体处理
            - False: 使用基础查询测试

    退出命令:
        - 'quit': 退出交互式模式
        - 'exit': 退出交互式模式
        - 'q': 退出交互式模式
        - Ctrl+C: 强制退出
    """
    console.print(Panel(
        "[bold cyan]🎯 进入交互式模式[/bold cyan]\n"
        "[yellow]提示：[/yellow]\n"
        "  - 输入您的查询问题，按 Enter 提交\n"
        "  - 输入 'quit'、'exit' 或 'q' 退出\n"
        "  - 使用 Ctrl+C 强制退出",
        border_style="cyan",
        title="交互式客服系统"
    ))

    # 启动模拟服务（在整个交互式会话期间保持运行）
    server_process = await start_services()

    try:
        # 如果使用AutoGen，预先创建智能体（避免每次查询都重新创建）
        agents_dict = None
        manager = None
        if use_autogen:
            console.print(Panel("[bold yellow]⏳ 正在初始化AutoGen智能体...[/bold yellow]", border_style="yellow"))
            agents_dict = create_autogen_agents()
            manager = create_group_chat(agents_dict)
            console.print(Panel("[bold green]✅ AutoGen智能体初始化完成[/bold green]", border_style="green"))

        query_count = 0

        while True:
            try:
                # 获取用户输入
                console.print()
                query = console.input("[bold green]请输入您的查询（输入 'quit' 退出）: [/bold green]")

                # 检查退出命令
                if query.lower() in ['quit', 'exit', 'q', '退出']:
                    console.print(Panel("[bold yellow]👋 感谢使用，再见！[/bold yellow]", border_style="yellow"))
                    break

                # 检查空输入
                if not query.strip():
                    console.print(Panel("[bold yellow]⚠️ 请输入有效的查询内容[/bold yellow]", border_style="yellow"))
                    continue

                query_count += 1
                console.print(f"\n[bold cyan]📋 查询 #{query_count}[/bold cyan]")

                # 从查询中提取订单ID
                order_id = extract_order_id_from_query(query) or "ORD001"

                # 处理查询
                if use_autogen:
                    # 使用AutoGen智能体处理（复用已创建的智能体）
                    await run_autogen_query(query, agents_dict=agents_dict, manager=manager)
                else:
                    # 运行基础查询测试
                    await run_query_test(query, order_id)

            except KeyboardInterrupt:
                # 处理 Ctrl+C
                console.print("\n[bold yellow]⚠️ 检测到中断信号，正在退出...[/bold yellow]")
                break
            except Exception as e:
                # 处理其他异常
                error_msg = f"处理查询时发生错误: {str(e)}"
                console.print(Panel(f"[bold red]❌ {error_msg}[/bold red]", border_style="red"))
                logger.error(error_msg, exc_info=True)

    finally:
        # 关闭服务器进程
        if server_process:
            server_process.terminate()
            logger.info("FastAPI模拟服务已关闭")

async def main_async():
    """
    异步主函数

    功能说明：
        主程序入口，解析命令行参数并根据参数执行相应的操作。
        支持单次查询和交互式模式两种运行方式。
    """
    args = parse_arguments()

    # 交互式模式
    if args.interactive:
        await run_interactive_mode(use_autogen=args.use_autogen)
        return 0

    # 单次查询模式
    # 如果没有提供查询，使用默认查询
    query = args.query or f"我的订单{args.order_id}为什么还没发货？"

    # 从查询中提取订单ID，如果提取失败则使用命令行参数或默认值
    extracted_order_id = extract_order_id_from_query(query) if args.query else None
    order_id = extracted_order_id or args.order_id or "ORD001"

    logger.info(f"📋 最终使用的订单ID: {order_id}")

    # 启动模拟服务
    server_process = await start_services()

    try:
        if args.use_autogen:
            # 使用AutoGen智能体处理查询
            await run_autogen_query(query)
        else:
            # 运行基础查询测试
            await run_query_test(query, order_id)
    finally:
        # 关闭服务器进程
        if server_process:
            server_process.terminate()
            logger.info("FastAPI模拟服务已关闭")

    return 0

def extract_order_id_from_query(query: str) -> str:
    """
    从查询文本中提取订单ID

    功能说明：
        使用正则表达式从用户查询文本中提取订单ID。
        订单ID格式为：ORD + 数字（例如：ORD001、ORD002、ORD123）。
        如果未找到订单ID，则返回默认值 ORD001。

    参数:
        query (str): 用户查询文本，可能包含订单ID

    Returns:
        str: 提取到的订单ID（大写格式），如果没有找到则返回默认值 "ORD001"

    正则表达式说明：
        - r'ORD\d+': 匹配 "ORD" 后跟一个或多个数字
        - re.IGNORECASE: 忽略大小写，可以匹配 "ord001" 或 "ORD001"
        - .upper(): 将结果转换为大写，确保格式统一

    示例:
        >>> extract_order_id_from_query("我的订单ORD001为什么还没发货？")
        'ORD001'
        >>> extract_order_id_from_query("查询订单ord002的状态")
        'ORD002'
        >>> extract_order_id_from_query("我的订单什么时候发货？")  # 没有订单ID
        'ORD001'  # 返回默认值
    """
    # 使用正则表达式匹配订单ID模式 (ORD + 数字)
    # 模式说明：
    #   - ORD: 订单ID的前缀（固定字符串）
    #   - \d+: 一个或多个数字（订单编号）
    #   - re.IGNORECASE: 忽略大小写，可以匹配 "ord001" 或 "ORD001"
    pattern = r'ORD\d+'
    match = re.search(pattern, query, re.IGNORECASE)

    if match:
        # 找到匹配的订单ID，转换为大写格式
        order_id = match.group().upper()
        logger.info(f"🔍 从查询中提取到订单ID: {order_id}")
        return order_id
    else:
        # 未找到订单ID，使用默认值并记录警告
        logger.warning(f"⚠️ 未能从查询中提取订单ID，使用默认值: ORD001")
        return "ORD001"

def main():
    """主函数"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    # 运行方法：python main.py --query  "我的订单ORD001为什么还没发货？"
    sys.exit(main())