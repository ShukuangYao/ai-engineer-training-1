"""
多任务问答助手 - 主程序入口

基于 LangChain 框架构建的智能问答助手，支持：
1. 自然语言对话
2. 天气查询（通过高德地图API）
3. 信息搜索（通过Tavily搜索API）

技术架构：
- 使用 LangChain 的 LCEL 语法构建对话链
- 使用 DeepSeek API 作为底层大语言模型（兼容 OpenAI API 格式）
- 支持 Function Calling（工具自动调用）
- 维护对话历史和上下文

运行方式：
    python main.py
"""

import sys
import os
from typing import Dict, Any

# 添加项目根目录到Python路径
# 这样可以直接导入项目内的模块，而不需要设置 PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from agents.qa_agent import create_qa_agent  # 问答代理创建函数
from core.logger import app_logger  # 日志记录模块
from config.settings import settings  # 配置管理模块


def print_welcome():
    """
    打印欢迎信息

    在程序启动时显示欢迎界面，介绍系统功能和用法。
    帮助用户了解如何使用这个问答助手。
    """
    print("=" * 60)
    print("🤖 多任务问答助手")
    print("=" * 60)
    print("支持功能:")
    print("  💬 日常对话 - 例: '你好 ，AI 助手，你能为我做什么？'")
    print("  🌤️  天气查询 - 例: '查询北京天气'")
    print("  🔍 信息搜索 - 例: '搜索最新财经'")
    print()
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 60)


def main():
    """
    主函数 - 程序入口点

    负责：
    1. 验证配置（API密钥等）
    2. 创建问答代理
    3. 启动交互式对话循环
    4. 处理用户输入和显示回复
    5. 优雅地处理退出和异常

    Returns:
        int: 退出码，0表示成功，1表示失败
    """
    # 步骤1：验证配置
    # 检查所有必需的API密钥是否已配置
    if not settings.validate_all():
        print("❌ 配置验证失败，请检查环境变量配置")
        print("💡 请确保 .env 文件中已设置以下环境变量：")
        print("   - DEEPSEEK_API_KEY: DeepSeek API密钥")
        print("   - AMAP_API_KEY: 高德地图API密钥")
        print("   - TAVILY_API_KEY: Tavily搜索API密钥")
        return 1

    # 步骤2：显示欢迎信息
    print_welcome()

    # 步骤3：创建问答代理
    try:
        # 创建问答代理实例
        # 代理会自动初始化LLM、工具等组件
        agent = create_qa_agent()
        print(f"✅ 问答助手已启动 (会话ID: {agent.session_id})")
        print()

        # 步骤4：启动交互式对话循环
        while True:
            try:
                # 步骤4.1：获取用户输入
                # 使用 input() 函数从标准输入读取用户输入
                # strip() 方法去除首尾空白字符
                user_input = input("👤 您: ").strip()

                # 步骤4.2：检查退出命令
                # 支持多种退出命令：quit, exit, 退出, q
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 再见！")
                    break

                # 步骤4.3：跳过空输入
                # 如果用户只输入了空白字符，跳过本次循环
                if not user_input:
                    continue

                # 步骤4.4：处理用户输入
                # 显示"正在思考"提示，让用户知道系统正在处理
                print("🤔 正在思考...")
                # 调用代理的 chat 方法处理用户输入
                # 代理会自动分析意图、调用工具（如果需要）、生成回复
                result = agent.chat(user_input)

                # 步骤4.5：显示响应
                # 打印助手的回复
                print(f"🤖 助手: {result['response']}")

                # 步骤4.6：显示使用的工具（如果有）
                # 如果本次对话中使用了工具，显示工具列表
                if result.get('tools_used'):
                    print(f"🔧 使用工具: {', '.join(result['tools_used'])}")

                # 步骤4.7：显示处理时间
                # 显示本次处理耗时，用于性能监控
                print(f"⏱️  处理时间: {result['processing_time_ms']:.1f}ms")
                print()  # 空行，分隔不同轮次的对话

            except KeyboardInterrupt:
                # 处理 Ctrl+C 中断
                # 用户按下 Ctrl+C 时，优雅地退出程序
                print("\n👋 再见！")
                break
            except Exception as e:
                # 处理其他异常
                # 显示友好的错误信息，记录日志，继续运行（不退出程序）
                print(f"❌ 处理错误: {str(e)}")
                app_logger.error(f"处理用户输入时出错: {e}")
                continue  # 继续下一轮循环，不退出程序

        # 步骤5：结束会话
        # 用户退出时，调用代理的 end_session 方法
        # 可以在这里添加清理资源、保存历史等操作
        agent.end_session()
        return 0  # 返回成功退出码

    except Exception as e:
        # 处理启动阶段的异常
        # 如果代理创建失败或其他启动错误，显示错误信息并退出
        print(f"❌ 启动失败: {str(e)}")
        app_logger.error(f"程序启动失败: {e}")
        return 1  # 返回失败退出码


if __name__ == "__main__":
    """
    程序入口点

    当直接运行此脚本时（而不是作为模块导入），执行 main() 函数。
    使用 sys.exit() 确保退出码正确传递给操作系统。
    """
    sys.exit(main())