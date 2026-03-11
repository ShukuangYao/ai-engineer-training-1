"""
智能客服作业入口：启动服务 或 运行客户端 demo。

在 week04-homework 根目录执行：
  python -m smart_customer_service.main server   # 启动 Agent 服务（默认 8000）
  python -m smart_customer_service.main client   # 调用服务端做一次对话 demo
  python -m smart_customer_service.main          # 默认执行 client demo
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "server":
        import uvicorn
        from smart_customer_service.agent_server import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        from smart_customer_service.client import main as client_main
        client_main()


if __name__ == "__main__":
    main()
