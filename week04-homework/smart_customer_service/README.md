# 智能客服 Agent

多轮对话 + 工具调用 + 模型/插件热更新。需配置 `DASHSCOPE_API_KEY`（通义）。

## 运行方式（在 week04-homework 根目录）

```bash
# 安装依赖
uv sync

# 启动服务（默认 http://0.0.0.0:8000）
python -m smart_customer_service.main server

# 另开终端：运行客户端 demo（健康检查 + 查订单两轮对话）
python -m smart_customer_service.main client
# 或
python -m smart_customer_service.main
```

## 客户端用法

```python
from smart_customer_service.client import AgentClient

client = AgentClient()  # 默认 http://127.0.0.1:8000
# 或 client = AgentClient("http://your-host:8000")

print(client.health())
r = client.chat("session-1", "查订单", history=[])
print(r["reply"])
# 多轮：把上一轮 user/assistant 放进 history 再发下一条
r2 = client.chat("session-1", "1234567890", history=[
    {"role": "user", "content": "查订单"},
    {"role": "assistant", "content": r["reply"]},
])
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /chat | 多轮对话（body: session_id, message, history?） |
| POST | /reload_plugins | 插件热重载 |
| POST | /reload_model | 模型热更新（重读 MODEL_NAME） |

## 目录

- `main.py` 入口（server / client）
- `agent_server.py` FastAPI 服务
- `plugin_loader.py` 插件加载与热重载
- `plugins/invoice_plugin.py` 发票开具插件
- `client.py` HTTP 客户端
