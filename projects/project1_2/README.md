# AutoGen 多智能体客服系统

基于 AutoGen 框架构建的多智能体协同客服系统，通过多个智能体分工协作来处理客户服务问题。

## 📋 项目简介

本项目是一个智能客服系统，使用 AutoGen 框架实现多个智能体的协同工作。系统能够自动处理客户关于订单状态和物流信息的查询，并提供智能化的回复服务。

### 核心功能

- **订单状态查询** (Agent A - 订单查询专员)
  - 从客户查询中提取订单号
  - 调用 FastAPI 模拟服务查询订单详细信息
  - 解释订单状态和处理进度

- **物流信息检查** (Agent B - 物流跟踪专员)
  - 查询包裹物流状态和位置
  - 提供准确的配送时间预估
  - 处理配送异常和延误问题

- **结果汇总回复** (Agent C - 客服主管)
  - 整合订单和物流信息
  - 生成完整的问题解答
  - 确保客户得到满意的答复

- **自动重试机制**
  - 支持指数级退避的网络请求重试
  - 自动处理网络失败和临时错误
  - 提高系统稳定性和可靠性

- **详细交互展示**
  - 实时显示智能体之间的交互过程
  - 使用 Rich 库提供美观的命令行界面
  - 记录详细的日志信息

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 或 uv（包管理器）

### 安装步骤

1. **克隆项目**（如果从仓库获取）

```bash
git clone <repository-url>
cd project1_2
```

2. **创建虚拟环境**（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置环境变量**

创建 `.env` 文件（参考 `.env.example`）：

```bash
# DeepSeek API 配置（必需）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# 向后兼容：如果设置了 OPENAI_API_KEY，也会被使用（优先使用 DEEPSEEK_API_KEY）
# OPENAI_API_KEY=your_openai_api_key_here  # 可选

# AutoGen 配置（可选，会使用 DEEPSEEK_API_KEY 作为默认值）
AUTOGEN_API_KEY=your_deepseek_api_key_here  # 可选
AUTOGEN_MODEL=deepseek-chat
AUTOGEN_BASE_URL=https://api.deepseek.com/v1

# 日志配置（可选）
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### 运行项目

#### 基础查询（不使用 AutoGen）

```bash
python main.py --query "我的订单ORD001为什么还没发货？"
```

#### 使用 AutoGen 智能体处理

```bash
python main.py --query "我的订单ORD001为什么还没发货？" --use_autogen
```

#### 指定订单ID

```bash
python main.py --order_id ORD002 --use_autogen
```

## 📁 项目结构

```
project1_2/
├── main.py                      # 项目主入口文件
├── requirements.txt             # Python 依赖包列表
├── README.md                    # 项目说明文档（本文档）
├── 架构图和流程图.md           # 系统架构图和流程图
├── 项目结构说明文档.md          # 详细的项目结构说明
├── 项目描述.txt                 # 项目需求和场景描述
├── agents/                      # 智能体模块目录
│   └── autogen_agents.py       # AutoGen 智能体实现
├── api/                         # API 服务模块目录
│   └── fastapi_server.py       # FastAPI 模拟服务器
├── config/                      # 配置模块目录
│   └── settings.py             # 项目配置管理
├── core/                        # 核心功能模块目录
│   └── logger.py               # 日志系统配置
├── tools/                       # 工具模块目录
│   └── api_client.py           # API 客户端工具
├── utils/                       # 工具函数目录
│   └── retry.py                # 重试机制实现
└── logs/                        # 日志文件目录
    └── app.log                  # 应用日志文件
```

## 🔧 技术栈

### 核心框架

- **AutoGen (>=0.2.0)**: 多智能体框架，负责智能体的创建、管理和协调
- **FastAPI (>=0.104.0)**: 现代化的 Web API 框架，用于构建模拟内部系统
- **Uvicorn (>=0.24.0)**: ASGI 服务器，用于运行 FastAPI 应用

### 数据处理

- **Pydantic (>=2.5.0)**: 数据验证和序列化
- **Pydantic-settings (>=2.1.0)**: 配置管理

### 异步和网络

- **aiofiles (>=23.2.0)**: 异步文件操作
- **httpx (>=0.25.0)**: 现代化的异步 HTTP 客户端
- **tenacity (>=8.2.0)**: 重试机制库（指数退避）

### AI 和语言模型

- **DeepSeek API**: LLM 服务提供商（兼容 OpenAI API 格式）
- **openai (>=1.3.0)**: OpenAI API 客户端（用于调用 DeepSeek API）

### 用户界面和工具

- **Rich (>=13.7.0)**: 丰富的命令行界面库
- **Colorama (>=0.4.6)**: 跨平台彩色终端输出
- **Python-dotenv (>=1.0.0)**: 环境变量管理

### 开发和测试

- **Pytest (>=7.4.0)**: 测试框架
- **Pytest-asyncio (>=0.21.0)**: 异步测试支持
- **Black (>=23.11.0)**: 代码格式化工具

## 🎯 使用示例

### 示例 1：查询订单状态

```bash
python main.py --query "我的订单ORD001为什么还没发货？" --use_autogen
```

**输出示例：**

```
🤖 启动AutoGen智能体处理查询
客户查询: 我的订单ORD001为什么还没发货？

🚀 正在初始化AutoGen智能体团队...

✅ AutoGen智能体团队创建完成！

🚀 开始智能体协作处理...

[客服接待员] 开始执行任务
[订单查询专员] 开始查询订单: ORD001
[订单查询专员] 订单查询成功: ORD001
[客服主管] 生成完整回复...

✅ AutoGen智能体处理完成
```

### 示例 2：查询物流信息

```bash
python main.py --query "我的订单ORD002物流状态如何？" --use_autogen
```

### 示例 3：综合查询

```bash
python main.py --query "我的订单ORD001什么时候能到？" --use_autogen
```

## 📖 详细文档

- **[架构图和流程图.md](./架构图和流程图.md)**: 系统架构图、流程图和交互图
- **[项目结构说明文档.md](./项目结构说明文档.md)**: 详细的项目结构说明和模块介绍

## 🔑 API 密钥获取

### DeepSeek API

1. 访问 [DeepSeek 官网](https://platform.deepseek.com/) 或 [DeepSeek 官网](https://www.deepseek.com/)
2. 注册账号并完成验证
3. 在 API Keys 页面创建新密钥
4. 注意：DeepSeek API 兼容 OpenAI API 格式，可以直接使用 `openai` 包

### 向后兼容 OpenAI API

如果暂时无法获取 DeepSeek API 密钥，可以使用 OpenAI API 密钥（向后兼容）：

1. 访问 [OpenAI 官网](https://platform.openai.com/)
2. 注册账号并完成验证
3. 在 API Keys 页面创建新密钥
4. 在 `.env` 文件中设置 `OPENAI_API_KEY`

## 🐛 常见问题

### Q1: DeepSeek API 连接失败

**问题**: `openai.error.AuthenticationError: Invalid API key`

**解决方案**:
1. 检查 `.env` 文件中的 `DEEPSEEK_API_KEY` 是否正确
2. 确认 DeepSeek API 密钥有效且有足够的配额
3. 检查 `DEEPSEEK_API_BASE` 是否正确（默认: https://api.deepseek.com/v1）
4. 检查网络连接和防火墙设置
5. 如果使用代理，确保代理配置正确

### Q2: FastAPI 服务启动失败

**问题**: 端口 8000 已被占用

**解决方案**:
1. 修改 `config/settings.py` 中的 `MOCK_SERVER_PORT` 配置
2. 或关闭占用 8000 端口的其他服务
3. 检查防火墙设置

### Q3: 智能体无法调用工具函数

**问题**: 工具函数调用失败

**解决方案**:
1. 确认 FastAPI 服务已启动
2. 检查 API 客户端配置（`tools/api_client.py`）
3. 查看日志文件了解详细错误信息
4. 确认网络连接正常

### Q4: 重试机制不工作

**问题**: 网络错误时没有自动重试

**解决方案**:
1. 检查 `utils/retry.py` 中的重试配置
2. 确认 `tenacity` 库已正确安装
3. 查看日志文件了解重试详情
4. 调整重试参数（最大重试次数、等待时间等）

## 🧪 测试

运行测试：

```bash
pytest tests/
```

运行特定测试：

```bash
pytest tests/test_api_client.py
```

## 📝 开发指南

### 添加新的智能体

1. 在 `agents/autogen_agents.py` 中定义新的智能体
2. 配置智能体的 `system_message` 和 `llm_config`
3. 在 `create_group_chat` 函数中添加新智能体
4. 注册必要的工具函数

### 添加新的工具函数

1. 在 `tools/` 目录下创建新的工具模块
2. 实现工具函数（同步或异步）
3. 在 `agents/autogen_agents.py` 中使用 `autogen.register_function` 注册工具
4. 在智能体的 `system_message` 中说明如何使用工具

### 修改重试策略

1. 编辑 `utils/retry.py` 中的重试配置
2. 调整 `max_attempts`、`min_wait`、`max_wait`、`multiplier` 等参数
3. 修改 `should_retry_http_error` 函数来调整重试条件

## 📄 许可证

本项目采用 MIT 许可证。

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

如有问题或建议，请通过项目 Issues 页面反馈。

---

**项目版本**: v1.0.0
**最后更新**: 2024年1月
**维护者**: AutoGen 多智能体客服系统开发团队
