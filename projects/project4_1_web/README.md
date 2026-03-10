# 订单查询客服系统 - Web 版本

这是一个基于 React + Node.js 的订单查询客服系统，支持多轮对话、语音输入、图像上传、RAG 知识检索和对话持久化。

## 项目结构

```
project4_1_web/
├── server/          # Node.js 后端服务
│   ├── package.json
│   ├── server.js    # 主服务器文件
│   ├── database.js  # 数据库操作
│   ├── rag.js       # RAG 检索
│   └── tools.js     # 工具函数
└── client/          # React 前端应用
    ├── package.json
    ├── public/
    │   └── index.html
    └── src/
        ├── index.js
        ├── index.css
        ├── App.js
        └── App.css
```

## 功能特性

1. **多轮对话**：支持上下文理解的连续对话
2. **对话持久化**：使用 SQLite 保存对话历史
3. **订单查询**：查询订单状态和物流信息
4. **RAG 知识检索**：基于政策知识库回答问题
5. **文件上传**：支持音频和图片上传
6. **美观的 UI**：现代化的聊天界面

## 快速开始

### 前置要求

- Node.js (v14 或更高版本)
- npm 或 yarn

### 安装依赖

#### 后端

```bash
cd server
npm install
```

#### 前端

```bash
cd client
npm install
```

### 运行项目

#### 启动后端服务

```bash
cd server
npm start
```

后端服务将在 http://localhost:3001 运行

#### 启动前端应用

```bash
cd client
npm start
```

前端应用将在 http://localhost:3000 运行

## 使用说明

### 1. 文本对话

在输入框中输入消息，点击"发送"按钮或按回车键发送消息。

### 2. 文件上传

点击"📎 上传文件"按钮，选择音频文件（.wav, .mp3）或图片文件（.jpg, .jpeg, .png）上传。

### 3. 示例对话

**查询订单：**
```
用户: 你好，我要查订单
客服: 请提供订单号。
用户: 订单号是 12345
客服: 订单 12345（Wireless Headphones）：当前状态为『shipped』。物流信息：Arrived at Beijing Sorting Center。
```

**查询政策：**
```
用户: 如果不喜欢可以退货吗？
客服: 退款政策：自签收之日起 7 天内，且商品未拆封，可发起退款申请。
物流政策：满 50 美元免邮，标准配送一般为 3-5 个工作日。
```

## 技术栈

### 后端

- **Express.js**：Web 框架
- **Better-SQLite3**：SQLite 数据库
- **Multer**：文件上传处理
- **CORS**：跨域支持

### 前端

- **React 18**：用户界面框架
- **Axios**：HTTP 客户端
- **CSS3**：样式设计

## 数据库

系统使用两个 SQLite 数据库：

1. **orders.db**：存储订单信息
2. **checkpoints.db**：存储对话检查点（多轮对话状态）

## API 接口

### POST /api/chat

发送聊天消息

**请求体：**
```json
{
  "threadId": "session_xxx",
  "message": "用户消息"
}
```

**响应：**
```json
{
  "response": "客服响应",
  "messages": [...]
}
```

### POST /api/upload

上传文件

**请求体：** FormData
- file: 文件

**响应：**
```json
{
  "filePath": "uploads/xxx",
  "filename": "original_filename.ext"
}
```

### GET /api/history/:threadId

获取对话历史

**响应：**
```json
{
  "messages": [...]
}
```

## 注意事项

1. 本项目使用模拟的 LLM，生产环境需要配置真实的 LLM API
2. 文件上传功能会在 server 目录下创建 uploads 文件夹
3. 对话历史保存在 checkpoints.db 中，使用 threadId 区分不同会话

## 扩展建议

1. 集成真实的 LLM（OpenAI GPT、通义千问等）
2. 添加用户认证功能
3. 实现真实的 ASR 和 OCR 功能
4. 添加更多订单操作（取消订单、修改地址等）
5. 实现更复杂的 RAG 检索（使用向量数据库）

## 许可证

MIT License
