const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
require('dotenv').config();

const { CheckpointDatabase } = require('./database');
const { Tools } = require('./tools');

const app = express();
const PORT = process.env.PORT || 3001;

// 中间件
app.use(cors());
app.use(express.json());

// 文件上传配置
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

// 确保 uploads 目录存在
const fs = require('fs');
if (!fs.existsSync('uploads')) {
    fs.mkdirSync('uploads');
}

// 初始化数据库和工具
const checkpointDB = new CheckpointDatabase();
const tools = new Tools();

// 简单的 LLM 模拟（演示用）
class MockLLM {
    invoke(messages) {
        const lastMsg = messages[messages.length - 1].content.toLowerCase();

        if (lastMsg.includes('查订单') || lastMsg.includes('订单') || lastMsg.includes('12345') || lastMsg.includes('67890')) {
            if (lastMsg.includes('12345')) {
                return {
                    content: '',
                    toolCalls: [{ name: 'check_order', args: { order_id: '12345' }, id: 'call_1' }]
                };
            } else if (lastMsg.includes('67890')) {
                return {
                    content: '',
                    toolCalls: [{ name: 'check_order', args: { order_id: '67890' }, id: 'call_2' }]
                };
            } else {
                return { content: '请提供订单号。', toolCalls: [] };
            }
        } else if (lastMsg.includes('政策') || lastMsg.includes('退款')) {
            return {
                content: '',
                toolCalls: [{ name: 'search_policy', args: { query: lastMsg }, id: 'call_3' }]
            };
        } else {
            return { content: '我可以帮您查询订单或解答政策相关问题。', toolCalls: [] };
        }
    }
}

const llm = new MockLLM();

// 生成唯一ID
function generateId() {
    return 'uuid-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

// 对话处理
async function processMessage(threadId, userContent) {
    // 获取最新检查点
    let checkpoint = checkpointDB.getLatestCheckpoint(threadId);
    let messages = [];
    let orderId = null;

    if (checkpoint) {
        messages = checkpoint.checkpointData.messages || [];
        orderId = checkpoint.checkpointData.orderId;
    }

    // 添加用户消息
    messages.push({ role: 'user', content: userContent });

    // 检查是否是文件路径
    let processedContent = userContent;
    if (userContent.endsWith('.wav') || userContent.endsWith('.mp3')) {
        processedContent = '音频转写：' + tools.processAudioInput(userContent);
        messages[messages.length - 1].content = processedContent;
    } else if (userContent.endsWith('.jpg') || userContent.endsWith('.png')) {
        processedContent = '图片识别：' + tools.processImageInput(userContent);
        messages[messages.length - 1].content = processedContent;
    }

    // 调用 LLM
    const llmMessages = messages.map(m => ({
        content: m.content,
        role: m.role
    }));

    let response = llm.invoke(llmMessages);

    // 处理工具调用
    while (response.toolCalls && response.toolCalls.length > 0) {
        messages.push({ role: 'assistant', content: response.content, toolCalls: response.toolCalls });

        for (const toolCall of response.toolCalls) {
            let toolResult;
            if (toolCall.name === 'check_order') {
                toolResult = tools.checkOrder(toolCall.args.order_id);
            } else if (toolCall.name === 'search_policy') {
                toolResult = tools.searchPolicy(toolCall.args.query);
            }

            messages.push({
                role: 'tool',
                content: toolResult,
                toolCallId: toolCall.id
            });
        }

        // 再次调用 LLM
        const newLlmMessages = messages.map(m => ({
            content: m.content,
            role: m.role
        }));
        response = llm.invoke(newLlmMessages);
    }

    // 添加最终响应
    messages.push({ role: 'assistant', content: response.content });

    // 保存检查点
    const checkpointId = generateId();
    const parentCheckpointId = checkpoint ? checkpoint.checkpointId : null;
    const checkpointData = { messages, orderId };
    const metadata = { step: messages.length / 2, timestamp: new Date().toISOString() };

    checkpointDB.saveCheckpoint(threadId, checkpointId, parentCheckpointId, checkpointData, metadata);

    return {
        response: response.content,
        messages: messages
    };
}

// API 路由
app.post('/api/chat', async (req, res) => {
    try {
        const { threadId, message } = req.body;

        if (!threadId || !message) {
            return res.status(400).json({ error: 'threadId 和 message 是必填项' });
        }

        const result = await processMessage(threadId, message);
        res.json(result);
    } catch (error) {
        console.error('聊天处理错误:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

app.post('/api/upload', upload.single('file'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: '没有上传文件' });
        }

        const filePath = req.file.path;
        res.json({
            filePath: filePath,
            filename: req.file.originalname
        });
    } catch (error) {
        console.error('文件上传错误:', error);
        res.status(500).json({ error: '文件上传失败' });
    }
});

app.get('/api/history/:threadId', (req, res) => {
    try {
        const { threadId } = req.params;
        const checkpoint = checkpointDB.getLatestCheckpoint(threadId);

        if (!checkpoint) {
            return res.json({ messages: [] });
        }

        res.json({ messages: checkpoint.checkpointData.messages });
    } catch (error) {
        console.error('获取历史记录错误:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 启动服务器
app.listen(PORT, () => {
    console.log(`服务器运行在 http://localhost:${PORT}`);
});

// 优雅关闭
process.on('SIGINT', () => {
    console.log('正在关闭服务器...');
    checkpointDB.close();
    tools.orderDB.close();
    process.exit(0);
});
