import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [threadId, setThreadId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // 初始化时生成 threadId
  useEffect(() => {
    const newThreadId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    setThreadId(newThreadId);
    loadHistory(newThreadId);
  }, []);

  // 滚动到底部
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // 加载历史记录
  const loadHistory = async (tid) => {
    try {
      const response = await axios.get(`/api/history/${tid}`);
      if (response.data.messages && response.data.messages.length > 0) {
        setMessages(response.data.messages);
      } else {
        // 添加欢迎消息
        setMessages([{
          role: 'assistant',
          content: '您好！我是订单查询客服，有什么可以帮助您的吗？'
        }]);
      }
    } catch (error) {
      console.error('加载历史记录失败:', error);
      // 添加欢迎消息
      setMessages([{
        role: 'assistant',
        content: '您好！我是订单查询客服，有什么可以帮助您的吗？'
      }]);
    }
  };

  // 发送消息
  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { role: 'user', content: inputValue.trim() };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await axios.post('/api/chat', {
        threadId: threadId,
        message: userMessage.content
      });

      setMessages(response.data.messages);
    } catch (error) {
      console.error('发送消息失败:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '抱歉，服务器出现错误，请稍后再试。'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // 处理文件上传
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || isLoading) return;

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      // 发送文件路径作为消息
      const userMessage = { role: 'user', content: response.data.filePath };
      setMessages(prev => [...prev, {
        role: 'user',
        content: `[上传文件: ${response.data.filename}]`
      }]);

      // 调用聊天 API
      const chatResponse = await axios.post('/api/chat', {
        threadId: threadId,
        message: userMessage.content
      });

      setMessages(chatResponse.data.messages);
    } catch (error) {
      console.error('文件上传失败:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '抱歉，文件上传失败，请稍后再试。'
      }]);
    } finally {
      setIsLoading(false);
      event.target.value = '';
    }
  };

  // 处理回车键
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <h1>📦 订单查询客服</h1>
          <p className="thread-id">会话ID: {threadId}</p>
        </div>

        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-content">
                <div className="message-text">{msg.content}</div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input">
          <div className="input-actions">
            <label className="upload-btn">
              <input
                type="file"
                accept=".wav,.mp3,.jpg,.jpeg,.png"
                onChange={handleFileUpload}
                disabled={isLoading}
              />
              📎 上传文件
            </label>
          </div>
          <div className="input-wrapper">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="输入您的消息..."
              disabled={isLoading}
              rows={1}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="send-btn"
            >
              发送
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
