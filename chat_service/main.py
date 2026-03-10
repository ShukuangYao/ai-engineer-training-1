from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import asyncio
from chat_chain import ChatChain
from session_manager import SessionManager
from contextlib import asynccontextmanager

# 全局实例
chat_chain = None
session_manager = SessionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    global chat_chain
    chat_chain = ChatChain()
    await chat_chain.initialize()
    yield
    # 关闭时
    pass

app = FastAPI(
    title="智能对话服务",
    description="基于 LangChain 0.3 的对话 API",
    version="1.0.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """对话接口"""
    try:
        # 获取会话历史
        history = session_manager.get_history(request.session_id)

        # 调用对话链
        reply = await chat_chain.process_message(
            message=request.message,
            history=history
        )

        # 更新会话历史
        session_manager.add_message(
            session_id=request.session_id,
            user_message=request.message,
            bot_reply=reply
        )

        return ChatResponse(
            reply=reply,
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "langchain_version": "0.3.x"}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清除会话历史"""
    session_manager.clear_session(session_id)
    return {"message": f"会话 {session_id} 已清除"}

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """获取会话历史"""
    history = session_manager.get_history(session_id)
    return {"session_id": session_id, "history": history}

if __name__ == "__main__":
    # 使用字符串导入应用以支持 reload 模式
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)