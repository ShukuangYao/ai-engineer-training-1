"""
FAQ知识库管理系统 Web API
基于FastAPI实现的RESTful API接口
"""
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from kb_manager import KnowledgeBaseManager
from data_loader import FAQDataLoader
from config import settings


# Pydantic模型定义
class ChatRequest(BaseModel):
    """聊天请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="会话ID")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str = Field(..., description="回答内容")
    session_id: str = Field(..., description="会话ID")
    sources: List[Dict[str, Any]] = Field(default=[], description="相关FAQ来源")
    timestamp: str = Field(..., description="响应时间戳")


class FAQItem(BaseModel):
    """FAQ条目模型"""
    question: str = Field(..., description="问题", min_length=1, max_length=500)
    answer: str = Field(..., description="答案", min_length=1, max_length=2000)


class FAQResponse(BaseModel):
    """FAQ响应模型"""
    id: int = Field(..., description="FAQ ID")
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="答案")


class FAQUpdateRequest(BaseModel):
    """FAQ更新请求模型"""
    question: Optional[str] = Field(None, description="问题", max_length=500)
    answer: Optional[str] = Field(None, description="答案", max_length=2000)


class BatchImportRequest(BaseModel):
    """批量导入请求模型"""
    faqs: List[FAQItem] = Field(..., description="FAQ列表")
    merge_strategy: str = Field("append", description="合并策略: append, replace")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="系统状态")
    timestamp: str = Field(..., description="检查时间")
    components: Dict[str, str] = Field(..., description="组件状态")


# 全局变量
app = FastAPI(
    title="FAQ知识库管理系统",
    description="基于向量检索的FAQ知识库管理和问答系统",
    version="1.0.0"
)

# 初始化组件
kb_manager = KnowledgeBaseManager()  # 知识库管理器
data_loader = FAQDataLoader()  # 数据加载器

# 会话存储（生产环境建议使用Redis等持久化存储）
chat_sessions: Dict[str, List[Dict[str, Any]]] = {}


# 工具函数
def get_or_create_session_id(session_id: Optional[str] = None) -> str:
    """获取或创建会话ID

    Args:
        session_id: 会话ID

    Returns:
        会话ID
    """
    if session_id and session_id in chat_sessions:
        return session_id

    new_session_id = str(uuid.uuid4())  # 生成新的会话ID
    chat_sessions[new_session_id] = []  # 初始化会话历史
    return new_session_id


def add_to_chat_history(session_id: str, question: str, answer: str, sources: List[Dict[str, Any]]):
    """添加到聊天历史

    Args:
        session_id: 会话ID
        question: 用户问题
        answer: 回答内容
        sources: 相关来源
    """
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_sessions[session_id].append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "timestamp": datetime.now().isoformat()
    })


# API路由实现

# 1. 问答接口
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送问题获取答案

    Args:
        request: 聊天请求

    Returns:
        聊天响应
    """
    try:
        # 获取或创建会话ID
        session_id = get_or_create_session_id(request.session_id)

        # 检查索引是否存在
        if not os.path.exists(settings.faiss_index_path):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="向量索引不存在，请先重建索引"
            )

        # 加载索引并查询
        index = data_loader.load_index()
        if index is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="索引加载失败"
            )

        # 执行查询
        query_engine = index.as_query_engine(similarity_top_k=settings.top_k)
        response = query_engine.query(request.question)

        # 提取相关来源
        sources = []
        for node in response.source_nodes:
            metadata = node.node.metadata
            sources.append({
                "question": metadata.get('question', 'N/A'),
                "answer": metadata.get('answer', 'N/A'),
                "similarity_score": float(node.score) if hasattr(node, 'score') else 0.0
            })

        answer = str(response)
        timestamp = datetime.now().isoformat()

        # 添加到聊天历史
        add_to_chat_history(session_id, request.question, answer, sources)

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            sources=sources,
            timestamp=timestamp
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询处理失败: {str(e)}"
        )


@app.get("/api/v1/chat/history")
async def get_chat_history(session_id: str):
    """获取对话历史

    Args:
        session_id: 会话ID

    Returns:
        对话历史
    """
    if session_id not in chat_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )

    return {
        "session_id": session_id,
        "history": chat_sessions[session_id],
        "total_count": len(chat_sessions[session_id])
    }


# 2. 知识库管理接口
@app.post("/api/v1/knowledge/faq", response_model=FAQResponse)
async def add_faq(faq: FAQItem):
    """添加FAQ条目

    Args:
        faq: FAQ条目

    Returns:
        添加的FAQ条目
    """
    try:
        new_faq = kb_manager.add_faq(
            question=faq.question,
            answer=faq.answer,
            auto_rebuild=True
        )

        return FAQResponse(
            id=new_faq['id'],
            question=new_faq['question'],
            answer=new_faq['answer']
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加FAQ失败: {str(e)}"
        )


@app.put("/api/v1/knowledge/faq/{faq_id}", response_model=FAQResponse)
async def update_faq(faq_id: int, request: FAQUpdateRequest):
    """更新FAQ条目

    Args:
        faq_id: FAQ ID
        request: 更新请求

    Returns:
        更新后的FAQ条目
    """
    try:
        success = kb_manager.update_faq(
            faq_id=faq_id,
            question=request.question,
            answer=request.answer,
            auto_rebuild=True
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAQ条目不存在"
            )

        # 获取更新后的FAQ
        updated_faq = kb_manager.get_faq_by_id(faq_id)
        if not updated_faq:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAQ条目不存在"
            )

        return FAQResponse(
            id=updated_faq['id'],
            question=updated_faq['question'],
            answer=updated_faq['answer']
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新FAQ失败: {str(e)}"
        )


@app.delete("/api/v1/knowledge/faq/{faq_id}")
async def delete_faq(faq_id: int):
    """删除FAQ条目

    Args:
        faq_id: FAQ ID

    Returns:
        删除结果
    """
    try:
        success = kb_manager.delete_faq(faq_id, auto_rebuild=True)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAQ条目不存在"
            )

        return {"message": f"FAQ条目 {faq_id} 已成功删除"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除FAQ失败: {str(e)}"
        )


@app.get("/api/v1/knowledge/faq", response_model=List[FAQResponse])
async def get_faqs(keyword: Optional[str] = None, limit: int = 100, offset: int = 0):
    """查询FAQ列表

    Args:
        keyword: 搜索关键词
        limit: 限制数量
        offset: 偏移量

    Returns:
        FAQ列表
    """
    try:
        if keyword:
            faqs = kb_manager.search_faqs(keyword)
        else:
            faqs = kb_manager.get_all_faqs()

        # 分页处理
        total_count = len(faqs)
        faqs = faqs[offset:offset + limit]

        response_faqs = [
            FAQResponse(
                id=faq['id'],
                question=faq['question'],
                answer=faq['answer']
            )
            for faq in faqs
        ]

        return response_faqs

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询FAQ失败: {str(e)}"
        )


@app.post("/api/v1/knowledge/batch")
async def batch_import_faqs(request: BatchImportRequest):
    """批量导入FAQ

    Args:
        request: 批量导入请求

    Returns:
        导入结果
    """
    try:
        # 转换为知识库管理器需要的格式
        new_faqs = [
            {"question": faq.question, "answer": faq.answer}
            for faq in request.faqs
        ]

        success = kb_manager.update_knowledge_base(
            new_faqs=new_faqs,
            merge_strategy=request.merge_strategy
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="批量导入失败"
            )

        return {
            "message": f"成功导入 {len(new_faqs)} 条FAQ",
            "merge_strategy": request.merge_strategy,
            "imported_count": len(new_faqs)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量导入失败: {str(e)}"
        )


# 3. 系统管理接口
@app.get("/api/v1/system/health", response_model=HealthResponse)
async def health_check():
    """系统健康检查

    Returns:
        健康检查结果
    """
    try:
        components = {}

        # 检查FAQ文件
        if os.path.exists(settings.faq_file_path):
            components["faq_file"] = "healthy"
        else:
            components["faq_file"] = "missing"

        # 检查向量索引
        if os.path.exists(settings.faiss_index_path):
            components["vector_index"] = "healthy"
        else:
            components["vector_index"] = "missing"

        # 检查API密钥
        if settings.dashscope_api_key:
            components["api_key"] = "configured"
        else:
            components["api_key"] = "missing"

        # 检查数据目录
        data_dir = "./data"
        if os.path.exists(data_dir):
            components["data_directory"] = "healthy"
        else:
            components["data_directory"] = "missing"

        # 确定整体状态
        overall_status = "healthy" if all(
            status in ["healthy", "configured"] for status in components.values()
        ) else "degraded"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            components=components
        )

    except Exception as e:
        return HealthResponse(
            status="error",
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)}
        )


@app.post("/api/v1/system/rebuild-index")
async def rebuild_index(force: bool = True):
    """重建向量索引

    Args:
        force: 是否强制重建

    Returns:
        重建结果
    """
    try:
        success = kb_manager.rebuild_index(force=force)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="索引重建失败"
            )

        return {
            "message": "向量索引重建成功",
            "timestamp": datetime.now().isoformat(),
            "force": force
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"索引重建失败: {str(e)}"
        )


# 异常处理器
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404异常处理"""
    return JSONResponse(
        status_code=404,
        content={"detail": "资源不存在"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """500异常处理"""
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"}
    )


# 启动配置
if __name__ == "__main__":
    """启动Web API服务"""
    uvicorn.run(
        "webapi:app",
        host="0.0.0.0",  # 监听所有网络接口
        port=8000,  # 端口号
        reload=True,  # 开发模式自动重载
        log_level="info"  # 日志级别
    )