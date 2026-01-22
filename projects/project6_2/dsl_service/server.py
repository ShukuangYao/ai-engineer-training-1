import os
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional

from src.workflow_manager import WorkflowManager
from src.logger import ExecutionLogger

# 初始化组件
app = FastAPI(title="LangGraph DSL Engine Service")
manager = WorkflowManager()
logger = ExecutionLogger()

# 定义请求模型
class RunRequest(BaseModel):
    dsl_path: str
    input: Dict[str, Any] = {}

class RunResponse(BaseModel):
    trace_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

# 定义任务执行函数 (后台运行)
def execute_workflow(trace_id: str, dsl_path: str, input_data: Dict[str, Any]):
    try:
        # 获取工作流 (自动处理缓存和热更新)
        workflow = manager.get_workflow(dsl_path)
        
        # 准备初始状态
        initial_state = {
            "context": input_data,
            "logs": [],
            "__router__": ""
        }
        
        # 记录开始事件
        logger.log_event(trace_id, "system", "start", {"dsl": dsl_path})
        
        # 执行工作流
        # 注意: 这里是同步调用，如果需要完全非阻塞，应该在 workflow 内部支持 async 或者放到线程池
        # LangGraph 的 invoke 通常是同步的，但在 FastAPI BackgroundTasks 中运行是安全的，不会阻塞主线程响应
        final_state = workflow.invoke(initial_state)
        
        # 记录详细日志
        for log_entry in final_state.get("logs", []):
             logger.log_event(trace_id, "step", "log", log_entry)
        
        # 记录结束
        logger.end_trace(trace_id, "SUCCESS", final_state.get("context", {}))
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.log_event(trace_id, "system", "error", {"error": error_msg, "stack": stack_trace})
        logger.end_trace(trace_id, "ERROR", {"error": error_msg})

@app.post("/run", response_model=RunResponse)
async def run_workflow(request: RunRequest, background_tasks: BackgroundTasks):
    """
    提交一个工作流执行请求。
    """
    # 验证文件是否存在
    if not os.path.exists(request.dsl_path):
        raise HTTPException(status_code=404, detail=f"DSL 文件未找到: {request.dsl_path}")
    
    # 开始追踪
    trace_id = logger.start_trace(request.dsl_path, request.input)
    
    # 添加到后台任务
    background_tasks.add_task(execute_workflow, trace_id, request.dsl_path, request.input)
    
    return RunResponse(
        trace_id=trace_id,
        status="ACCEPTED",
        message="工作流已提交后台执行"
    )

@app.get("/logs/{trace_id}")
async def get_logs(trace_id: str):
    """
    获取指定 trace_id 的执行日志。
    """
    trace = logger.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace ID 未找到")
    return trace

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # 确保 dsl_files 目录存在
    os.makedirs("dsl_files", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8001)
