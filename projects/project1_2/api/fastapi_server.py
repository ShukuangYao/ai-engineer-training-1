"""
FastAPI模拟内部系统服务
提供订单状态和物流信息的REST API接口

功能说明：
    - 提供订单状态查询接口（/api/orders/{order_id}）
    - 提供物流信息查询接口（/api/logistics/{order_id}）
    - 模拟内部系统的API服务，用于测试智能体系统
    - 支持CORS跨域请求
    - 包含模拟数据和错误处理
"""
import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
import asyncio
from pydantic import BaseModel
from rich.console import Console
from core.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)
console = Console()
# 添加项目根目录到 Python 路径，确保能够正确导入模块
# 这样无论从哪个目录运行，都能正确找到 api 模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 数据模型
class OrderResponse(BaseModel):
    order_id: str
    status: str
    customer_name: str
    total_amount: float
    items: List[str]
    shipping_address: str
    created_at: str
    updated_at: str
    estimated_delivery: Optional[str] = None

class LogisticsResponse(BaseModel):
    order_id: str
    tracking_number: str
    status: str
    current_location: str
    carrier: str
    estimated_delivery: Optional[str] = None
    tracking_history: List[Dict[str, str]]

# 创建FastAPI应用
app = FastAPI(
    title="客服系统模拟API",
    description="模拟内部订单和物流系统的API接口",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟数据
MOCK_ORDERS = {
    "ORD001": {
        "order_id": "ORD001",
        "status": "processing",
        "customer_name": "张三",
        "total_amount": 299.99,
        "items": ["iPhone手机壳", "无线充电器"],
        "shipping_address": "北京市朝阳区xxx街道xxx号",
        "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
        "updated_at": (datetime.now() - timedelta(hours=6)).isoformat(),
        "estimated_delivery": (datetime.now() + timedelta(days=1)).isoformat()
    },
    "ORD002": {
        "order_id": "ORD002",
        "status": "shipped",
        "customer_name": "李四",
        "total_amount": 1299.00,
        "items": ["蓝牙耳机", "充电线"],
        "shipping_address": "上海市浦东新区xxx路xxx号",
        "created_at": (datetime.now() - timedelta(days=5)).isoformat(),
        "updated_at": (datetime.now() - timedelta(days=1)).isoformat(),
        "estimated_delivery": (datetime.now() + timedelta(hours=4)).isoformat()
    },
    "ORD003": {
        "order_id": "ORD003",
        "status": "confirmed",
        "customer_name": "王五",
        "total_amount": 599.50,
        "items": ["智能手表"],
        "shipping_address": "广州市天河区xxx大道xxx号",
        "created_at": (datetime.now() - timedelta(hours=12)).isoformat(),
        "updated_at": (datetime.now() - timedelta(hours=10)).isoformat(),
        "estimated_delivery": None
    }
}

MOCK_LOGISTICS = {
    "ORD001": {
        "order_id": "ORD001",
        "tracking_number": "SF1234567890",
        "status": "in_transit",
        "current_location": "北京分拣中心",
        "carrier": "顺丰速运",
        "estimated_delivery": (datetime.now() + timedelta(days=1)).isoformat(),
        "tracking_history": [
            {"time": "2024-01-15 10:00", "location": "北京仓库", "status": "已发货"},
            {"time": "2024-01-15 14:30", "location": "北京分拣中心", "status": "运输中"}
        ]
    },
    "ORD002": {
        "order_id": "ORD002",
        "tracking_number": "YTO9876543210",
        "status": "out_for_delivery",
        "current_location": "上海配送站",
        "carrier": "圆通速递",
        "estimated_delivery": (datetime.now() + timedelta(hours=4)).isoformat(),
        "tracking_history": [
            {"time": "2024-01-14 09:00", "location": "上海仓库", "status": "已发货"},
            {"time": "2024-01-14 15:20", "location": "上海分拣中心", "status": "运输中"},
            {"time": "2024-01-15 08:00", "location": "上海配送站", "status": "派送中"}
        ]
    },
    "ORD003": {
        "order_id": "ORD003",
        "tracking_number": "",
        "status": "not_shipped",
        "current_location": "广州仓库",
        "carrier": "待分配",
        "estimated_delivery": None,
        "tracking_history": []
    }
}

@app.get("/")
async def root():
    """根路径"""
    return {"message": "客服系统模拟API服务正在运行", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/orders/{order_id}", response_model=OrderResponse)
async def get_order_status(order_id: str):
    """
    获取订单状态信息
    Agent A 调用此接口查询订单状态
    """
    # 模拟网络延迟
    await asyncio.sleep(random.uniform(0.1, 0.5))

    # 模拟偶发性网络错误（10%概率）
    if random.random() < 0.1:
        raise HTTPException(status_code=500, detail="内部服务暂时不可用")

    if order_id not in MOCK_ORDERS:
        raise HTTPException(status_code=404, detail=f"订单 {order_id} 不存在")

    order_data = MOCK_ORDERS[order_id]
    return OrderResponse(**order_data)

@app.get("/api/logistics/{order_id}", response_model=LogisticsResponse)
async def get_logistics_info(order_id: str):
    """
    获取物流信息
    Agent B 调用此接口查询物流状态
    """
    # 模拟网络延迟
    await asyncio.sleep(random.uniform(0.1, 0.8))

    # 模拟偶发性网络错误（10%概率）
    if random.random() < 0.1:
        raise HTTPException(status_code=500, detail="物流服务暂时不可用")

    if order_id not in MOCK_LOGISTICS:
        raise HTTPException(status_code=404, detail=f"订单 {order_id} 的物流信息不存在")

    logistics_data = MOCK_LOGISTICS[order_id]
    logger.info(f"物流信息查询成功: {logistics_data}")
    return LogisticsResponse(**logistics_data)

@app.get("/api/orders")
async def list_orders():
    """获取所有订单列表"""
    return {"orders": list(MOCK_ORDERS.keys()), "total": len(MOCK_ORDERS)}

# 启动服务器的函数
def start_server(host: str = "127.0.0.1", port: int = 8000):
    """
    启动FastAPI服务器

    功能说明：
        启动 FastAPI 模拟服务器，提供订单和物流查询接口。
        支持从项目根目录或 api 目录直接运行。

    参数:
        host (str): 服务器地址（默认：127.0.0.1）
        port (int): 服务器端口（默认：8000）

    使用方式:
        方式1: 从项目根目录运行
            python -m api.fastapi_server

        方式2: 从 api 目录运行
            python fastapi_server.py

        方式3: 从项目根目录运行
            python api/fastapi_server.py
    """
    print(f"🚀 启动FastAPI模拟服务器...")
    print(f"📍 服务地址: http://{host}:{port}")
    print(f"📖 API文档: http://{host}:{port}/docs")
    print(f"📁 项目根目录: {project_root}")

    # 确保从项目根目录运行，这样 uvicorn 才能正确导入 api 模块
    original_cwd = os.getcwd()
    try:
        # 切换到项目根目录
        os.chdir(project_root)

        # 使用字符串路径方式启动（需要从项目根目录导入）
        uvicorn.run(
            "api.fastapi_server:app",
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)

if __name__ == "__main__":
    """
    直接运行此文件时启动服务器

    注意：
        - 可以从项目根目录运行：python api/fastapi_server.py
        - 也可以从 api 目录运行：python fastapi_server.py
        - 推荐使用：python -m api.fastapi_server（从项目根目录）
    """
    start_server()