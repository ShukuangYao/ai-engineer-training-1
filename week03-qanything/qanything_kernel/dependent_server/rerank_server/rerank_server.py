import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)
print(root_dir)

from sanic import Sanic
from sanic.response import json
from qanything_kernel.dependent_server.rerank_server.rerank_async_backend import RerankAsyncBackend
from qanything_kernel.dependent_server.rerank_server.rerank_onnx_backend import RerankOnnxBackend
from qanything_kernel.configs.model_config import LOCAL_RERANK_MODEL_PATH, LOCAL_RERANK_THREADS
from qanything_kernel.utils.general_utils import get_time_async
import argparse

# 接收外部参数
parser = argparse.ArgumentParser()
# 是否使用 GPU 加速
parser.add_argument('--use_gpu', action="store_true", help='use gpu or not')
# 服务 worker 数量
parser.add_argument('--workers', type=int, default=1, help='workers')

# 解析参数
args = parser.parse_args()
print("args:", args)

# 创建 Sanic 应用实例
app = Sanic("rerank_server")


@get_time_async
@app.route("/rerank", methods=["POST"])
async def rerank(request):
    """
    处理重排序请求的 HTTP 端点

    接收包含查询和文档段落的 POST 请求，返回相关性分数列表

    Args:
        request: Sanic 请求对象，包含 JSON 格式的请求数据

    Returns:
        JSON: 相关性分数列表，对应输入文档的顺序
    """
    # 解析请求数据
    data = request.json
    query = data.get('query')  # 用户查询
    passages = data.get('passages')  # 文档段落列表

    # 获取 ONNX 后端实例
    onnx_backend: RerankOnnxBackend = request.app.ctx.onnx_backend
    # 注释掉的代码：使用异步后端
    # onnx_backend: RerankAsyncBackend = request.app.ctx.onnx_backend

    # 注释掉的代码：使用异步重排序
    # result_data = await onnx_backend.get_rerank_async(query, passages)

    # 执行重排序，获取相关性分数
    result_data = onnx_backend.get_rerank(query, passages)

    # 调试信息（已注释）
    # print("local rerank query:", query, flush=True)
    # print("local rerank passages number:", len(passages), flush=True)

    # 返回 JSON 格式的分数列表
    return json(result_data)


@app.listener('before_server_start')
async def setup_onnx_backend(app, loop):
    """
    服务启动前的初始化函数

    创建并初始化 ONNX 后端实例，存储到应用上下文中

    Args:
        app: Sanic 应用实例
        loop: 事件循环
    """
    # 注释掉的代码：使用异步后端
    # app.ctx.onnx_backend = RerankAsyncBackend(model_path=LOCAL_RERANK_MODEL_PATH, use_cpu=not args.use_gpu,
    #                                           num_threads=LOCAL_RERANK_THREADS)

    # 创建 ONNX 后端实例，传入是否使用 CPU 的参数
    # 注意：use_cpu 参数取反，因为 RerankOnnxBackend 的 use_cpu=True 表示使用 CPU
    app.ctx.onnx_backend = RerankOnnxBackend(use_cpu=not args.use_gpu)


if __name__ == "__main__":
    """
    服务入口点

    启动 Sanic 服务器，监听指定的主机和端口
    """
    # 启动服务器
    # host="0.0.0.0"：监听所有网络接口
    # port=8001：使用 8001 端口
    # workers=args.workers：使用指定数量的 worker 进程
    app.run(host="0.0.0.0", port=8001, workers=args.workers)
