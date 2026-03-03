import onnxruntime
from qanything_kernel.dependent_server.rerank_server.rerank_backend import RerankBackend
from qanything_kernel.configs.model_config import LOCAL_RERANK_MODEL_PATH
from qanything_kernel.utils.custom_log import debug_logger
import numpy as np


def sigmoid(x):
    """
    Sigmoid 激活函数，将 logits 转换为 0-1 之间的概率分数
    
    增强版 sigmoid，包含以下优化：
    1. 确保输入为 float32 类型
    2. 应用标准 sigmoid 函数
    3. 对分数进行线性变换，增强区分度
    4. 裁剪分数到 [0, 1] 范围
    
    Args:
        x: 模型输出的 logits
        
    Returns:
        np.array: 转换后的概率分数
    """
    # 确保输入为 float32 类型
    x = x.astype('float32')
    # 应用标准 sigmoid 函数
    scores = 1/(1+np.exp(-x))
    # 线性变换增强区分度：将中间值 (0.5) 附近的分数拉开
    scores = np.clip(1.5*(scores-0.5)+0.5, 0, 1)
    return scores


class RerankOnnxBackend(RerankBackend):
    """
    ONNX 重排序模型后端
    
    使用 ONNX Runtime 执行重排序模型推理，支持 CPU 和 GPU 加速
    """
    def __init__(self, use_cpu: bool = False):
        """
        初始化 ONNX 重排序后端
        
        Args:
            use_cpu: 是否使用 CPU 进行推理
        """
        super().__init__(use_cpu)
        # 设置返回张量类型为 numpy 数组
        self.return_tensors = "np"
        
        # 创建 ONNX Runtime 会话设置
        sess_options = onnxruntime.SessionOptions()
        # 启用所有图形优化
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 线程数设置为 0，使用系统默认值
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        
        # 根据配置选择执行提供者
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            # 优先使用 CUDA，如果不可用则回退到 CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 创建 ONNX 推理会话
        self.session = onnxruntime.InferenceSession(
            LOCAL_RERANK_MODEL_PATH, 
            sess_options, 
            providers=providers
        )

    def inference(self, batch):
        """
        执行模型推理，计算相关性分数
        
        Args:
            batch: 批量输入数据，包含 input_ids、attention_mask 等
            
        Returns:
            List[float]: 每个输入的相关性分数列表
        """
        # 准备输入数据，映射到模型的输入名称
        inputs = {
            # 第一个输入：token ID 序列
            self.session.get_inputs()[0].name: batch['input_ids'],
            # 第二个输入：注意力掩码
            self.session.get_inputs()[1].name: batch['attention_mask']
        }

        # 如果存在 token_type_ids，则添加到输入中
        if 'token_type_ids' in batch:
            inputs[self.session.get_inputs()[2].name] = batch['token_type_ids']

        # 执行推理，输出为 logits（原始模型输出，未经过激活函数）
        result = self.session.run(None, inputs)  # None 表示获取所有输出
        # debug_logger.info(f"rerank result: {result}")

        # 应用 sigmoid 函数，将 logits 转换为 0-1 之间的概率分数
        sigmoid_scores = sigmoid(np.array(result[0]))

        # 将分数重塑为一维数组并转换为列表返回
        return sigmoid_scores.reshape(-1).tolist()
