import logging
import numpy as np
import os
import dashscope
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)  # 初始化日志记录器

# 设置通义千问API密钥，优先从环境变量获取，否则使用默认值
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")

class QwenEmbedding:
    """通义千问文本嵌入服务

    用于将文本编码为向量表示，支持通义千问API和本地模型降级
    当API调用失败时，会自动切换到本地的SentenceTransformer模型
    """

    def __init__(self, model_name="text-embedding-v3"):
        """初始化嵌入服务

        Args:
            model_name: 嵌入模型名称，默认使用text-embedding-v3
        """
        self.model_name = model_name  # 存储模型名称

    def encode(self, texts):
        """编码文本为向量

        Args:
            texts: 要编码的文本，可以是单个字符串或字符串列表

        Returns:
            文本的向量表示，如果输入是单个字符串则返回单个向量，否则返回向量列表
        """
        # 确保输入是列表形式，方便统一处理
        if isinstance(texts, str):
            texts = [texts]

        try:
            from dashscope import TextEmbedding  # 动态导入，减少初始化时的依赖

            # 调用通义千问文本嵌入API
            response = TextEmbedding.call(
                model=self.model_name,  # 指定使用的模型
                input=texts  # 输入文本列表
            )

            if response.status_code == 200:  # API调用成功
                embeddings = []
                # 处理API返回的嵌入结果，将每个嵌入转换为numpy数组
                for output in response.output['embeddings']:
                    embeddings.append(np.array(output['embedding']))

                # 根据输入格式返回相应格式的结果
                # 如果输入是单个字符串（转换后列表长度为1），返回单个向量
                # 否则返回向量列表
                return embeddings[0] if len(embeddings) == 1 else embeddings
            else:
                # API调用失败，记录错误并降级到本地模型
                logger.error(f"通义千问embedding调用失败: {response}")
                # 初始化本地备用模型
                fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
                # 使用本地模型编码文本
                return fallback_model.encode(texts)

        except Exception as e:
            # 发生异常，记录错误并降级到本地模型
            logger.error(f"通义千问embedding异常: {e}")
            # 初始化本地备用模型
            fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
            # 使用本地模型编码文本
            return fallback_model.encode(texts)