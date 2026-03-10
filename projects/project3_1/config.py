"""
配置文件
定义系统的全局配置参数
"""
import os


# DashScope配置（仅用于LLM，不再用于嵌入模型）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # 从环境变量获取API密钥，默认为空
DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v3"  # 已弃用，保留用于兼容性

# 向量配置
FAISS_INDEX_PATH = "./data/faiss_index"  # FAISS向量索引存储路径
VECTOR_DIMENSION = 1024  # BGE-M3模型的向量维度为1024

# FAQ文件
FAQ_FILE_PATH = "./FAQ.txt"  # FAQ文件存储路径

# 检索配置
TOP_K = 3  # 检索时返回的最相关结果数量


class Settings:
    """系统配置类"""
    dashscope_api_key = DASHSCOPE_API_KEY  # DashScope API密钥
    dashscope_embedding_model = DASHSCOPE_EMBEDDING_MODEL  # 嵌入模型名称（已弃用）
    faiss_index_path = FAISS_INDEX_PATH  # FAISS索引路径
    vector_dimension = VECTOR_DIMENSION  # 向量维度
    faq_file_path = FAQ_FILE_PATH  # FAQ文件路径
    top_k = TOP_K  # 检索结果数量


settings = Settings()  # 创建配置实例