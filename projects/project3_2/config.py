"""
混合RAG系统配置文件
包含所有系统配置参数和常量定义
用于统一管理系统的所有配置项，便于后续维护和调整
"""

import os
import logging

# 日志配置
LOG_LEVEL = logging.INFO  # 日志级别，INFO级别记录一般信息
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 日志格式，包含时间、模块名、级别和消息
LOG_FILE = 'hybrid_rag.log'  # 日志文件路径，系统运行时生成的日志文件

# DashScope API配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # DashScope API密钥，从环境变量获取，为空时使用本地模型

# 系统默认配置
DEFAULT_CONFIDENCE_THRESHOLD = 0.3  # 默认置信度阈值，用于过滤低置信度结果
MAX_RETRIEVAL_RESULTS = 10  # 最大检索结果数量，控制检索返回的结果上限
ERROR_PROPAGATION_THRESHOLD = 0.3  # 错误传播阈值，低于此值的结果将被过滤

# 嵌入模型配置
EMBEDDING_MODEL_NAME = "text-embedding-v3"  # 嵌入模型名称，使用通义千问的文本嵌入模型
FALLBACK_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # 备用嵌入模型，当API调用失败时使用
EMBEDDING_DIMENSION = 1536  # 嵌入向量维度，对应text-embedding-v3模型的输出维度

# Neo4j配置
NEO4J_URI = "bolt://localhost:7687"  # Neo4j数据库URI，用于连接图数据库
NEO4J_USERNAME = "neo4j"  # Neo4j用户名，默认用户
NEO4J_PASSWORD = "password"  # Neo4j密码，默认密码

# LLM配置
LLM_MODEL_NAME = "qwen-turbo"  # 主LLM模型名称，用于生成答案
LLM_JSON_MODEL = "qwen-turbo"  # JSON格式输出的LLM模型，用于结构化输出
LLM_TEXT_MODEL = "qwen-turbo"  # 文本输出的LLM模型，用于生成自然语言文本

# 检索配置
RETRIEVAL_TOP_K = 5  # 检索返回的top-k结果数，控制每次检索返回的结果数量
RETRIEVAL_VECTOR_WEIGHT = 0.5  # 向量检索权重，在联合评分中占比50%
RETRIEVAL_KEYWORD_WEIGHT = 0.3  # 关键词检索权重，在联合评分中占比30%
RETRIEVAL_GRAPH_WEIGHT = 0.2  # 图谱检索权重，在联合评分中占比20%

# 关键词配置
KEYWORD_THRESHOLD = 0.1  # 关键词提取阈值，低于此值的关键词将被过滤
KEYWORD_MAX_KEYWORDS = 10  # 最大关键词数量，控制每次提取的关键词上限
KEYWORD_MIN_KEYWORD_LENGTH = 2  # 关键词最小长度，过滤过短的关键词

# 文档配置
DOCUMENT_MIN_LENGTH = 10  # 文档最小长度，过滤过短的文档
DOCUMENT_MAX_LENGTH = 10000  # 文档最大长度，避免处理过长的文档
DOCUMENT_ENCODING = 'utf-8'  # 文档编码，确保正确读取文档内容

# 图数据库配置
GRAPH_MAX_DEPTH = 3  # 图谱查询最大深度，控制图谱遍历的深度
GRAPH_MAX_HOPS = 3  # 图谱查询最大跳数，控制多跳推理的步数
GRAPH_MIN_CONFIDENCE = 0.2  # 图谱关系最小置信度，过滤低置信度的关系
GRAPH_MAX_RELATIONSHIPS = 50  # 最大关系数量，控制图谱返回的关系上限

# 答案生成配置
ANSWER_MAX_TOKENS = 1000  # 答案最大token数，控制生成答案的长度
ANSWER_TEMPERATURE = 0.7  # 生成温度参数，控制答案的随机性，值越高随机性越大
ANSWER_TOP_P = 0.9  # 生成top-p参数，控制采样范围，值越高生成的内容越丰富
ANSWER_MAX_CONTEXT_DOCS = 5  # 答案生成时使用的最大文档数，控制上下文窗口大小
ANSWER_MAX_GRAPH_RELATIONS = 10  # 答案生成时使用的最大图谱关系数，控制图谱信息的使用量