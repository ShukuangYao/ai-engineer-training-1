"""
FAQ数据加载模块
负责FAQ数据的解析、向量索引的构建和管理
"""
import os
import re
from typing import List, Dict, Any

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope

from config import settings


class FAQDataLoader:
    """FAQ数据加载器"""

    def __init__(self):
        """初始化FAQ数据加载器
        
        配置嵌入模型和语言模型
        """
        # 初始化HuggingFace嵌入模型
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",  # 使用BGE-M3模型
            device="cpu",  # 使用CPU模式
            trust_remote_code=True  # 允许加载远程代码
        )
        
        # 初始化DashScope语言模型
        llm = DashScope(
            model_name="deepseek-chat",  # 使用deepseek-chat模型
            api_key=settings.dashscope_api_key  # 从配置中获取API密钥
        )
        
        # 配置LlamaIndex的全局设置
        Settings.embed_model = embed_model  # 设置嵌入模型
        Settings.llm = llm  # 设置语言模型

    def parse_faq_file(self, file_path: str) -> List[Dict[str, Any]]:
        """解析FAQ文件
        
        Args:
            file_path: FAQ文件路径
            
        Returns:
            FAQ条目列表，每个条目包含id、question和answer
        """
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 正则表达式匹配FAQ格式
        qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
        matches = re.findall(qa_pattern, content, re.DOTALL)  # 使用DOTALL模式匹配多行

        # 解析匹配结果
        faq_items = []
        for i, (question, answer) in enumerate(matches):
            faq_items.append({
                'id': i + 1,  # 生成自增ID
                'question': question.strip(),  # 去除首尾空白
                'answer': answer.strip()  # 去除首尾空白
            })

        return faq_items

    def create_documents(self, faq_items: List[Dict[str, Any]]) -> List[Document]:
        """创建文档对象
        
        Args:
            faq_items: FAQ条目列表
            
        Returns:
            文档对象列表
        """
        documents = []
        for item in faq_items:
            # 构建文档内容
            content = f"问题: {item['question']}\n答案: {item['answer']}"
            # 创建文档对象
            doc = Document(
                text=content,  # 文档文本
                metadata={
                    'id': item['id'],  # 文档ID
                    'question': item['question'],  # 问题
                    'answer': item['answer']  # 答案
                }
            )
            documents.append(doc)
        return documents

    def build_vector_index(self, documents: List[Document]) -> VectorStoreIndex:
        """构建向量索引
        
        Args:
            documents: 文档对象列表
            
        Returns:
            向量存储索引
        """
        # 从文档构建向量索引
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        return index

    def save_index(self, index: VectorStoreIndex, index_path: str):
        """保存索引
        
        Args:
            index: 向量存储索引
            index_path: 索引保存路径
        """
        # 确保索引目录存在
        os.makedirs(index_path, exist_ok=True)
        # 持久化索引
        index.storage_context.persist(persist_dir=index_path)

    def load_index(self, index_path: str = None) -> VectorStoreIndex:
        """加载索引
        
        Args:
            index_path: 索引路径，默认为配置中的路径
            
        Returns:
            向量存储索引
        """
        if index_path is None:
            index_path = settings.faiss_index_path
        # 创建存储上下文
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        # 从存储加载索引
        index = load_index_from_storage(storage_context)
        return index

    def initialize_faq_system(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """初始化FAQ系统
        
        Args:
            force_rebuild: 是否强制重建索引
            
        Returns:
            向量存储索引
        """
        index_path = settings.faiss_index_path

        # 检查是否需要重建索引
        if force_rebuild or not os.path.exists(index_path):
            print("构建新的向量索引...")
            # 解析FAQ文件
            faq_items = self.parse_faq_file(settings.faq_file_path)
            print(f"解析到 {len(faq_items)} 个FAQ条目")
            # 创建文档对象
            documents = self.create_documents(faq_items)
            # 构建向量索引
            index = self.build_vector_index(documents)
            # 保存索引
            self.save_index(index, index_path)
            print(f"索引已保存到: {index_path}")
        else:
            print("加载现有索引...")
            # 加载现有索引
            index = self.load_index(index_path)

        return index