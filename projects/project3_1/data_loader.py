"""
FAQ数据加载模块
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
        """初始化"""
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            device="cpu",  # 使用CPU模式
            trust_remote_code=True
        )
        llm = DashScope(
            model_name="deepseek-chat",
            api_key=settings.dashscope_api_key
        )
        Settings.embed_model = embed_model
        Settings.llm = llm

    def parse_faq_file(self, file_path: str) -> List[Dict[str, Any]]:
        """解析FAQ文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
        matches = re.findall(qa_pattern, content, re.DOTALL)

        faq_items = []
        for i, (question, answer) in enumerate(matches):
            faq_items.append({
                'id': i + 1,
                'question': question.strip(),
                'answer': answer.strip()
            })

        return faq_items

    def create_documents(self, faq_items: List[Dict[str, Any]]) -> List[Document]:
        """创建文档对象"""
        documents = []
        for item in faq_items:
            content = f"问题: {item['question']}\n答案: {item['answer']}"
            doc = Document(
                text=content,
                metadata={
                    'id': item['id'],
                    'question': item['question'],
                    'answer': item['answer']
                }
            )
            documents.append(doc)
        return documents

    def build_vector_index(self, documents: List[Document]) -> VectorStoreIndex:
        """构建向量索引"""
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        return index

    def save_index(self, index: VectorStoreIndex, index_path: str):
        """保存索引"""
        os.makedirs(index_path, exist_ok=True)
        index.storage_context.persist(persist_dir=index_path)

    def load_index(self, index_path: str = None) -> VectorStoreIndex:
        """加载索引"""
        if index_path is None:
            index_path = settings.faiss_index_path
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        return index

    def initialize_faq_system(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """初始化FAQ系统"""
        index_path = settings.faiss_index_path

        if force_rebuild or not os.path.exists(index_path):
            print("构建新的向量索引...")
            faq_items = self.parse_faq_file(settings.faq_file_path)
            print(f"解析到 {len(faq_items)} 个FAQ条目")
            documents = self.create_documents(faq_items)
            index = self.build_vector_index(documents)
            self.save_index(index, index_path)
            print(f"索引已保存到: {index_path}")
        else:
            print("加载现有索引...")
            index = self.load_index(index_path)

        return index