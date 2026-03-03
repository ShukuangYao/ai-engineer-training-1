import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 全局变量
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'  # 多语言模型，支持中文
DIMENSION = 384  # 模型输出的向量维度

# 从 MD 文件中读取 FAQ 数据
def load_faq_from_md(file_path):
    """从 MD 文件中加载 FAQ 数据"""
    faqs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    question = None
    answer = []

    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            # 如果已经有问题，先保存之前的 FAQ
            if question and answer:
                faqs.append({
                    "question": question,
                    "answer": ' '.join(answer)
                })
            # 提取新问题
            question = line[3:].strip()
            answer = []
        elif question and line:
            # 累积回答内容
            answer.append(line)

    # 保存最后一个 FAQ
    if question and answer:
        faqs.append({
            "question": question,
            "answer": ' '.join(answer)
        })

    return faqs

# 加载 FAQ 数据
SAMPLE_FAQS = load_faq_from_md('faq_data.md')

class MemoryFAQSystem:
    def __init__(self):
        """初始化内存 FAQ 系统"""
        # 加载预训练模型
        self.model = SentenceTransformer(MODEL_NAME)

        # 存储 FAQ 数据和向量
        self.faqs = []
        self.embeddings = []

    def insert_faqs(self, faqs):
        """插入 FAQ 数据到内存"""
        # 准备数据
        questions = [faq["question"] for faq in faqs]
        answers = [faq["answer"] for faq in faqs]

        # 生成向量嵌入
        new_embeddings = self.model.encode(questions).tolist()

        # 添加到存储
        for i, faq in enumerate(faqs):
            self.faqs.append(faq)
            self.embeddings.append(new_embeddings[i])

        print(f"插入 {len(faqs)} 条 FAQ 数据")

    def search_faq(self, query, top_k=3):
        """搜索相关的 FAQ"""
        # 生成查询向量
        query_embedding = self.model.encode([query])

        # 计算相似度
        similarities = cosine_similarity(query_embedding, np.array(self.embeddings))[0]

        # 排序并获取 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 处理搜索结果
        search_results = []
        for idx in top_indices:
            search_results.append({
                "question": self.faqs[idx]["question"],
                "answer": self.faqs[idx]["answer"],
                "distance": 1 - similarities[idx]  # 转换为距离，值越小相似度越高
            })

        return search_results

def main():
    """主函数"""
    # 初始化系统
    print("初始化内存 FAQ 系统...")
    faq_system = MemoryFAQSystem()

    # 插入示例数据
    print("插入示例 FAQ 数据...")
    faq_system.insert_faqs(SAMPLE_FAQS)

    # 测试查询
    print("\n测试查询...")
    test_queries = [
        "Milvus 是什么？",
        "如何安装 Milvus？",
        "Milvus 支持哪些语言？"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = faq_system.search_faq(query)
        for i, result in enumerate(results, 1):
            print(f"{i}. 问题: {result['question']}")
            print(f"   回答: {result['answer']}")
            print(f"   距离: {result['distance']:.4f}")

    # 交互式查询
    print("\n交互式查询（输入 'exit' 退出）...")
    while True:
        query = input("请输入您的问题: ")
        if query.lower() == 'exit':
            break

        results = faq_system.search_faq(query)
        print("搜索结果:")
        for i, result in enumerate(results, 1):
            print(f"{i}. 问题: {result['question']}")
            print(f"   回答: {result['answer']}")
            print(f"   距离: {result['distance']:.4f}")

if __name__ == "__main__":
    main()