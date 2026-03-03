from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from contextlib import asynccontextmanager

# 导入 llama-index 相关模块
from llama_index import VectorStoreIndex, Document, ServiceContext
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import MilvusVectorStore
from llama_index.storage.storage_context import StorageContext

# 导入 Milvus 相关模块
from pymilvus import connections, utility

# 全局变量
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'  # 多语言模型，支持中文
FAQ_FILE_PATH = 'faq_data.md'  # FAQ 数据文件路径
MILVUS_HOST = 'localhost'  # Milvus 服务地址
MILVUS_PORT = '19530'  # Milvus 服务端口
COLLECTION_NAME = 'faq_collection'  # Milvus 集合名称

# 全局 FAQ 系统实例
faq_system = None

# 从 MD 文件中读取 FAQ 数据
def load_faq_from_md(file_path):
    """从 MD 文件中加载 FAQ 数据"""
    faqs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 分割 FAQ 条目
        faq_entries = content.strip().split('\n\n')

        for entry in faq_entries:
            if entry.strip():
                # 提取问题和答案
                lines = entry.strip().split('\n')
                question = ''
                answer = ''

                for line in lines:
                    line = line.strip()
                    if line.startswith('Q:'):
                        question = line[2:].strip()
                    elif line.startswith('A:'):
                        answer = line[2:].strip()

                if question and answer:
                    faqs.append({'question': question, 'answer': answer})
    except Exception as e:
        print(f"加载 FAQ 数据失败: {e}")

    return faqs

# 保存 FAQ 数据到 MD 文件
def save_faq_to_md(file_path, faqs):
    """保存 FAQ 数据到 MD 文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for faq in faqs:
                f.write(f"Q: {faq['question']}\n")
                f.write(f"A: {faq['answer']}\n\n")
    except Exception as e:
        print(f"保存 FAQ 数据失败: {e}")

# 定义请求和响应模型
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResponse(BaseModel):
    results: list

class MemoryFAQSystem:
    def __init__(self):
        """初始化 FAQ 系统"""
        # 初始化 llama-index 嵌入模型
        self.llama_embedding = HuggingFaceEmbedding(model_name=MODEL_NAME)

        # 配置文档切片器（语义切分+重叠）
        self.splitter = SentenceSplitter(
            chunk_size=256,  # 每个切片的大小
            chunk_overlap=50,  # 切片之间的重叠部分
            separator="\n"
        )

        # 创建服务上下文（无论使用哪种存储方式都需要）
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.llama_embedding,
            text_splitter=self.splitter,
            llm=None  # 不需要 LLM，因为我们只需要向量索引
        )

        # 存储 FAQ 数据
        self.faqs = []

        # 初始化 Milvus 连接
        self.connect_to_milvus()

        # 初始加载 FAQ 数据
        self.reload_faqs()

    def connect_to_milvus(self):
        """连接到 Milvus 服务"""
        try:
            # 连接到 Milvus 服务
            print(f"正在连接 Milvus 服务: {MILVUS_HOST}:{MILVUS_PORT}")
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            print("成功连接到 Milvus 服务")

            # 检查 Milvus 服务状态
            print("检查 Milvus 服务状态...")
            status = connections.get_connection_addr("default")
            print(f"Milvus 服务状态: {status}")

            # 初始化向量存储
            self.init_vector_store()
        except Exception as e:
            print(f"连接 Milvus 服务失败: {e}")
            print("使用内存存储作为备选方案")
            # 如果连接失败，使用内存存储作为备选
            self.use_memory_storage = True
            self.documents = []
            self.index = None
        else:
            self.use_memory_storage = False

    def init_vector_store(self):
        """初始化 Milvus 向量存储"""
        # 如果集合已存在，删除它
        if utility.has_collection(COLLECTION_NAME):
            print(f"删除现有集合: {COLLECTION_NAME}")
            utility.drop_collection(COLLECTION_NAME)

        # 创建 Milvus 向量存储
        self.vector_store = MilvusVectorStore(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=COLLECTION_NAME,
            dim=384,  # 明确指定向量维度
            overwrite=True
        )

        # 创建存储上下文
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def reload_faqs(self):
        """重新加载 FAQ 数据（热更新）"""
        # 从 MD 文件加载数据
        self.faqs = load_faq_from_md(FAQ_FILE_PATH)

        # 如果没有加载到数据，使用默认数据
        if not self.faqs:
            print("未加载到 FAQ 数据，使用默认数据")
            self.faqs = [
                {"question": "如何申请退货？", "answer": "您可以在订单详情页面点击「申请退货」按钮，按照提示填写退货原因和相关信息，提交后等待审核。审核通过后，系统会生成退货地址，您按照地址寄回商品即可。"},
                {"question": "退货需要满足什么条件？", "answer": "商品需在收到后7天内申请退货，且商品需保持全新未使用状态，包装完整，配件齐全。部分特殊商品（如贴身用品、定制商品等）可能不支持退货，请参考商品详情页的退换货政策。"},
                {"question": "退货流程需要多长时间？", "answer": "退货流程通常需要3-7个工作日完成，具体时间取决于您的寄回速度和银行处理时间。退货申请审核一般在1-2个工作日内完成，商品收到并检验合格后，退款会在1-3个工作日内退回您的支付账户。"},
                {"question": "退货后多久能收到退款？", "answer": "商品收到并检验合格后，退款会在1-3个工作日内退回您的支付账户。不同支付方式的到账时间可能有所不同，银行卡支付一般需要1-3个工作日，第三方支付平台（如支付宝、微信支付）一般需要1-2个工作日。"},
                {"question": "退换货需要承担运费吗？", "answer": "如果是商品质量问题或我们的错误导致的退换货，运费由我们承担。如果是个人原因（如不喜欢、买错等）导致的退换货，运费需要由您承担。具体运费政策请参考商品详情页的退换货政策。"},
                {"question": "如何查询退换货进度？", "answer": "您可以在订单详情页面查看退换货进度，系统会实时更新审核状态、寄回状态、退款状态等信息。如果您有任何疑问，也可以联系客服咨询。"},
                {"question": "退货后可以重新购买吗？", "answer": "是的，退货后您可以重新购买商品。如果您需要重新购买，建议您在确认退货成功后再进行购买，以免影响您的购物体验。"},
                {"question": "如何申请换货？", "answer": "您可以在订单详情页面点击「申请换货」按钮，按照提示填写换货原因和相关信息，提交后等待审核。审核通过后，系统会生成退货地址，您按照地址寄回商品，我们收到后会为您寄出更换的商品。"},
                {"question": "换货需要满足什么条件？", "answer": "商品需在收到后7天内申请换货，且商品需保持全新未使用状态，包装完整，配件齐全。部分特殊商品（如贴身用品、定制商品等）可能不支持换货，请参考商品详情页的退换货政策。"},
                {"question": "换货流程需要多长时间？", "answer": "换货流程通常需要5-10个工作日完成，具体时间取决于您的寄回速度和商品库存情况。换货申请审核一般在1-2个工作日内完成，商品收到并检验合格后，我们会在1-3个工作日内为您寄出更换的商品。"}
            ]
            # 保存默认数据到 MD 文件
            save_faq_to_md(FAQ_FILE_PATH, self.faqs)
            print("已保存默认 FAQ 数据到 MD 文件")

        # 重新构建索引
        self.build_index()

        print("FAQ 数据已热更新")

    def build_index(self):
        """使用 llama-index 构建索引，实现文档切片优化"""
        # 准备文档内容
        documents = []
        for i, faq in enumerate(self.faqs):
            # 组合问题和答案作为文档内容
            content = f"问题: {faq['question']}\n答案: {faq['answer']}"
            # 创建文档对象
            doc = Document(
                text=content,
                metadata={
                    "question": faq['question'],
                    "answer": faq['answer'],
                    "id": i
                }
            )
            documents.append(doc)

        if self.use_memory_storage:
            # 使用内存存储
            self.documents = documents
            # 创建内存索引
            self.index = VectorStoreIndex.from_documents(
                documents,
                service_context=self.service_context
            )
            print(f"构建索引完成，处理了 {len(documents)} 个文档（内存存储）")
        else:
            # 使用 Milvus 存储
            # 创建索引
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                service_context=self.service_context
            )
            print(f"构建索引完成，处理了 {len(documents)} 个文档（Milvus 存储）")

    def add_faq(self, question, answer):
        """添加单个 FAQ"""
        # 创建 FAQ 条目
        faq = {"question": question, "answer": answer}

        # 添加到存储
        self.faqs.append(faq)

        # 更新 MD 文件
        save_faq_to_md(FAQ_FILE_PATH, self.faqs)

        # 重新构建索引
        self.build_index()

        return faq

    def update_faq(self, index, question, answer):
        """更新 FAQ"""
        if 0 <= index < len(self.faqs):
            # 更新 FAQ 条目
            self.faqs[index] = {"question": question, "answer": answer}

            # 更新 MD 文件
            save_faq_to_md(FAQ_FILE_PATH, self.faqs)

            # 重新构建索引
            self.build_index()

            return self.faqs[index]
        else:
            return None

    def delete_faq(self, index):
        """删除 FAQ"""
        if 0 <= index < len(self.faqs):
            # 删除 FAQ 条目
            deleted_faq = self.faqs.pop(index)

            # 更新 MD 文件
            save_faq_to_md(FAQ_FILE_PATH, self.faqs)

            # 重新构建索引
            self.build_index()

            return deleted_faq
        else:
            return None

    def search_faq(self, query, top_k=3):
        """搜索相关的 FAQ"""
        if not self.index:
            return []

        # 创建查询引擎
        query_engine = self.index.as_retriever(similarity_top_k=top_k * 2)

        # 执行搜索
        nodes = query_engine.retrieve(query)

        # 处理结果
        search_results = []
        seen_questions = set()

        for node in nodes:
            question = node.metadata.get("question", "")

            # 去重（基于问题）
            if question not in seen_questions:
                seen_questions.add(question)
                search_results.append({
                    "question": question,
                    "answer": node.metadata.get("answer", ""),
                    "distance": node.score  # llama-index 使用相似度得分，值越高越好
                })

            if len(search_results) >= top_k:
                break

        return search_results

# 定义 lifespan 事件处理程序
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化 FAQ 系统
    global faq_system
    print("初始化 FAQ 系统...")
    faq_system = MemoryFAQSystem()
    yield
    # 关闭时清理资源
    print("清理 FAQ 系统资源...")

# 创建 FastAPI 应用
app = FastAPI(
    title="退换货 FAQ 检索系统",
    description="基于向量搜索的退换货 FAQ 检索系统，支持热更新知识库和 RESTful API 接口",
    version="1.0.0",
    lifespan=lifespan
)

# API 端点
@app.post("/search", response_model=SearchResponse)
async def search_faq(request: SearchRequest):
    """搜索相关的 FAQ"""
    if faq_system is None:
        raise HTTPException(status_code=500, detail="FAQ 系统未初始化")

    results = faq_system.search_faq(request.query, request.top_k)
    return {"results": results}

@app.post("/add")
async def add_faq(question: str, answer: str):
    """添加新的 FAQ"""
    if faq_system is None:
        raise HTTPException(status_code=500, detail="FAQ 系统未初始化")

    result = faq_system.add_faq(question, answer)
    if result:
        return {"message": "FAQ 添加成功", "faq": result}
    else:
        raise HTTPException(status_code=400, detail="FAQ 添加失败")

@app.put("/update/{index}")
async def update_faq(index: int, question: str, answer: str):
    """更新 FAQ"""
    if faq_system is None:
        raise HTTPException(status_code=500, detail="FAQ 系统未初始化")

    result = faq_system.update_faq(index, question, answer)
    if result:
        return {"message": "FAQ 更新成功", "faq": result}
    else:
        raise HTTPException(status_code=400, detail="FAQ 更新失败")

@app.delete("/delete/{index}")
async def delete_faq(index: int):
    """删除 FAQ"""
    if faq_system is None:
        raise HTTPException(status_code=500, detail="FAQ 系统未初始化")

    result = faq_system.delete_faq(index)
    if result:
        return {"message": "FAQ 删除成功", "faq": result}
    else:
        raise HTTPException(status_code=400, detail="FAQ 删除失败")

@app.post("/reload")
async def reload_faqs():
    """重新加载 FAQ 数据（热更新）"""
    if faq_system is None:
        raise HTTPException(status_code=500, detail="FAQ 系统未初始化")

    faq_system.reload_faqs()
    return {"message": "FAQ 数据已热更新"}

def main():
    """主函数"""
    # 初始化 FAQ 系统
    global faq_system
    faq_system = MemoryFAQSystem()

    # 测试查询
    print("\n测试查询...\n")

    test_queries = [
        "如何申请退货？",
        "退货需要满足什么条件？",
        "退货后多久能收到退款？"
    ]

    for query in test_queries:
        print(f"查询: {query}")
        results = faq_system.search_faq(query)
        for i, result in enumerate(results, 1):
            print(f"{i}. 问题: {result['question']}")
            print(f"   回答: {result['answer']}")
            print(f"   相似度: {result['distance']:.4f}")
        print()

    # 交互式查询
    print("交互式查询（输入 'exit' 退出）...")
    while True:
        query = input("请输入您的问题: ")
        if query.lower() == 'exit':
            break

        results = faq_system.search_faq(query)
        for i, result in enumerate(results, 1):
            print(f"{i}. 问题: {result['question']}")
            print(f"   回答: {result['answer']}")
            print(f"   相似度: {result['distance']:.4f}")
        print()

if __name__ == "__main__":
    main()
