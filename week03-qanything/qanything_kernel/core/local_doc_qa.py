"""
本地文档问答（LocalDocQA）核心模块，实现完整的 RAG（检索增强生成）流程。

本模块提供 LocalDocQA 类，负责：
- 文档检索：向量检索（Milvus）+ 关键词检索（ES）+ 可选混合检索
- 重排序：对检索结果进行精排，过滤低分文档
- 上下文构建：在 token 限制内选取文档、生成 prompt
- 问答生成：多轮对话、问题改写、流式/非流式回答、FAQ 匹配、联网搜索

依赖：嵌入模型（YouDaoEmbeddings）、重排模型（YouDaoRerank）、Milvus、ES、MySQL（KnowledgeBaseManager）。
"""
from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING, SYSTEM, INSTRUCTIONS, SIMPLE_PROMPT_TEMPLATE, CUSTOM_PROMPT_TEMPLATE, \
    LOCAL_RERANK_MODEL_NAME, LOCAL_EMBED_MAX_LENGTH, SEPARATORS
from typing import List, Tuple, Union, Dict
import time
from scipy.spatial import cKDTree
from scipy.spatial.distance import cosine
from scipy.stats import gmean
from qanything_kernel.connector.embedding.embedding_for_online_client import YouDaoEmbeddings
from qanything_kernel.connector.rerank.rerank_for_online_client import YouDaoRerank
from qanything_kernel.connector.llm import OpenAILLM
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.core.retriever.vectorstore import VectorStoreMilvusClient
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
from qanything_kernel.core.retriever.parent_retriever import ParentRetriever
from qanything_kernel.utils.general_utils import (get_time, clear_string, get_time_async, num_tokens,
                                                  cosine_similarity, clear_string_is_equal, num_tokens_embed,
                                                  num_tokens_rerank, deduplicate_documents, replace_image_references)
from qanything_kernel.utils.custom_log import debug_logger, qa_logger, rerank_logger
from qanything_kernel.core.chains.condense_q_chain import RewriteQuestionChain
from qanything_kernel.core.tools.web_search_tool import duckduckgo_search
import copy
import requests
import json
import numpy as np
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import traceback
import re


class LocalDocQA:
    """
    本地文档问答系统核心类 - 实现完整的RAG(Retrieval-Augmented Generation)能力

    核心RAG流程：
    1. 文档检索(Retrieval): 通过向量相似度和混合搜索从知识库中检索相关文档
    2. 重排序(Rerank): 使用专门的重排序模型对检索结果进行精确排序
    3. 上下文构建: 智能处理token限制，构建最优的prompt上下文
    4. 生成回答(Generation): 基于检索到的文档生成准确回答

    为什么要这样设计：
    - 多模态检索: 结合向量检索(语义相似)和关键词检索(精确匹配)提高召回率
    - 智能重排序: 解决向量检索可能的语义偏差，提高检索精度
    - Token优化: 在有限的上下文窗口内最大化有用信息的利用
    - 流式生成: 提供更好的用户体验，支持实时响应
    """

    def __init__(self, port):
        """
        初始化 LocalDocQA 实例，仅设置端口与默认属性；实际组件（embeddings、rerank、milvus、es 等）由 init_cfg 初始化。

        Args:
            port: 服务端口号，用于标识或绑定服务。
        """
        self.port = port
        self.milvus_cache = None  # Milvus 向量库缓存（若按 kb 缓存 collection 可在此维护）
        self.embeddings: YouDaoEmbeddings = None  # 文本嵌入模型，将文本转为向量
        self.rerank: YouDaoRerank = None  # 重排序模型，对检索结果精排
        self.chunk_conent: bool = True  # 是否启用文档分块
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD  # 向量检索分数阈值
        self.milvus_kb: VectorStoreMilvusClient = None  # Milvus 向量库客户端
        self.retriever: ParentRetriever = None  # 统一检索器（向量 + 关键词）
        self.milvus_summary: KnowledgeBaseManager = None  # 知识库元数据与文档索引（MySQL）
        self.es_client: StoreElasticSearchClient = None  # ES 客户端，用于关键词检索
        self.session = self.create_retry_session(retries=3, backoff_factor=1)  # 带重试的 HTTP 会话
        # 文档分割器：将长文档切成适合嵌入与重排的小段（用于 calculate_relevance_optimized 等）
        self.doc_splitter = CharacterTextSplitter(
            chunk_size=LOCAL_EMBED_MAX_LENGTH / 2,
            chunk_overlap=0,
            length_function=len
        )

    @staticmethod
    def create_retry_session(retries, backoff_factor):
        """
        创建带重试机制的HTTP会话

        为什么需要重试机制：
        - 网络不稳定时保证服务可用性
        - 处理临时的服务器错误(5xx错误)
        - 提高系统的鲁棒性和用户体验

        Args:
            retries: 重试次数
            backoff_factor: 退避因子，控制重试间隔

        Returns:
            配置了重试策略的requests.Session对象
        """
        session = requests.Session()
        retry = Retry(
            total=retries,  # 总重试次数
            read=retries,   # 读取重试次数
            connect=retries,  # 连接重试次数
            backoff_factor=backoff_factor,  # 重试间隔的退避因子
            status_forcelist=[500, 502, 503, 504],  # 需要重试的HTTP状态码
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def init_cfg(self, args=None):
        """
        初始化配置 - 构建完整的RAG技术栈

        为什么需要这些组件：
        1. YouDaoEmbeddings: 将文本转换为高维向量，支持语义相似度计算
        2. YouDaoRerank: 对初步检索结果进行精确重排序，提高相关性
        3. KnowledgeBaseManager: 管理知识库元数据和文档索引
        4. VectorStoreMilvusClient: 高性能向量数据库，支持大规模向量检索
        5. StoreElasticSearchClient: 全文检索引擎，支持关键词精确匹配
        6. ParentRetriever: 整合多种检索策略的统一检索器

        Args:
            args: 可选的配置参数
        """
        self.embeddings = YouDaoEmbeddings()  # 初始化嵌入模型
        self.rerank = YouDaoRerank()  # 初始化重排序模型
        self.milvus_summary = KnowledgeBaseManager()  # 初始化知识库管理器
        self.milvus_kb = VectorStoreMilvusClient()  # 初始化向量数据库客户端
        self.es_client = StoreElasticSearchClient()  # 初始化ElasticSearch客户端
        # 初始化父级检索器，整合向量检索和关键词检索
        self.retriever = ParentRetriever(self.milvus_kb, self.milvus_summary, self.es_client)

    @get_time
    def get_web_search(self, queries, top_k):
        """
        使用 DuckDuckGo 进行联网搜索，将结果转为与知识库文档格式一致的 Document 列表，便于后续与 KB 结果合并。

        Args:
            queries: 查询列表，当前仅使用第一个元素。
            top_k: 返回的搜索结果条数上限。

        Returns:
            (web_content, source_documents): 汇总文本与 Document 列表；每个 doc 带 file_id、file_name、score、embed_version 等。
        """
        query = queries[0]
        web_content, web_documents = duckduckgo_search(query, top_k)
        source_documents = []
        for idx, doc in enumerate(web_documents):
            if 'title' not in doc.metadata:
                continue
            doc.metadata['retrieval_query'] = query
            debug_logger.info(f"web search doc: {doc.metadata}")
            file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', doc.metadata['title'])
            doc.metadata['file_name'] = file_name + '.web'
            doc.metadata['file_url'] = doc.metadata['source']
            doc.metadata['embed_version'] = self.embeddings.embed_version
            doc.metadata['score'] = 1 - (idx / len(web_documents))  # 按顺序给分，越靠前分数越高
            doc.metadata['file_id'] = 'websearch' + str(idx)
            doc.metadata['headers'] = {"新闻标题": file_name}
            if 'description' in doc.metadata:
                desc_doc = Document(page_content=doc.metadata['description'], metadata=doc.metadata)
                source_documents.append(desc_doc)
            source_documents.append(doc)
        return web_content, source_documents

    def web_page_search(self, query, top_k=None):
        """
        对外联网搜索接口：调用 get_web_search，异常时返回空列表，避免影响主流程。

        Args:
            query: 用户查询字符串。
            top_k: 返回条数，默认由调用方指定。

        Returns:
            格式化后的 Document 列表；失败时返回 []。
        """
        try:
            web_content, source_documents = self.get_web_search([query], top_k)
        except Exception as e:
            debug_logger.error(f"web search error: {traceback.format_exc()}")
            return []
        return source_documents

    @get_time_async
    async def get_source_documents(self, query, retriever: ParentRetriever, kb_ids, time_record, hybrid_search, top_k):
        """
        从知识库检索相关文档 - RAG的核心检索阶段

        为什么这样设计检索流程：
        1. 混合检索策略: 结合向量检索(语义相似)和关键词检索(精确匹配)
        2. 容错机制: 当Milvus连接失败时自动重启客户端，保证服务可用性
        3. 文档过滤: 过滤已删除的文档，确保检索结果的有效性
        4. 分数标准化: 为后续重排序提供统一的分数基准

        Args:
            query: 用户查询
            retriever: 检索器实例
            kb_ids: 知识库ID列表
            time_record: 时间记录字典
            hybrid_search: 是否启用混合搜索
            top_k: 返回的文档数量

        Returns:
            检索到的相关文档列表
        """
        source_documents = []
        start_time = time.perf_counter()

        # 执行检索：向量检索或混合检索（向量 + ES），按 kb_ids 分区
        query_docs = await retriever.get_retrieved_documents(query, partition_keys=kb_ids, time_record=time_record,
                                                             hybrid_search=hybrid_search, top_k=top_k)

        # 容错：检索为空时重启 Milvus 客户端并重试一次
        if len(query_docs) == 0:
            debug_logger.warning("MILVUS SEARCH ERROR, RESTARTING MILVUS CLIENT!")
            retriever.vectorstore_client = VectorStoreMilvusClient()
            debug_logger.warning("MILVUS CLIENT RESTARTED!")
            query_docs = await retriever.get_retrieved_documents(query, partition_keys=kb_ids, time_record=time_record,
                                                                    hybrid_search=hybrid_search, top_k=top_k)

        end_time = time.perf_counter()
        time_record['retriever_search'] = round(end_time - start_time, 2)
        debug_logger.info(f"retriever_search time: {time_record['retriever_search']}s")

        # 处理检索结果，添加元数据和分数标准化
        for idx, doc in enumerate(query_docs):
            # 过滤已删除的文档
            if retriever.mysql_client.is_deleted_file(doc.metadata['file_id']):
                debug_logger.warning(f"file_id: {doc.metadata['file_id']} is deleted")
                continue

            doc.metadata['retrieval_query'] = query  # 记录检索查询，用于后续分析
            doc.metadata['embed_version'] = self.embeddings.embed_version  # 记录嵌入模型版本

            # 如果没有分数，使用位置倒序作为默认分数
            if 'score' not in doc.metadata:
                doc.metadata['score'] = 1 - (idx / len(query_docs))

            source_documents.append(doc)

        debug_logger.info(f"embed scores: {[doc.metadata['score'] for doc in source_documents]}")
        return source_documents

    def reprocess_source_documents(self, custom_llm: OpenAILLM, query: str,
                                   source_docs: List[Document],
                                   history: List[str],
                                   prompt_template: str) -> Tuple[List[Document], int, str]:
        """
        智能处理源文档以适应Token限制 - RAG系统的关键优化环节

        为什么需要这个函数：
        1. Token限制: LLM有固定的上下文窗口，需要在有限空间内最大化信息利用
        2. 成本控制: 减少不必要的token消耗，降低API调用成本
        3. 质量保证: 确保最相关的文档内容能够被包含在prompt中
        4. 性能优化: 避免超长prompt导致的响应延迟

        处理策略：
        - 精确计算各部分token消耗
        - 优先保留高质量文档
        - 智能截断而非简单丢弃

        Args:
            custom_llm: LLM实例
            query: 用户查询
            source_docs: 源文档列表
            history: 对话历史
            prompt_template: prompt模板

        Returns:
            (处理后的文档列表, 可用token数量, token使用说明)
        """
        # 计算各固定部分的 token 消耗
        query_token_num = int(custom_llm.num_tokens_from_messages([query]) * 4)  # 查询预留 4 倍空间
        history_token_num = int(custom_llm.num_tokens_from_messages([x for sublist in history for x in sublist]))
        template_token_num = int(custom_llm.num_tokens_from_messages([prompt_template]))
        reference_field_token_num = int(custom_llm.num_tokens_from_messages(
            [f"<reference>[{idx + 1}]</reference>" for idx in range(len(source_docs))]))

        # 文档可用 token = 总窗口 - 输出预留 - 安全边界 - 查询/历史/模板/引用标签
        limited_token_nums = custom_llm.token_window - custom_llm.max_token - custom_llm.offcut_token - query_token_num - history_token_num - template_token_num - reference_field_token_num

        debug_logger.info(f"=============================================")
        debug_logger.info(f"token_window = {custom_llm.token_window}")
        debug_logger.info(f"max_token = {custom_llm.max_token}")
        debug_logger.info(f"offcut_token = {custom_llm.offcut_token}")
        debug_logger.info(f"limited token nums: {limited_token_nums}")
        debug_logger.info(f"template token nums: {template_token_num}")
        debug_logger.info(f"reference_field token nums: {reference_field_token_num}")
        debug_logger.info(f"query token nums: {query_token_num}")
        debug_logger.info(f"history token nums: {history_token_num}")
        debug_logger.info(f"=============================================")

        tokens_msg = """
        token_window = {custom_llm.token_window}, max_token = {custom_llm.max_token},
        offcut_token = {custom_llm.offcut_token}, docs_available_token_nums: {limited_token_nums},
        template token nums: {template_token_num}, reference_field token nums: {reference_field_token_num},
        query token nums: {query_token_num}, history token nums: {history_token_num}
        docs_available_token_nums = token_window - max_token - offcut_token - query_token_num * 4 - history_token_num - template_token_num - reference_field_token_num
        """.format(custom_llm=custom_llm, limited_token_nums=limited_token_nums, template_token_num=template_token_num,
                     reference_field_token_num=reference_field_token_num, query_token_num=query_token_num // 4,
                     history_token_num=history_token_num)

        # 按文档顺序“装填”：在 limited_token_nums 内尽可能多保留靠前的文档（重排后已按相关性排序）
        new_source_docs = []
        total_token_num = 0
        not_repeated_file_ids = []
        for doc in source_docs:
            headers_token_num = 0
            file_id = doc.metadata['file_id']
            if file_id not in not_repeated_file_ids:
                not_repeated_file_ids.append(file_id)
                if 'headers' in doc.metadata:
                    headers = f"headers={doc.metadata['headers']}"
                    headers_token_num = custom_llm.num_tokens_from_messages([headers])
            doc_valid_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)  # 去掉图片占位再算 token
            doc_token_num = custom_llm.num_tokens_from_messages([doc_valid_content])
            doc_token_num += headers_token_num
            if total_token_num + doc_token_num <= limited_token_nums:
                new_source_docs.append(doc)
                total_token_num += doc_token_num
            else:
                break

        debug_logger.info(f"new_source_docs token nums: {custom_llm.num_tokens_from_docs(new_source_docs)}")
        return new_source_docs, limited_token_nums, tokens_msg

    def generate_prompt(self, query, source_docs, prompt_template):
        """
        根据检索到的文档与模板生成最终发给 LLM 的 prompt。同一 file_id 的多个 chunk 合并为一段，用 <reference>[n] 包裹。

        Args:
            query: 用户问题。
            source_docs: 已选中的源文档列表（含 page_content、metadata）。
            prompt_template: 含 {{context}}、{{question}} 占位符的模板。

        Returns:
            替换占位符后的完整 prompt 字符串。
        """
        if source_docs:
            context = ''
            not_repeated_file_ids = []
            for doc in source_docs:
                doc_valid_content = re.sub(r'!\[figure]\(.*?\)', '', doc.page_content)
                file_id = doc.metadata['file_id']
                if file_id not in not_repeated_file_ids:
                    if len(not_repeated_file_ids) != 0:
                        context += '</reference>\n'
                    not_repeated_file_ids.append(file_id)
                    if 'headers' in doc.metadata:
                        headers = f"headers={doc.metadata['headers']}"
                        context += f"<reference {headers}>[{len(not_repeated_file_ids)}]" + '\n' + doc_valid_content + '\n'
                    else:
                        context += f"<reference>[{len(not_repeated_file_ids)}]" + '\n' + doc_valid_content + '\n'
                else:
                    context += doc_valid_content + '\n'
            context += '</reference>\n'
            prompt = prompt_template.replace("{{context}}", context).replace("{{question}}", query)
        else:
            prompt = prompt_template.replace("{{question}}", query)
        return prompt

    async def get_rerank_results(self, query, doc_ids=None, doc_strs=None):
        """
        对给定 query 与文档列表（doc_ids 或 doc_strs）做重排序，返回带 score 的 Document 列表。
        用于 handler 中单独调用重排接口或调试。query 过长（>300 token）或仅 1 条文档时用向量相似度打分。

        Args:
            query: 检索/重排查询。
            doc_ids: 文档 ID 列表，从 MySQL 取内容；与 doc_strs 二选一。
            doc_strs: 文档文本列表，直接构造 Document；与 doc_ids 二选一。

        Returns:
            按相关性排序的 Document 列表，每条带 metadata['score']；重排时过滤 score < 0.28。
        """
        docs = []
        if doc_strs:
            docs = [Document(page_content=doc_str) for doc_str in doc_strs]
        else:
            for doc_id in doc_ids:
                doc_json = self.milvus_summary.get_document_by_doc_id(doc_id)
                if doc_json is None:
                    docs.append(None)
                    continue
                user_id, file_id, file_name, kb_id = doc_json['kwargs']['metadata']['user_id'], \
                    doc_json['kwargs']['metadata']['file_id'], doc_json['kwargs']['metadata']['file_name'], \
                    doc_json['kwargs']['metadata']['kb_id']
                doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])
                doc.metadata['doc_id'] = doc_id
                doc.metadata['retrieval_query'] = query
                doc.metadata['embed_version'] = self.embeddings.embed_version
                if file_name.endswith('.faq'):
                    faq_dict = doc.metadata['faq_dict']
                    page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                    nos_keys = faq_dict.get('nos_keys')
                    doc.page_content = page_content
                    doc.metadata['nos_keys'] = nos_keys
                docs.append(doc)

        if len(docs) > 1 and num_tokens_rerank(query) <= 300:
            try:
                debug_logger.info(f"use rerank, rerank docs num: {len(docs)}")
                docs = await self.rerank.arerank_documents(query, docs)
                if len(docs) > 1:
                    docs = [doc for doc in docs if doc.metadata['score'] >= 0.28]
                return docs
            except Exception as e:
                debug_logger.error(f"query tokens: {num_tokens_rerank(query)}, rerank error: {e}")
                # 重排失败时退化为 query 与每段内容的向量相似度
                embed1 = await self.embeddings.aembed_query(query)
                for doc in docs:
                    embed2 = await self.embeddings.aembed_query(doc.page_content)
                    doc.metadata['score'] = cosine_similarity(embed1, embed2)
                return docs
        else:
            # 仅 1 条或 query 过长：用嵌入相似度打分，不调用重排
            embed1 = await self.embeddings.aembed_query(query)
            for doc in docs:
                embed2 = await self.embeddings.aembed_query(doc.page_content)
                doc.metadata['score'] = cosine_similarity(embed1, embed2)
            return docs

    async def prepare_source_documents(self, custom_llm: OpenAILLM, retrieval_documents: List[Document],
                                       limited_token_nums: int, rerank: bool):
        """
        在 token 限制内对检索文档做聚合或排序，得到最终用于生成 prompt 的 source_documents。
        可选逻辑：aggregate_documents 将候选集中在一两个文件时返回完整文档或按 doc_id 范围截取。

        Args:
            custom_llm: LLM 实例，用于 token 统计。
            retrieval_documents: 经 reprocess_source_documents 裁切后的检索文档列表。
            limited_token_nums: 文档可用 token 上限。
            rerank: 是否做过重排（影响 aggregate 时取 max/min score）。

        Returns:
            (source_documents, retrieval_documents)：前者用于 generate_prompt，后者用于返回给前端。
        """
        debug_logger.info(f"retrieval_documents len: {len(retrieval_documents)}")
        source_documents = retrieval_documents

        try:
            # 检查是否需要聚合（候选集中在一两个文件）
            file_ids = set(doc.metadata.get('file_id', '') for doc in retrieval_documents)
            if len(file_ids) <= 2 and len(retrieval_documents) > 1:
                # 候选集中在一两个文件，尝试聚合
                new_docs = self.aggregate_documents(retrieval_documents, limited_token_nums, custom_llm, rerank)
                if new_docs:
                    source_documents = new_docs
                    debug_logger.info(f"Aggregated documents, new length: {len(source_documents)}")

            # 如果没有聚合或聚合失败，按 file_id 合并并排序
            if source_documents == retrieval_documents:
                # 合并所有候选文档，从前往后，所有file_id相同的文档合并，按照doc_id排序
                merged_documents_file_ids = []
                for doc in retrieval_documents:
                    file_id = doc.metadata.get('file_id', '')
                    if file_id not in merged_documents_file_ids:
                        merged_documents_file_ids.append(file_id)

                merged_source_documents = []
                for file_id in merged_documents_file_ids:
                    docs = [doc for doc in retrieval_documents if doc.metadata.get('file_id', '') == file_id]
                    # 按 doc_id 排序
                    try:
                        docs = sorted(docs, key=lambda x: int(x.metadata.get('doc_id', '0').split('_')[-1]))
                    except (ValueError, IndexError):
                        # 如果 doc_id 格式不正确，保持原顺序
                        pass
                    merged_source_documents.extend(docs)

                if merged_source_documents:
                    source_documents = merged_source_documents
                    debug_logger.info(f"Merged documents by file_id, new length: {len(source_documents)}")

            # 可选：处理不完整的表格
            # source_documents = self.incomplete_table(source_documents, limited_token_nums, custom_llm)

        except Exception as e:
            debug_logger.error(f"aggregate_documents error w/ {e}: {traceback.format_exc()}")
            source_documents = retrieval_documents

        debug_logger.info(f"source_documents len: {len(source_documents)}")
        return source_documents, retrieval_documents

    async def calculate_relevance_optimized(
            self,
            question: str,
            llm_answer: str,
            reference_docs: List[Document],
            top_k: int = 5
    ) -> List[Dict]:
        """
        根据 LLM 回答与引用文档的相似度，选出与回答最相关的 top_k 个文档/段落，用于带图回答时挑选展示图片。
        流程：将 reference_docs 按 doc_splitter 切段 → 对 LLM 回答与各段做 embedding → 用 KD 树找最相似 top_k 段
        → 按段反推所属文档 → 用加权几何平均（问题分数与 LLM-段相似度各 0.5）得综合分并排序。

        Args:
            question: 用户问题（用于取 reference_docs 的 score）。
            llm_answer: LLM 生成的回答文本。
            reference_docs: 带图片的引用文档列表（通常为 file_name 非 .faq 的 doc）。
            top_k: 返回的相关文档/段数量。

        Returns:
            按 combined_score 降序的列表，每项含 document、segment、similarity_llm、question_score、combined_score。
        """
        # 从参考文档中提取问题相关分数
        # 每个文档的 metadata 中存储了与原始问题的相关分数
        question_scores = [doc.metadata['score'] for doc in reference_docs]

        # 计算LLM回答的embedding
        # 将LLM的回答转换为向量表示，用于后续的相似度计算
        llm_answer_embedding = await self.embeddings.aembed_query(llm_answer)

        # 计算所有引用文档分段的embeddings
        # 1. 将每个文档分割成更小的段落，以便更精确地计算相似度
        all_segments_docs = self.doc_splitter.split_documents(reference_docs)
        # 2. 提取每个段落的内容
        all_segments = [doc.page_content for doc in all_segments_docs]
        # 3. 将所有段落转换为向量表示
        reference_embeddings = await self.embeddings.aembed_documents(all_segments)

        # 将嵌入向量转换为numpy数组以便使用scipy的cosine函数
        llm_answer_embedding = np.array(llm_answer_embedding)
        reference_embeddings = np.array(reference_embeddings)

        # 构建KD树
        # KD树是一种空间划分数据结构，用于高效地进行最近邻搜索
        tree = cKDTree(reference_embeddings)

        # 使用KD树找到最相似的分段
        # 1. 将LLM回答的embedding重塑为二维数组
        # 2. 查找与LLM回答最相似的top_k个段落
        _, indices = tree.query(llm_answer_embedding.reshape(1, -1), k=top_k)
        # 处理返回结果的格式，确保indices是二维列表
        if isinstance(indices[0], np.int64):
            indices = [indices]

        # 计算每个文档的分段数量，以便根据索引找到对应的文档
        # 为每个文档计算其被分割成的段落数量
        doc_segment_lengths = [len(self.doc_splitter.split_documents([doc])) for doc in reference_docs]

        # 创建一个累积的段落索引，用于根据段落找到文档ID
        # 例如，如果文档0有2个段落，文档1有3个段落，那么累积索引为[0, 2, 5]
        # 这样，段落索引2-4属于文档1
        cumulative_lengths = np.cumsum([0] + doc_segment_lengths)

        # 定义加权几何平均函数
        # 用于计算综合得分，结合LLM回答与段落的相似度和原始问题的相关分数
        def weighted_geometric_mean(scores, weights):
            # 对每个分数按权重取幂，然后计算几何平均值
            return gmean([score ** weight for score, weight in zip(scores, weights)])

        # 计算相似度和综合得分
        relevant_docs = []
        # 遍历找到的最相似段落的索引
        for doc_index in indices[0]:
            # 根据doc_index找到对应的文档ID
            # 使用searchsorted找到doc_index所在的文档区间
            doc_id = np.searchsorted(cumulative_lengths, doc_index, side='right') - 1

            # 获取该文档内的实际分段索引
            # 从段落的全局索引中减去文档的起始索引，得到文档内的相对索引
            segment_index_in_doc = doc_index - cumulative_lengths[doc_id]

            # 计算1 - cosine距离来计算相似度
            # cosine距离范围是[0, 2]，所以1 - cosine距离范围是[-1, 1]，值越大表示相似度越高
            similarity_llm = 1 - cosine(llm_answer_embedding, reference_embeddings[doc_index])
            # 获取该文档的原始问题相关分数
            rerank_score = question_scores[doc_id]

            # 设置rerank分数和LLM回答与文档余弦相似度的权重
            # 两者权重各为0.5，平衡原始问题相关性和LLM回答相似度
            weights = [0.5, 0.5]  # 分别对应similarity_llm和rerank_score
            # 计算综合得分
            combined_score = weighted_geometric_mean([similarity_llm, rerank_score], weights)

            # 添加到结果列表
            relevant_docs.append({
                'document': reference_docs[doc_id],  # 原始文档对象
                'segment': all_segments_docs[doc_index],  # 最相似的段落
                'similarity_llm': float(similarity_llm),  # LLM回答与段落的相似度
                'question_score': question_scores[doc_id],  # 原始问题与文档的相关分数
                'combined_score': float(combined_score)  # 综合得分
            })

        # 按综合得分降序排序
        # 确保最相关的文档排在前面
        relevant_docs.sort(key=lambda x: x['combined_score'], reverse=True)

        # 返回排序后的结果
        return relevant_docs

    @staticmethod
    async def generate_response(query, res, condense_question, source_documents, time_record, chat_history, streaming, prompt):
        """
        生成response并使用yield返回。

        :param query: 用户的原始查询
        :param res: 生成的答案
        :param condense_question: 压缩后的问题
        :param source_documents: 从检索中获取的文档
        :param time_record: 记录时间的字典
        :param chat_history: 聊天历史
        :param streaming: 是否启用流式输出
        :param prompt: 生成response时的prompt类型
        """
        history = chat_history + [[query, res]]

        if streaming:
            res = 'data: ' + json.dumps({'answer': res}, ensure_ascii=False)

        response = {
            "query": query,
            "prompt": prompt,  # 允许自定义 prompt
            "result": res,
            "condense_question": condense_question,
            "retrieval_documents": source_documents,
            "source_documents": source_documents
        }

        if 'llm_completed' not in time_record:
            time_record['llm_completed'] = 0.0
        if 'total_tokens' not in time_record:
            time_record['total_tokens'] = 0
        if 'prompt_tokens' not in time_record:
            time_record['prompt_tokens'] = 0
        if 'completion_tokens' not in time_record:
            time_record['completion_tokens'] = 0

        yield response, history

        if streaming:
            response['result'] = "data: [DONE]\n\n"
            yield response, history

    async def get_knowledge_based_answer(self, model, max_token, kb_ids, query, retriever, custom_prompt, time_record,
                                         temperature, api_base, api_key, api_context_length, top_p, top_k, web_chunk_size,
                                         chat_history=None, streaming: bool = STREAMING, rerank: bool = False,
                                         only_need_search_results: bool = False, need_web_search=False,
                                         hybrid_search=False):
        """
        知识库问答主流程（异步生成器）。步骤：多轮改写问题 → 知识库检索 → 可选联网检索 → 去重 → 可选重排 → FAQ 完全匹配短路
        → 选 prompt 模板 → reprocess 裁切 token → 生成 prompt → 流式/非流式调用 LLM → 带图时计算相关文档并附加 show_images。

        Args:
            model, max_token, api_base, api_key, api_context_length, top_p, temperature: LLM 与上下文参数。
            kb_ids: 知识库 ID 列表；空则仅联网/纯对话。
            query: 用户问题。
            retriever: 检索器实例。
            custom_prompt: 自定义 prompt 文本；为空用默认模板。
            time_record: 记录各阶段耗时的字典。
            chat_history: 多轮对话历史 [[user, assistant], ...]。
            streaming: 是否流式输出。
            rerank: 是否对检索结果重排。
            only_need_search_results: 为 True 时只返回检索文档，不调用 LLM。
            need_web_search: 是否联网搜索并合并结果。
            hybrid_search: 是否启用向量+关键词混合检索。
            web_chunk_size: 联网结果分片大小（token）。

        Yields:
            (response_dict, history)：response_dict 含 query、prompt、result、condense_question、retrieval_documents、source_documents 等；流式结束时 result 为 data: [DONE]。
        """
        # 初始化 LLM 客户端，用于后续的文本生成和 token 计算
        custom_llm = OpenAILLM(model, max_token, api_base, api_key, api_context_length, top_p, temperature)

        # 初始化对话历史（如果为 None）
        if chat_history is None:
            chat_history = []

        # 初始化检索查询和浓缩问题（默认为原始查询）
        retrieval_query = query
        condense_question = query

        # 处理多轮对话历史
        if chat_history:
            # 格式化对话历史为 LLM 可理解的消息格式
            formatted_chat_history = []
            for msg in chat_history:
                formatted_chat_history += [
                    HumanMessage(content=msg[0]),  # 用户消息
                    AIMessage(content=msg[1]),     # 助手回复
                ]
            debug_logger.info(f"formatted_chat_history: {formatted_chat_history}")

            # 初始化问题重写链，用于将多轮对话浓缩为单个查询
            rewrite_q_chain = RewriteQuestionChain(model_name=model, openai_api_base=api_base, openai_api_key=api_key)

            # 构建问题重写的完整提示
            full_prompt = rewrite_q_chain.condense_q_prompt.format(
                chat_history=formatted_chat_history,
                question=query
            )

            # 历史过长时从最早的一轮开始丢弃，直到 prompt token 小于 4096-256
            while custom_llm.num_tokens_from_messages([full_prompt]) >= 4096 - 256:
                # 格式通常是 [HumanMessage1, AIMessage1, HumanMessage2, AIMessage2, ...]
                # 所以从index为2开始，就是移除最早的一轮的HumanMessage1, AIMessage1
                formatted_chat_history = formatted_chat_history[2:]  # 移除最早的一轮对话
                full_prompt = rewrite_q_chain.condense_q_prompt.format(
                    chat_history=formatted_chat_history,
                    question=query
                )
            debug_logger.info(
                f"Subtract formatted_chat_history: {len(chat_history) * 2} -> {len(formatted_chat_history)}")

            try:
                # 记录问题重写开始时间
                t1 = time.perf_counter()
                # 异步调用问题重写链，生成浓缩后的问题
                condense_question = await rewrite_q_chain.condense_q_chain.ainvoke(
                    {
                        "chat_history": formatted_chat_history,
                        "question": query,
                    },
                )
                # 记录问题重写结束时间并计算耗时
                t2 = time.perf_counter()
                # 时间保留两位小数
                time_record['condense_q_chain'] = round(t2 - t1, 2)
                # 记录重写完成的 token 数
                time_record['rewrite_completion_tokens'] = custom_llm.num_tokens_from_messages([condense_question])
                debug_logger.info(f"condense_q_chain time: {time_record['condense_q_chain']}s")
            except Exception as e:
                # 问题重写出错时，使用原始查询作为浓缩问题
                debug_logger.error(f"condense_q_chain error: {e}")
                condense_question = query

            # 打印浓缩后的问题
            debug_logger.info(f"condense_question: {condense_question}")
            # 记录重写提示的 token 数
            time_record['rewrite_prompt_tokens'] = custom_llm.num_tokens_from_messages([full_prompt, condense_question])

            # 判断两个字符串是否相似：只保留中文，英文和数字
            if clear_string(condense_question) != clear_string(query):
                # 如果浓缩问题与原始查询不同，使用浓缩问题进行检索
                retrieval_query = condense_question

        # 知识库检索（若有 kb_ids）
        if kb_ids:
            # 异步获取源文档，支持混合检索
            source_documents = await self.get_source_documents(retrieval_query, retriever, kb_ids, time_record,
                                                               hybrid_search, top_k)
        else:
            # 无知识库 ID 时，源文档为空列表
            source_documents = []

        # 可选：联网搜索，分片后并入 source_documents 并写入 MySQL 文档表
        if need_web_search:
            # 记录联网搜索开始时间
            t1 = time.perf_counter()
            # 执行联网搜索，获取前 3 个结果
            web_search_results = self.web_page_search(query, top_k=3)

            # 初始化文本分片器，用于将网页内容分片
            web_splitter = RecursiveCharacterTextSplitter(
                separators=SEPARATORS,
                chunk_size=web_chunk_size,
                chunk_overlap=int(web_chunk_size / 4),  # 分片重叠度为 1/4
                length_function=num_tokens_embed,  # 使用嵌入模型的 token 计算函数
            )
            # 对网页搜索结果进行分片
            web_search_results = web_splitter.split_documents(web_search_results)

            # 为每个分片生成唯一的文档 ID
            current_doc_id = 0
            current_file_id = web_search_results[0].metadata['file_id']
            for doc in web_search_results:
                if doc.metadata['file_id'] == current_file_id:
                    # 同一文件的分片，递增文档 ID
                    doc.metadata['doc_id'] = current_file_id + '_' + str(current_doc_id)
                    current_doc_id += 1
                else:
                    # 新文件，重置文档 ID 计数器
                    current_file_id = doc.metadata['file_id']
                    current_doc_id = 0
                    doc.metadata['doc_id'] = current_file_id + '_' + str(current_doc_id)
                    current_doc_id += 1

                # 将文档转换为 JSON 格式并存储到数据库
                doc_json = doc.to_json()
                if doc_json['kwargs'].get('metadata') is None:
                    doc_json['kwargs']['metadata'] = doc.metadata
                self.milvus_summary.add_document(doc_id=doc.metadata['doc_id'], json_data=doc_json)

            # 记录联网搜索结束时间并计算耗时
            t2 = time.perf_counter()
            time_record['web_search'] = round(t2 - t1, 2)

            # 将网页搜索结果添加到源文档列表
            source_documents += web_search_results

        # 对源文档进行去重
        source_documents = deduplicate_documents(source_documents)

        # 可选重排：过滤 score < 0.28，并相对最高分落差超过 50% 的后续文档丢弃
        if rerank and len(source_documents) > 1 and num_tokens_rerank(query) <= 300:
            try:
                # 记录重排开始时间
                t1 = time.perf_counter()
                debug_logger.info(f"use rerank, rerank docs num: {len(source_documents)}")

                # 异步调用重排模型，对文档进行重排序
                source_documents = await self.rerank.arerank_documents(condense_question, source_documents)

                # 记录重排结束时间并计算耗时
                t2 = time.perf_counter()
                time_record['rerank'] = round(t2 - t1, 2)

                # 过滤掉低分的文档
                debug_logger.info(f"rerank step1 num: {len(source_documents)}")
                debug_logger.info(f"rerank step1 scores: {[doc.metadata['score'] for doc in source_documents]}")

                if len(source_documents) > 1:
                    # 过滤分数 >= 0.28 的文档
                    if filtered_documents := [doc for doc in source_documents if doc.metadata['score'] >= 0.28]:
                        source_documents = filtered_documents
                    debug_logger.info(f"rerank step2 num: {len(source_documents)}")

                    # 保留与最高分差距不超过 50% 的文档
                    saved_docs = [source_documents[0]]  # 保留最高分文档
                    for doc in source_documents[1:]:
                        debug_logger.info(f"rerank doc score: {doc.metadata['score']}")
                        # 计算与最高分的相对差距
                        relative_difference = (saved_docs[0].metadata['score'] - doc.metadata['score']) / saved_docs[0].metadata['score']
                        if relative_difference > 0.5:
                            # 差距超过 50%，停止保留后续文档
                            break
                        else:
                            # 差距在 50% 以内，保留该文档
                            saved_docs.append(doc)
                    source_documents = saved_docs
                    debug_logger.info(f"rerank step3 num: {len(source_documents)}")
            except Exception as e:
                # 重排出错时，记录错误并设置重排耗时为 0
                time_record['rerank'] = 0.0
                debug_logger.error(f"query {query}: kb_ids: {kb_ids}, rerank error: {traceback.format_exc()}")

        # 限制文档数量为 top_k
        source_documents = source_documents[:top_k]

        # 去掉 [headers](...) 行，只保留正文，便于放入 prompt
        for doc in source_documents:
            doc.page_content = re.sub(r'^\[headers]\(.*?\)\n', '', doc.page_content)

        # 若存在高分 FAQ（≥0.9），则仅用 FAQ 作为来源，忽略其他文档
        high_score_faq_documents = [doc for doc in source_documents if
                                    doc.metadata['file_name'].endswith('.faq') and doc.metadata['score'] >= 0.9]
        if high_score_faq_documents:
            source_documents = high_score_faq_documents

        # FAQ 完全匹配：问题与某条 FAQ 的 question 一致时直接返回该 answer，不再调用 LLM
        for doc in source_documents:
            if doc.metadata['file_name'].endswith('.faq') and clear_string_is_equal(
                    doc.metadata['faq_dict']['question'], query):
                debug_logger.info(f"match faq question: {query}")

                # 仅需要搜索结果时，直接返回文档
                if only_need_search_results:
                    yield source_documents, None
                    return

                # 直接使用 FAQ 的答案作为回复
                res = doc.metadata['faq_dict']['answer']
                async for response, history in self.generate_response(query, res, condense_question, source_documents,
                                                                      time_record, chat_history, streaming, 'MATCH_FAQ'):
                    yield response, history
                return

        # 获取今日日期和当前时间，用于 prompt 中的时间信息
        today = time.strftime("%Y-%m-%d", time.localtime())
        now = time.strftime("%H:%M:%S", time.localtime())

        # 初始化额外消息、图片数量和检索文档列表
        extra_msg = None
        total_images_number = 0
        retrieval_documents = []
        # 处理源文档
        if source_documents:
            # 选择 prompt 模板：自定义或默认（含今日日期与当前时间）
            if custom_prompt:
                # 使用自定义 prompt 模板
                prompt_template = CUSTOM_PROMPT_TEMPLATE.replace("{{custom_prompt}}", custom_prompt)
            else:
                # 使用默认 prompt 模板，替换时间信息
                system_prompt = SYSTEM.replace("{{today_date}}", today).replace("{{current_time}}", now)
                prompt_template = PROMPT_TEMPLATE.replace("{{system}}", system_prompt).replace("{{instructions}}",
                                                                                               INSTRUCTIONS)

            # 记录文档处理开始时间
            t1 = time.perf_counter()

            # 重新处理源文档，优化 token 使用
            retrieval_documents, limited_token_nums, tokens_msg = self.reprocess_source_documents(custom_llm=custom_llm,
                                                                                                  query=query,
                                                                                                  source_docs=source_documents,
                                                                                                  history=chat_history,
                                                                                                  prompt_template=prompt_template)

            # 检查是否有文档被裁切
            if len(retrieval_documents) < len(source_documents):
                # token 不足导致部分文档被裁切
                if len(retrieval_documents) == 0:
                    # 没有文档可用，返回错误信息
                    debug_logger.error(f"limited_token_nums: {limited_token_nums} < {web_chunk_size}!")
                    res = (
                        f"抱歉，由于留给相关文档使用的token数量不足(docs_available_token_nums: {limited_token_nums} < 文本分片大小: {web_chunk_size})，"
                        f"\n无法保证回答质量，请在模型配置中提高【总Token数量】或减少【输出Tokens数量】或减少【上下文消息数量】再继续提问。"
                        f"\n计算方式：{tokens_msg}")
                    async for response, history in self.generate_response(query, res, condense_question, source_documents,
                                                                          time_record, chat_history, streaming,
                                                                          'TOKENS_NOT_ENOUGH'):
                        yield response, history
                    return

                # 部分文档被裁切，添加警告信息
                extra_msg = (
                    f"\n\nWARNING: 由于留给相关文档使用的token数量不足(docs_available_token_nums: {limited_token_nums})，"
                    f"\n检索到的部分文档chunk被裁切，原始来源数量：{len(source_documents)}，裁切后数量：{len(retrieval_documents)}，"
                    f"\n可能会影响回答质量，尤其是问题涉及的相关内容较多时。"
                    f"\n可在模型配置中提高【总Token数量】或减少【输出Tokens数量】或减少【上下文消息数量】再继续提问。\n")

            # 准备源文档，可能进行文档聚合
            source_documents, retrieval_documents = await self.prepare_source_documents(custom_llm,
                                                                                        retrieval_documents,
                                                                                        limited_token_nums,
                                                                                        rerank)

            # 统计图片数量并处理图片引用
            for doc in source_documents:
                if doc.metadata.get('images', []):
                    total_images_number += len(doc.metadata['images'])
                    # 替换文档内容中的图片引用
                    doc.page_content = replace_image_references(doc.page_content, doc.metadata['file_id'])
            debug_logger.info(f"total_images_number: {total_images_number}")

            # 记录文档处理结束时间并计算耗时
            t2 = time.perf_counter()
            time_record['reprocess'] = round(t2 - t1, 2)
        else:
            # 无源文档时，使用简单 prompt 模板
            if custom_prompt:
                # 使用自定义简单 prompt 模板
                prompt_template = SIMPLE_PROMPT_TEMPLATE.replace("{{today}}", today).replace("{{now}}", now).replace(
                    "{{custom_prompt}}", custom_prompt)
            else:
                # 使用默认简单 prompt 模板
                simple_custom_prompt = """
                - If you cannot answer based on the given information, you will return the sentence "抱歉，已知的信息不足，因此无法回答。".
                """
                prompt_template = SIMPLE_PROMPT_TEMPLATE.replace("{{today}}", today).replace("{{now}}", now).replace(
                    "{{custom_prompt}}", simple_custom_prompt)



        # 仅需要搜索结果时，直接返回文档
        if only_need_search_results:
            yield source_documents, None
            return

        # 记录 LLM 调用开始时间
        t1 = time.perf_counter()
        has_first_return = False  # 标记是否已返回第一个结果
        acc_resp = ''  # 累积的 LLM 回复

        # 生成最终 prompt
        prompt = self.generate_prompt(query=query,
                                      source_docs=source_documents,
                                      prompt_template=prompt_template)

        # 估计 prompt 的 token 数
        est_prompt_tokens = num_tokens(prompt) + num_tokens(str(chat_history))
        # 异步调用 LLM，获取流式回答
        # 这里使用 async for 迭代 LLM 的流式输出，每次迭代获取一个回答片段
        async for answer_result in custom_llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
            # 获取 LLM 的输出内容
            resp = answer_result.llm_output["answer"]  # LLM 的输出

            # 累积回复内容
            # 当输出中包含 'answer' 字段时，将其添加到累积回复中
            if 'answer' in resp:
                # resp[6:] 是为了去除前缀 'data: '
                acc_resp += json.loads(resp[6:])['answer']

            # 获取 LLM 返回的信息
            prompt = answer_result.prompt  # 获取实际使用的 prompt
            history = answer_result.history  # 获取更新后的对话历史
            total_tokens = answer_result.total_tokens  # 总 token 数
            prompt_tokens = answer_result.prompt_tokens  # prompt 的 token 数
            completion_tokens = answer_result.completion_tokens  # 完成部分的 token 数

            # 更新对话历史中的最后一个用户问题
            # 确保对话历史中的问题与用户输入的原始问题一致
            history[-1][0] = query

            # 构建响应字典
            response = {"query": query,  # 用户的原始问题
                        "prompt": prompt,  # 发送给 LLM 的 prompt
                        "result": resp,  # LLM 的当前输出
                        "condense_question": condense_question,  # 重写后的问题
                        "retrieval_documents": retrieval_documents,  # 检索到的原始文档
                        "source_documents": source_documents}  # 处理后的源文档

            # 记录 token 使用情况
            # 如果 LLM 没有返回 token 数，则使用估计值
            time_record['prompt_tokens'] = prompt_tokens if prompt_tokens != 0 else est_prompt_tokens
            time_record['completion_tokens'] = completion_tokens if completion_tokens != 0 else num_tokens(acc_resp)
            time_record['total_tokens'] = total_tokens if total_tokens != 0 else time_record['prompt_tokens'] + \
                                                                                 time_record['completion_tokens']

            # 记录首次返回时间
            # 标记 LLM 首次返回响应的时间
            if has_first_return is False:
                first_return_time = time.perf_counter()
                has_first_return = True
                time_record['llm_first_return'] = round(first_return_time - t1, 2)

            # 处理流式结束标记
            # 当 LLM 输出 [DONE] 时，表示流式输出结束
            if resp[6:].startswith("[DONE]"):
                # 添加额外消息（如果有）
                # 例如，当需要向用户添加额外提示或说明时
                if extra_msg is not None:
                    msg_response = {"query": query,
                                "prompt": prompt,
                                "result": f"data: {json.dumps({'answer': extra_msg}, ensure_ascii=False)}",
                                "condense_question": condense_question,
                                "retrieval_documents": retrieval_documents,
                                "source_documents": source_documents}
                    # 生成额外消息的响应
                    yield msg_response, history

                # 记录 LLM 完成时间
                # 计算 LLM 从首次返回开始到完成的时间
                last_return_time = time.perf_counter()
                time_record['llm_completed'] = round(last_return_time - t1, 2) - time_record['llm_first_return']

                # 更新对话历史中的最后一个助手回复
                # 将累积的回复内容更新到对话历史中，用于多轮对话
                history[-1][1] = acc_resp

                # 有图片时：根据 LLM 回答与引用文档相似度选出最相关文档，并收集其图片作为 show_images
                if total_images_number != 0:
                    # 筛选包含图片的文档
                    docs_with_images = [doc for doc in source_documents if doc.metadata.get('images', [])]
                    time1 = time.perf_counter()

                    # 计算文档与问题和 LLM 回答的相关性
                    # 选择与问题和回答最相关的文档，用于展示图片
                    relevant_docs = await self.calculate_relevance_optimized(
                        question=query,  # 用户原始问题
                        llm_answer=acc_resp,  # LLM 的回答
                        reference_docs=docs_with_images,  # 包含图片的文档
                        top_k=1  # 只选择最相关的1个文档
                    )

                    # 构建图片展示内容
                    show_images = ["\n### 引用图文如下：\n"]
                    for doc in relevant_docs:
                        # 打印文档相关性信息（调试用）
                        print(f"文档: {doc['document']}...")  # 只打印前50个字符
                        print(f"最相关段落: {doc['segment']}...")  # 打印最相关段落的前100个字符
                        print(f"与LLM回答的相似度: {doc['similarity_llm']:.4f}")
                        print(f"原始问题相关性分数: {doc['question_score']:.4f}")
                        print(f"综合得分: {doc['combined_score']:.4f}")
                        print()

                        # 处理文档中的图片
                        # 将图片引用转换为前端可访问的格式
                        for image in doc['document'].metadata.get('images', []):
                            image_str = replace_image_references(image, doc['document'].metadata['file_id'])
                            debug_logger.info(f"image_str: {image} -> {image_str}")
                            show_images.append(image_str + '\n')

                    debug_logger.info(f"show_images: {show_images}")
                    # 记录获取图片的耗时
                    time_record['obtain_images'] = round(time.perf_counter() - last_return_time, 2)
                    time2 = time.perf_counter()
                    debug_logger.info(f"obtain_images time: {time2 - time1}s")
                    time_record["obtain_images_time"] = round(time2 - time1, 2)

                    # 添加图片到响应中
                    # 只有当有图片时才添加到响应
                    if len(show_images) > 1:
                        response['show_images'] = show_images

            # 生成响应
            # 使用 yield 返回响应和更新后的对话历史，支持流式输出
            yield response, history

    def get_completed_document(self, file_id, limit=None):
        """
        按 file_id 从 MySQL 取出该文件下所有文档分块，按顺序拼接成完整文档（含图版与去图版）。
        若传 limit=[start, end]，则只取该下标范围内的分块。FAQ 会格式化为「问题：答案」。

        Args:
            file_id: 文件 ID。
            limit: 可选 [start, end]，只取 sorted_json_datas[start:end+1]。

        Returns:
            (completed_doc, completed_doc_with_figure)：去图版与含图版 Document，metadata 含 has_table、images。
        """
        # 从 MySQL 获取指定文件的所有文档分块，按 doc_id 排序
        sorted_json_datas = self.milvus_summary.get_document_by_file_id(file_id)

        # 如果指定了 limit，则只取该范围内的分块
        if limit:
            sorted_json_datas = sorted_json_datas[limit[0]: limit[1] + 1]

        # 初始化内容变量：含图版和去图版
        completed_content_with_figure = ''
        completed_content = ''

        # 遍历所有文档分块
        for doc_json in sorted_json_datas:
            # 创建 Document 对象
            doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])

            # 删除文档头部信息，只保留文本内容
            doc.page_content = re.sub(r'^\[headers]\(.*?\)\n', '', doc.page_content)

            # 处理 FAQ 文档，格式化为「问题：答案」
            if doc_json['kwargs']['metadata']['file_name'].endswith('.faq'):
                faq_dict = doc_json['kwargs']['metadata']['faq_dict']
                doc.page_content = f"{faq_dict['question']}：{faq_dict['answer']}"

            # 添加到含图版内容
            completed_content_with_figure += doc.page_content + '\n\n'
            # 添加到去图版内容（删除图片占位符）
            completed_content += re.sub(r'!\[figure]\(.*?\)', '', doc.page_content) + '\n\n'

        # 创建含图版 Document 对象
        completed_doc_with_figure = Document(page_content=completed_content_with_figure, metadata=sorted_json_datas[0]['kwargs']['metadata'])
        # 创建去图版 Document 对象
        completed_doc = Document(page_content=completed_content, metadata=sorted_json_datas[0]['kwargs']['metadata'])

        # 初始化表格和图片标记
        has_table = False
        images = []

        # 检查文档分块是否包含表格和图片
        for doc_json in sorted_json_datas:
            # 检查是否包含表格
            if doc_json['kwargs']['metadata'].get('has_table'):
                has_table = True
                break  # 只要有一个分块包含表格，整个文档就标记为包含表格

            # 收集图片信息
            if doc_json['kwargs']['metadata'].get('images'):
                images.extend(doc_json['kwargs']['metadata']['images'])

        # 设置文档 metadata
        completed_doc.metadata['has_table'] = has_table
        completed_doc.metadata['images'] = images
        completed_doc_with_figure.metadata['has_table'] = has_table
        completed_doc_with_figure.metadata['images'] = images

        # 返回去图版和含图版文档
        return completed_doc, completed_doc_with_figure

    def aggregate_documents(self, source_documents, limited_token_nums, custom_llm, rerank):
        """
        在候选文档仅来自一或两个文件时，尽量返回完整文件内容（或按 doc_id 范围截取），以保留上下文完整性。
        若候选来自超过两个文件则返回 []，由调用方退化为按原 source_documents 使用。

        Args:
            source_documents: 检索得到的文档列表（已按 file_id、doc_id 有序）。
            limited_token_nums: 文档总 token 上限。
            custom_llm: 用于统计 token。
            rerank: 为 True 时取同文件内 max(score)，否则取 min(score)。

        Returns:
            聚合后的 Document 列表（含图版）；无法聚合时返回 []。
        """
        # 初始化变量：存储第一个文件的信息
        first_file_dict = {}
        # 存储第一个文件的原始文档列表
        ori_first_docs = []
        # 初始化变量：存储第二个文件的信息
        second_file_dict = {}
        # 存储第二个文件的原始文档列表
        ori_second_docs = []

        # 遍历所有检索到的文档
        for doc in source_documents:
            # 获取当前文档的文件ID
            file_id = doc.metadata['file_id']

            # 如果第一个文件信息为空，说明这是第一个文件
            if not first_file_dict:
                # 记录第一个文件的ID
                first_file_dict['file_id'] = file_id
                # 记录第一个文件的文档ID列表（提取数字部分）
                first_file_dict['doc_ids'] = [int(doc.metadata['doc_id'].split('_')[-1])]
                # 添加当前文档到第一个文件的原始文档列表
                ori_first_docs.append(doc)
                # 根据是否重排，计算第一个文件的分数
                if rerank:
                    # 重排时取同文件内的最高分数
                    first_file_dict['score'] = max(
                        [doc.metadata['score'] for doc in source_documents if doc.metadata['file_id'] == file_id])
                else:
                    # 非重排时取同文件内的最低分数
                    first_file_dict['score'] = min(
                        [doc.metadata['score'] for doc in source_documents if doc.metadata['file_id'] == file_id])

            # 如果当前文件ID与第一个文件ID相同
            elif first_file_dict['file_id'] == file_id:
                # 添加当前文档的ID到第一个文件的文档ID列表
                first_file_dict['doc_ids'].append(int(doc.metadata['doc_id'].split('_')[-1]))
                # 添加当前文档到第一个文件的原始文档列表
                ori_first_docs.append(doc)

            # 如果第二个文件信息为空，说明这是第二个文件
            elif not second_file_dict:
                # 记录第二个文件的ID
                second_file_dict['file_id'] = file_id
                # 记录第二个文件的文档ID列表（提取数字部分）
                second_file_dict['doc_ids'] = [int(doc.metadata['doc_id'].split('_')[-1])]
                # 添加当前文档到第二个文件的原始文档列表
                ori_second_docs.append(doc)
                # 根据是否重排，计算第二个文件的分数
                if rerank:
                    # 重排时取同文件内的最高分数
                    second_file_dict['score'] = max(
                        [doc.metadata['score'] for doc in source_documents if doc.metadata['file_id'] == file_id])
                else:
                    # 非重排时取同文件内的最低分数
                    second_file_dict['score'] = min(
                        [doc.metadata['score'] for doc in source_documents if doc.metadata['file_id'] == file_id])

            # 如果当前文件ID与第二个文件ID相同
            elif second_file_dict['file_id'] == file_id:
                # 添加当前文档的ID到第二个文件的文档ID列表
                second_file_dict['doc_ids'].append(int(doc.metadata['doc_id'].split('_')[-1]))
                # 添加当前文档到第二个文件的原始文档列表
                ori_second_docs.append(doc)

            # 如果遇到第三个文件，直接返回空列表（无法聚合）
            else:
                return []

        # 计算第一个文件原始文档的token数
        ori_first_docs_tokens = custom_llm.num_tokens_from_docs(ori_first_docs)
        # 计算第二个文件原始文档的token数
        ori_second_docs_tokens = custom_llm.num_tokens_from_docs(ori_second_docs)

        # 初始化聚合后的文档列表
        new_docs = []

        # 获取第一个文件的完整文档（含图版和去图版）
        first_completed_doc, first_completed_doc_with_figure = self.get_completed_document(first_file_dict['file_id'])
        # 设置第一个文件的分数
        first_completed_doc.metadata['score'] = first_file_dict['score']
        # 计算第一个文件完整文档的token数
        first_doc_tokens = custom_llm.num_tokens_from_docs([first_completed_doc])

        # 检查第一个文件完整文档加上第二个文件原始文档的token数是否超过限制
        if first_doc_tokens + ori_second_docs_tokens > limited_token_nums:
            # 如果第一个文件只有一个文档，直接返回空列表
            if len(ori_first_docs) == 1:
                debug_logger.info(f"first_file_docs number is one")
                return new_docs

            # 尝试缩小文档范围以减少token数
            doc_limit = [min(first_file_dict['doc_ids']), max(first_file_dict['doc_ids'])]
            while True:
                # 获取第一个文件在指定ID范围内的文档
                first_completed_doc_limit, first_completed_doc_limit_with_figure = self.get_completed_document(
                    first_file_dict['file_id'], doc_limit)
                # 设置第一个文件的分数
                first_completed_doc_limit.metadata['score'] = first_file_dict['score']
                # 计算第一个文件截取后文档的token数
                first_doc_tokens = custom_llm.num_tokens_from_docs([first_completed_doc_limit])

                # 检查token数是否超过限制
                if first_doc_tokens + ori_second_docs_tokens <= limited_token_nums:
                    # 如果未超过限制，添加第一个文件截取后的含图版文档
                    debug_logger.info(
                        f"first_limit_doc_tokens {doc_limit}: {first_doc_tokens} + ori_second_docs_tokens: {ori_second_docs_tokens} <= limited_token_nums: {limited_token_nums}")
                    new_docs.append(first_completed_doc_limit_with_figure)
                    break
                else:
                    # 如果仍超过限制，尝试缩小范围
                    if doc_limit[0] >= doc_limit[1]:
                        # 无法再缩小，返回空列表
                        debug_logger.info(
                            f"Cannot reduce doc_limit further: {doc_limit}")
                        return new_docs
                    # 缩小范围，取前半部分
                    new_max = (doc_limit[0] + doc_limit[1]) // 2
                    debug_logger.info(
                        f"Reducing doc_limit from {doc_limit} to [{doc_limit[0]}, {new_max}]")
                    doc_limit = [doc_limit[0], new_max]
        else:
            # 如果未超过限制，添加第一个文件的完整含图版文档
            debug_logger.info(
                f"first_doc_tokens: {first_doc_tokens} + ori_second_docs_tokens: {ori_second_docs_tokens} <= limited_token_nums: {limited_token_nums}")
            new_docs.append(first_completed_doc_with_figure)

        # 如果存在第二个文件
        if second_file_dict:
            # 获取第二个文件的完整文档（含图版和去图版）
            second_completed_doc, second_completed_doc_with_figure = self.get_completed_document(second_file_dict['file_id'])
            # 设置第二个文件的分数
            second_completed_doc.metadata['score'] = second_file_dict['score']
            # 计算第二个文件完整文档的token数
            second_doc_tokens = custom_llm.num_tokens_from_docs([second_completed_doc])

            # 检查第一个文件文档加上第二个文件完整文档的token数是否超过限制
            if first_doc_tokens + second_doc_tokens > limited_token_nums:
                # 如果第二个文件只有一个文档，直接添加原始文档并返回
                if len(ori_second_docs) == 1:
                    debug_logger.info(f"second_file_docs number is one")
                    new_docs.extend(ori_second_docs)
                    return new_docs

                # 尝试缩小文档范围以减少token数
                doc_limit = [min(second_file_dict['doc_ids']), max(second_file_dict['doc_ids'])]
                while True:
                    # 获取第二个文件在指定ID范围内的文档
                    second_completed_doc_limit, second_completed_doc_limit_with_figure = self.get_completed_document(
                        second_file_dict['file_id'], doc_limit)
                    # 设置第二个文件的分数
                    second_completed_doc_limit.metadata['score'] = second_file_dict['score']
                    # 计算第二个文件截取后文档的token数
                    second_doc_tokens = custom_llm.num_tokens_from_docs([second_completed_doc_limit])

                    # 检查token数是否超过限制
                    if first_doc_tokens + second_doc_tokens <= limited_token_nums:
                        # 如果未超过限制，添加第二个文件截取后的含图版文档
                        debug_logger.info(
                            f"first_doc_tokens: {first_doc_tokens} + second_limit_doc_tokens {doc_limit}: {second_doc_tokens} <= limited_token_nums: {limited_token_nums}")
                        new_docs.append(second_completed_doc_limit_with_figure)
                        break
                    else:
                        # 如果仍超过限制，尝试缩小范围
                        if doc_limit[0] >= doc_limit[1]:
                            # 无法再缩小，添加原始文档并返回
                            debug_logger.info(
                                f"Cannot reduce doc_limit further: {doc_limit}")
                            new_docs.extend(ori_second_docs)
                            return new_docs
                        # 缩小范围，取前半部分
                        new_max = (doc_limit[0] + doc_limit[1]) // 2
                        debug_logger.info(
                            f"Reducing doc_limit from {doc_limit} to [{doc_limit[0]}, {new_max}]")
                        doc_limit = [doc_limit[0], new_max]
            else:
                # 如果未超过限制，添加第二个文件的完整含图版文档
                debug_logger.info(
                    f"first_doc_tokens: {first_doc_tokens} + second_doc_tokens: {second_doc_tokens} <= limited_token_nums: {limited_token_nums}")
                new_docs.append(second_completed_doc_with_figure)

        # 返回聚合后的文档列表
        return new_docs

    def incomplete_table(self, source_documents, limited_token_nums, custom_llm):
        """
        若某条 doc 属于表格的一部分（有 table_doc_id），则尝试用完整表格文档替换该段，以保持表格完整。
        若替换后总 token 超限则保留原 doc，并将该 table_doc_id 记为“已验证放不下”，避免重复尝试。

        Args:
            source_documents: 当前候选文档列表。
            limited_token_nums: token 上限。
            custom_llm: 用于 token 统计。

        Returns:
            替换部分表格片段为完整表格后的新文档列表。
        """
        existing_table_docs = [doc for doc in source_documents if doc.metadata.get('has_table', False)]
        if not existing_table_docs:
            return source_documents
        new_docs = []
        existing_table_ids = []
        verified_table_ids = []
        current_doc_tokens = custom_llm.num_tokens_from_docs(source_documents)
        for doc in source_documents:
            if 'doc_id' not in doc.metadata:
                new_docs.append(doc)
                continue
            if table_doc_id := doc.metadata.get('table_doc_id', None):
                if table_doc_id in existing_table_ids:  # 已经不全了完整表格
                    continue
                if table_doc_id in verified_table_ids:  # 已经确认了完整表格太大放不大
                    new_docs.append(doc)
                    continue
                doc_json = self.milvus_summary.get_document_by_doc_id(table_doc_id)
                if doc_json is None:
                    new_docs.append(doc)
                    continue
                table_doc = Document(page_content=doc_json['kwargs']['page_content'],
                                     metadata=doc_json['kwargs']['metadata'])
                table_doc.metadata['score'] = doc.metadata['score']
                table_doc_tokens = custom_llm.num_tokens_from_docs([table_doc])
                current_table_docs = [doc for doc in source_documents if
                                      doc.metadata.get('table_doc_id', None) == table_doc_id]
                subtract_table_doc_tokens = custom_llm.num_tokens_from_docs(current_table_docs)
                if current_doc_tokens + table_doc_tokens - subtract_table_doc_tokens > limited_token_nums:
                    debug_logger.info(
                        f"Add table_doc_tokens: {table_doc_tokens} > limited_token_nums: {limited_token_nums}")
                    new_docs.append(doc)
                    verified_table_ids.append(table_doc_id)
                    continue
                else:
                    debug_logger.info(f"Incomplete table_doc: {table_doc_id}")
                    new_docs.append(table_doc)
                    existing_table_ids.append(table_doc_id)
                    current_doc_tokens = current_doc_tokens + table_doc_tokens - subtract_table_doc_tokens
        return new_docs
