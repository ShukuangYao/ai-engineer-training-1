from langchain.retrievers import ParentDocumentRetriever
from qanything_kernel.core.retriever.vectorstore import VectorStoreMilvusClient
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.core.retriever.docstrore import MysqlStore
from qanything_kernel.configs.model_config import DEFAULT_CHILD_CHUNK_SIZE, DEFAULT_PARENT_CHUNK_SIZE, SEPARATORS
from qanything_kernel.utils.custom_log import debug_logger, insert_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qanything_kernel.utils.general_utils import num_tokens_embed, get_time_async
import copy
from typing import List, Optional, Tuple, Dict
from langchain_core.documents import Document
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
)
from langchain_community.vectorstores.milvus import Milvus
from langchain_elasticsearch import ElasticsearchStore
import time
import traceback


class SelfParentRetriever(ParentDocumentRetriever):
    def set_search_kwargs(self, search_type, **kwargs):
        self.search_type = search_type
        self.search_kwargs = kwargs
        debug_logger.info(f"Set search kwargs: {self.search_kwargs}")

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        # 记录搜索查询、搜索类型和搜索参数的日志
        debug_logger.info(f"Search: query: {query}, {self.search_type} with {self.search_kwargs}")

        # 初始化分数列表
        scores = []

        # 根据搜索类型执行不同的搜索策略
        if self.search_type == "mmr":
            # 使用最大边际相关性搜索（Max Marginal Relevance）
            # 这种搜索方式平衡了相关性和多样性
            sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            # 使用相似度搜索并获取相关分数
            res = await self.vectorstore.asimilarity_search_with_score(
                query, **self.search_kwargs
            )
            # 提取分数和文档
            scores = [score for _, score in res]
            sub_docs = [doc for doc, _ in res]

        # 从子文档中提取唯一的父文档 ID，保持返回顺序
        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            # 检查文档元数据中是否包含 id_key，并且该 ID 尚未添加
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])

        # 使用 docstore 异步获取父文档
        docs = await self.docstore.amget(ids)

        # 如果有分数信息，将分数添加到父文档的元数据中
        if scores:
            for i, doc in enumerate(docs):
                if doc is not None:
                    doc.metadata['score'] = scores[i]

        # 过滤掉 None 文档
        res = [d for d in docs if d is not None]

        # 计算子文档和父文档的内容长度，用于日志记录
        sub_docs_lengths = [len(d.page_content) for d in sub_docs]
        res_lengths = [len(d.page_content) for d in res]

        # 记录子文档和父文档的数量和长度信息
        debug_logger.info(
            f"Got child docs: {len(sub_docs)}, {sub_docs_lengths} and Parent docs: {len(res)}, {res_lengths}")

        # 返回相关文档列表
        return res

    async def aadd_documents(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            add_to_docstore: bool = True,
            parent_chunk_size: Optional[int] = None,
            es_store: Optional[ElasticsearchStore] = None,
            single_parent: bool = False,
    ) -> Tuple[int, Dict]:
        # insert_logger.info(f"Inserting {len(documents)} complete documents, single_parent: {single_parent}")
        split_start = time.perf_counter()
        if self.parent_splitter is not None and not single_parent:
            # documents = self.parent_splitter.split_documents(documents)
            split_documents = []
            need_split_docs = []
            for doc in documents:
                if doc.metadata['has_table'] or num_tokens_embed(doc.page_content) <= parent_chunk_size:
                    if need_split_docs:
                        split_documents.extend(self.parent_splitter.split_documents(need_split_docs))
                        need_split_docs = []
                    split_documents.append(doc)
                else:
                    need_split_docs.append(doc)
            if need_split_docs:
                split_documents.extend(self.parent_splitter.split_documents(need_split_docs))
            documents = split_documents
        insert_logger.info(f"Inserting {len(documents)} parent documents")
        if ids is None:
            file_id = documents[0].metadata['file_id']
            doc_ids = [file_id + '_' + str(i) for i, _ in enumerate(documents)]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
                _doc.page_content = f"[headers]({_doc.metadata['headers']})\n" + _doc.page_content  # 存入page_content，向量检索时会带上headers
            docs.extend(sub_docs)
            doc.page_content = f"[headers]({doc.metadata['headers']})\n" + doc.page_content  # 存入page_content，等检索后rerank时会带上headers信息
            full_docs.append((_id, doc))
        insert_logger.info(f"Inserting {len(docs)} child documents, metadata: {docs[0].metadata}, page_content: {docs[0].page_content[:100]}...")
        time_record = {"split_time": round(time.perf_counter() - split_start, 2)}

        embed_docs = copy.deepcopy(docs)
        # 补充metadata信息
        for idx, doc in enumerate(embed_docs):
            del doc.metadata['title_lst']
            del doc.metadata['has_table']
            del doc.metadata['images']
            del doc.metadata['file_name']
            del doc.metadata['nos_key']
            del doc.metadata['faq_dict']
            del doc.metadata['page_id']

        res = await self.vectorstore.aadd_documents(embed_docs, time_record=time_record)
        insert_logger.info(f'vectorstore insert number: {len(res)}, {res[0]}')
        if es_store is not None:
            try:
                es_start = time.perf_counter()
                # docs的doc_id是file_id + '_' + i
                docs_ids = [doc.metadata['file_id'] + '_' + str(i) for i, doc in enumerate(embed_docs)]
                es_res = await es_store.aadd_documents(embed_docs, ids=docs_ids)
                time_record['es_insert_time'] = round(time.perf_counter() - es_start, 2)
                insert_logger.info(f'es_store insert number: {len(es_res)}, {es_res[0]}')
            except Exception as e:
                insert_logger.error(f"Error in aadd_documents on es_store: {traceback.format_exc()}")

        if add_to_docstore:
            await self.docstore.amset(full_docs)
        return len(res), time_record


class ParentRetriever:
    def __init__(self, vectorstore_client: VectorStoreMilvusClient, mysql_client: KnowledgeBaseManager, es_client: StoreElasticSearchClient):
        self.mysql_client = mysql_client
        self.vectorstore_client = vectorstore_client
        # This text splitter is used to create the parent documents
        init_parent_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=DEFAULT_PARENT_CHUNK_SIZE,
            chunk_overlap=0,
            length_function=num_tokens_embed)
        # # This text splitter is used to create the child documents
        # # It should create documents smaller than the parent
        init_child_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=DEFAULT_CHILD_CHUNK_SIZE,
            chunk_overlap=int(DEFAULT_CHILD_CHUNK_SIZE / 4),
            length_function=num_tokens_embed)
        self.retriever = SelfParentRetriever(
            vectorstore=vectorstore_client.local_vectorstore,
            docstore=MysqlStore(mysql_client),
            child_splitter=init_child_splitter,
            parent_splitter=init_parent_splitter,
        )
        self.backup_vectorstore: Optional[Milvus] = None
        self.es_store = es_client.es_store
        self.parent_chunk_size = DEFAULT_PARENT_CHUNK_SIZE

    @get_time_async
    async def insert_documents(self, docs, parent_chunk_size, single_parent=False):
        insert_logger.info(f"Inserting {len(docs)} documents, parent_chunk_size: {parent_chunk_size}, single_parent: {single_parent}")
        if parent_chunk_size != self.parent_chunk_size:
            self.parent_chunk_size = parent_chunk_size
            parent_splitter = RecursiveCharacterTextSplitter(
                separators=SEPARATORS,
                chunk_size=parent_chunk_size,
                chunk_overlap=0,
                length_function=num_tokens_embed)
            child_chunk_size = min(DEFAULT_CHILD_CHUNK_SIZE, int(parent_chunk_size / 2))
            child_splitter = RecursiveCharacterTextSplitter(
                separators=SEPARATORS,
                chunk_size=child_chunk_size,
                chunk_overlap=int(child_chunk_size / 4),
                length_function=num_tokens_embed)
            self.retriever = SelfParentRetriever(
                vectorstore=self.vectorstore_client.local_vectorstore,
                docstore=MysqlStore(self.mysql_client),
                child_splitter=child_splitter,
                parent_splitter=parent_splitter
            )
        # insert_logger.info(f'insert documents: {len(docs)}')
        ids = None if not single_parent else [doc.metadata['doc_id'] for doc in docs]
        return await self.retriever.aadd_documents(docs, parent_chunk_size=parent_chunk_size,
                                                   es_store=self.es_store, ids=ids, single_parent=single_parent)

    async def get_retrieved_documents(self, query: str, partition_keys: List[str], time_record: dict,
                                      hybrid_search: bool, top_k: int):
        """
        混合检索的核心实现：结合向量检索和全文检索

        【设计原理】
        1. 向量检索：理解语义相似性，擅长处理同义词、概念匹配
        2. 全文检索：精确关键词匹配，擅长处理专有名词、数字、代码
        3. 混合检索：两者结合，既有语义理解又有精确匹配

        【为什么这样设计】
        - 单一检索方式都有局限性
        - 向量检索可能错过精确匹配的重要文档
        - 全文检索可能错过语义相关但用词不同的文档
        - 混合检索提供更全面的召回率
        """

        # ========== 第一阶段：向量检索 (Milvus) ==========
        """
        使用向量数据库进行语义检索
        - 将查询转换为向量表示
        - 在高维空间中寻找最相似的文档向量
        - 擅长理解语义和概念层面的相似性
        """
        milvus_start_time = time.perf_counter()

        # 构建过滤表达式：只在指定的知识库中搜索
        expr = f'kb_id in {partition_keys}'

        # 设置搜索参数
        # 注释掉的MMR(Maximal Marginal Relevance)算法可以增加结果多样性，但这里使用简单的相似度搜索
        # self.retriever.set_search_kwargs("mmr", k=VECTOR_SEARCH_TOP_K, expr=expr)
        self.retriever.set_search_kwargs("similarity", k=top_k, expr=expr)

        # 执行向量检索
        query_docs = await self.retriever.aget_relevant_documents(query)

        # 标记检索来源，便于后续分析和调试
        for doc in query_docs:
            doc.metadata['retrieval_source'] = 'milvus'

        milvus_end_time = time.perf_counter()
        time_record['retriever_search_by_milvus'] = round(milvus_end_time - milvus_start_time, 2)

        # ========== 混合检索开关判断 ==========
        """
        如果不启用混合检索，直接返回向量检索结果
        这样设计的好处：
        1. 灵活性：可以根据场景选择检索策略
        2. 性能：某些场景下只需要向量检索，节省计算资源
        3. 调试：便于对比不同检索策略的效果
        """
        if not hybrid_search:
            return query_docs

        # ========== 第二阶段：全文检索 (Elasticsearch) ==========
        """
        使用Elasticsearch进行关键词检索
        - 基于倒排索引进行精确匹配
        - 擅长处理专有名词、数字、代码片段等
        - 补充向量检索可能遗漏的精确匹配文档
        """
        try:
            # 构建ES查询过滤器：同样只在指定知识库中搜索
            # 注释掉的代码显示了另一种构建过滤器的方式
            # filter = []
            # for partition_key in partition_keys:
            filter = [{"terms": {"metadata.kb_id.keyword": partition_keys}}]

            # 执行ES检索，获取子文档（chunk级别的文档片段）
            es_sub_docs = await self.es_store.asimilarity_search(query, k=top_k, filter=filter)

            # ========== 去重处理：避免重复文档 ==========
            """
            为什么需要去重？
            1. 向量检索和全文检索可能返回相同的文档
            2. 重复文档会浪费token，降低信息密度
            3. 影响后续的重排序和过滤效果

            去重策略：
            1. 收集已有的Milvus文档ID
            2. 只添加ES独有的文档
            3. 确保每个文档只出现一次
            """
            es_ids = []
            milvus_doc_ids = [d.metadata[self.retriever.id_key] for d in query_docs]

            # 遍历ES检索结果，筛选出不重复的文档ID
            for d in es_sub_docs:
                doc_id = d.metadata.get(self.retriever.id_key)
                if (doc_id and
                    doc_id not in es_ids and           # 避免ES内部重复
                    doc_id not in milvus_doc_ids):     # 避免与Milvus结果重复
                    es_ids.append(doc_id)

            # ========== 获取完整文档内容 ==========
            """
            为什么需要这一步？
            1. ES检索返回的是子文档(chunk)，需要获取完整的父文档
            2. 保持与Milvus检索结果的格式一致性
            3. 确保后续处理流程的统一性
            """
            es_docs = await self.retriever.docstore.amget(es_ids)
            es_docs = [d for d in es_docs if d is not None]  # 过滤掉可能的None值

            # 标记ES检索来源
            for doc in es_docs:
                doc.metadata['retrieval_source'] = 'es'

            # 记录ES检索耗时
            time_record['retriever_search_by_es'] = round(time.perf_counter() - milvus_end_time, 2)

            # 记录检索统计信息，便于监控和调优
            debug_logger.info(f"Got {len(query_docs)} documents from vectorstore and {len(es_sub_docs)} documents from es, total {len(query_docs) + len(es_docs)} merged documents.")

            # ========== 结果合并 ==========
            """
            简单的列表合并策略：
            1. 先放置向量检索结果（通常质量更稳定）
            2. 再添加ES独有的结果（作为补充）
            3. 后续会通过重排序模型重新排序

            为什么不在这里做复杂的分数融合？
            1. 向量相似度分数和ES分数的量纲不同，直接融合意义不大
            2. 重排序模型会重新计算所有文档的相关性分数
            3. 保持代码简洁，职责分离
            """
            query_docs.extend(es_docs)

        except Exception as e:
            """
            容错处理：ES检索失败不应该影响整个检索流程
            1. 记录错误日志，便于问题排查
            2. 继续返回向量检索结果，保证基本功能
            3. 避免因为ES问题导致整个系统不可用
            """
            debug_logger.error(f"Error in get_retrieved_documents on es_search: {e}")

        return query_docs
