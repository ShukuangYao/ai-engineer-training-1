"""
混合RAG系统主入口
包含完整的混合RAG系统实现和演示功能
实现了融合向量检索、关键词检索和图谱推理的多跳问答系统
"""

import asyncio
import json
import logging
import numpy as np
import os
import dashscope
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.analyse

# 导入模型定义
from models import Entity, Relationship, Document, RetrievalResult, GraphResult
from embedding import QwenEmbedding

logging.basicConfig(level=logging.INFO)  # 配置日志级别
logger = logging.getLogger(__name__)  # 初始化日志记录器

# 设置API密钥，优先从环境变量获取，否则使用默认值
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")

class ImprovedHybridRAGSystem:
    """
    改进的混合RAG系统

    主要改进：
    1. 使用通义千问text-embedding-v3提升中文理解
    2. 优化关键词提取，使用jieba分词
    3. 改进相似度计算和阈值设置
    4. 增强图谱推理逻辑
    5. 添加向量数据库内置函数对比测试
    """

    def __init__(self, neo4j_driver, llm_json, llm_text, use_qwen_embedding=True):
        """
        初始化改进的混合RAG系统

        Args:
            neo4j_driver: Neo4j数据库驱动，用于连接图数据库
            llm_json: 用于结构化输出的LLM，用于实体和关系提取
            llm_text: 用于文本生成的LLM，用于生成最终答案
            use_qwen_embedding: 是否使用通义千问嵌入模型，默认为True
        """
        self.driver = neo4j_driver  # 存储Neo4j驱动
        self.llm_json = llm_json  # 结构化输出LLM
        self.llm_text = llm_text  # 文本生成LLM

        # 初始化向量模型
        if use_qwen_embedding:
            # 使用通义千问text-embedding-v3模型
            self.embedding_model = QwenEmbedding("text-embedding-v3")
            logger.info("✅ 使用通义千问text-embedding-v3模型")
        else:
            # 使用本地SentenceTransformer模型作为备选
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ 使用SentenceTransformer模型")

        # 文档存储
        self.documents: List[Document] = []  # 存储文档列表

        # 关键词索引，用于快速关键词检索
        self.keyword_index = {
            'terms': {},  # term -> [doc_indices]，词项到文档索引的映射
            'doc_terms': {}  # doc_id -> [terms]，文档ID到词项的映射
        }

        # 配置参数 - 降低阈值提升召回率
        self.config = {
            'confidence_threshold': 0.3,  # 降低置信度阈值，提升召回率
            'max_retrieval_results': 10,   # 增加检索结果数，获取更多相关信息
            'vector_weight': 0.5,         # 提高向量检索权重
            'keyword_weight': 0.3,        # 关键词检索权重
            'graph_weight': 0.2,          # 图谱推理权重
            'error_propagation_threshold': 0.3,  # 降低错误传播阈值
            'keyword_threshold': 0.1      # 关键词匹配阈值，降低以提升召回率
        }

    def clear_vector_database(self):
        """清理向量数据库，删除所有文档和嵌入"""
        # 清空文档列表
        self.documents = []
        # 重置关键词索引
        self.keyword_index = {
            'terms': {},  # term -> [doc_indices]
            'doc_terms': {}  # doc_id -> [terms]
        }
        logger.info("✅ 向量数据库已清理")

    def clear_graph_database(self):
        """清理图数据库，删除所有节点和关系"""
        with self.driver.session() as session:
            # 执行Cypher查询，删除所有节点和关系
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✅ 图数据库已清理")

    def clear_all_databases(self):
        """清理所有数据库"""
        # 清理向量数据库
        self.clear_vector_database()
        # 清理图数据库
        self.clear_graph_database()
        logger.info("🧹 所有数据库已清理完成")

    def load_data_from_file(self, file_path: str) -> str:
        """从文件加载数据

        Args:
            file_path: 文件路径

        Returns:
            文件内容，读取失败返回空字符串
        """
        try:
            # 打开文件并读取内容
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
            return ""  # 读取失败返回空字符串

    async def process_text_to_documents(self, raw_text: str) -> List[Dict[str, Any]]:
        """将原始文本处理为文档

        Args:
            raw_text: 原始文本

        Returns:
            文档列表，每个文档包含id、content和metadata字段
        """
        # 按段落分割文本
        paragraphs = [p.strip() for p in raw_text.split('\n') if p.strip()]

        documents = []
        for i, paragraph in enumerate(paragraphs):
            # 过滤太短的段落
            if len(paragraph) > 10:
                documents.append({
                    'id': f'doc_{i}',  # 生成文档ID
                    'content': paragraph,  # 文档内容
                    'metadata': {'source': 'file', 'paragraph_id': i}  # 文档元数据
                })

        return documents

    async def extract_relationships_from_text(self, raw_text: str) -> List[Tuple[str, str]]:
        """从文本中提取关系

        Args:
            raw_text: 原始文本

        Returns:
            关系列表，每个关系是(source, target)元组
        """
        # 构建提示词，要求LLM提取公司控股关系
        prompt = f"""
        从以下文本中提取公司控股关系，返回JSON格式：

        文本：{raw_text}

        返回格式：
        {{
            "relationships": [
                {{"source": "公司A", "target": "公司B", "type": "控股"}}
            ]
        }}

        注意：只提取明确的控股关系，包括"控股"、"持股"、"投资"等关系。
        """

        try:
            # 调用LLM获取关系提取结果
            response = await self.llm_json.ainvoke(prompt)
            # 解析JSON结果
            result = json.loads(response.content)

            relationships = []
            # 提取关系列表
            for rel in result.get("relationships", []):
                relationships.append((rel["source"], rel["target"]))

            return relationships
        except Exception as e:
            logger.error(f"关系提取失败: {e}")
            return []  # 提取失败返回空列表

    def extract_keywords(self, text: str) -> List[str]:
        """使用jieba提取关键词

        Args:
            text: 文本

        Returns:
            关键词列表
        """
        # 使用TF-IDF提取关键词，取前10个
        keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)

        # 添加基础分词，获取更多可能的关键词
        words = jieba.cut(text)
        basic_words = [w for w in words if len(w) > 1 and w.isalnum()]

        # 合并并去重，确保关键词唯一性
        all_keywords = list(set(keywords + basic_words))
        return all_keywords

    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档到向量数据库

        Args:
            documents: 文档列表，每个文档包含id、content和metadata字段
        """
        for doc_data in documents:
            # 创建文档对象
            doc = Document(
                id=doc_data['id'],
                content=doc_data['content'],
                metadata=doc_data['metadata']
            )

            # 生成向量嵌入
            doc.embedding = self.embedding_model.encode(doc.content)

            # 提取关键词并建立索引
            keywords = self.extract_keywords(doc.content)
            self.keyword_index['doc_terms'][doc.id] = keywords

            # 更新词项到文档索引的映射
            for keyword in keywords:
                if keyword not in self.keyword_index['terms']:
                    self.keyword_index['terms'][keyword] = []
                # 添加文档索引到词项映射
                self.keyword_index['terms'][keyword].append(len(self.documents))

            # 将文档添加到文档列表
            self.documents.append(doc)

        logger.info(f"添加了 {len(documents)} 个文档到检索库")

    def vector_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        改进的向量检索：使用更好的相似度计算

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认为10

        Returns:
            检索结果列表，按相似度降序排列
        """
        if not self.documents:
            return []  # 无文档时返回空列表

        # 查询向量化：将查询文本转换为向量表示
        query_embedding = self.embedding_model.encode(query)

        # 计算相似度：使用余弦相似度计算查询向量与所有文档向量的相似度
        doc_embeddings = np.array([doc.embedding for doc in self.documents])
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # 使用更低的阈值，提升召回率
        results = []
        for i, score in enumerate(similarities):
            # 只保留相似度高于置信度阈值的结果
            if score > self.config['confidence_threshold']:
                results.append(RetrievalResult(
                    document=self.documents[i],  # 文档对象
                    score=float(score),  # 相似度分数
                    source='vector'  # 检索来源
                ))

        # 按分数降序排序，确保最相关的结果排在前面
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"向量检索: {len(results)} 个结果")
        return results[:top_k]  # 返回前top_k个结果

    def keyword_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        改进的关键词检索：使用jieba分词和更灵活的匹配

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认为10

        Returns:
            检索结果列表，按匹配度降序排列
        """
        if not self.documents:
            return []  # 无文档时返回空列表

        # 提取查询关键词：使用jieba分词提取查询文本中的关键词
        query_keywords = self.extract_keywords(query)

        results = []
        for i, doc in enumerate(self.documents):
            # 获取文档的关键词列表
            doc_keywords = self.keyword_index['doc_terms'].get(doc.id, [])

            # 计算关键词匹配度
            if query_keywords and doc_keywords:
                # 计算交集和并集：用于Jaccard相似度计算
                intersection = set(query_keywords).intersection(set(doc_keywords))
                union = set(query_keywords).union(set(doc_keywords))

                # Jaccard相似度：交集大小除以并集大小
                jaccard_score = len(intersection) / len(union) if union else 0

                # 重要性加权：匹配的关键词占查询关键词的比例，确保查询中的关键词被重视
                importance_score = len(intersection) / len(query_keywords) if query_keywords else 0

                # 综合分数：Jaccard相似度和重要性加权的平均值，平衡两个指标
                final_score = (jaccard_score + importance_score) / 2

                # 只保留分数高于关键词阈值的结果
                if final_score > self.config['keyword_threshold']:
                    results.append(RetrievalResult(
                        document=doc,  # 文档对象
                        score=final_score,  # 关键词匹配分数
                        source='keyword'  # 检索来源
                    ))

        # 按分数降序排序，确保最相关的结果排在前面
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"关键词检索: {len(results)} 个结果")
        return results[:top_k]  # 返回前top_k个结果

    async def extract_entities_from_query(self, query: str) -> List[Entity]:
        """从查询中提取实体

        Args:
            query: 查询文本

        Returns:
            实体列表
        """
        # 构建提示词，要求LLM提取公司实体
        prompt = f"""
        从以下查询中提取公司实体，返回JSON格式：

        查询：{query}

        返回格式：
        {{
            "entities": [
                {{"name": "公司名称", "type": "公司", "confidence": 0.9}}
            ]
        }}

        注意：只提取明确的公司名称实体。
        """

        try:
            # 调用LLM获取实体提取结果
            response = await self.llm_json.ainvoke(prompt)
            # 解析JSON结果
            result = json.loads(response.content)

            entities = []
            # 提取实体列表
            for ent in result.get("entities", []):
                entities.append(Entity(
                    name=ent["name"],
                    type=ent["type"],
                    confidence=ent["confidence"]
                ))

            return entities
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []  # 提取失败返回空列表

    def graph_reasoning(self, entities: List[Entity], query: str) -> GraphResult:
        """
        改进的图谱推理：支持多跳查询和更灵活的匹配

        Args:
            entities: 实体列表
            query: 查询文本

        Returns:
            图谱推理结果
        """
        if not entities:
            return GraphResult([], [], 0.0, [])  # 无实体时返回空结果

        # 提取实体名称列表，用于构建Cypher查询
        entity_names = [e.name for e in entities]

        # 构建更灵活的查询，支持模糊匹配
        cypher_queries = []

        # 1. 直接关系查询（1跳）
        for entity in entity_names:
            cypher_queries.append(f"""
                MATCH (a)-[r]->(b)
                WHERE a.name CONTAINS '{entity}' OR b.name CONTAINS '{entity}'
                RETURN a.name as source, type(r) as relation, b.name as target, 1.0 as confidence
            """)

        # 2. 多跳关系查询（2跳）
        for entity in entity_names:
            cypher_queries.append(f"""
                MATCH (a)-[r1]->(b)-[r2]->(c)
                WHERE a.name CONTAINS '{entity}' OR c.name CONTAINS '{entity}'
                RETURN a.name as source, type(r1) + '->' + type(r2) as relation, c.name as target, 0.8 as confidence
            """)

        all_relationships = []  # 存储所有关系
        reasoning_paths = []  # 存储推理路径

        # 执行图谱查询
        with self.driver.session() as session:
            for cypher in cypher_queries:
                try:
                    # 执行Cypher查询
                    result = session.run(cypher)
                    # 处理查询结果
                    for record in result:
                        # 构建关系字典
                        relationship = {
                            'source': record['source'],  # 源实体
                            'relation': record['relation'],  # 关系类型
                            'target': record['target'],  # 目标实体
                            'confidence': record['confidence']  # 置信度
                        }
                        all_relationships.append(relationship)
                        # 构建推理路径字符串
                        reasoning_paths.append(f"{record['source']} -> {record['relation']} -> {record['target']}")
                except Exception as e:
                    logger.error(f"图谱查询失败: {e}")

        # 去重，避免重复关系
        unique_relationships = []
        seen = set()
        for rel in all_relationships:
            # 使用(source, relation, target)作为唯一键
            key = (rel['source'], rel['relation'], rel['target'])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        # 计算整体置信度
        if unique_relationships:
            # 计算所有关系的平均置信度
            avg_confidence = sum(r['confidence'] for r in unique_relationships) / len(unique_relationships)
        else:
            avg_confidence = 0.0

        logger.info(f"图谱推理: {len(unique_relationships)} 个关系")

        # 返回图谱推理结果
        return GraphResult(
            entities=entity_names,  # 实体列表
            relationships=unique_relationships,  # 去重后的关系列表
            confidence=avg_confidence,  # 平均置信度
            reasoning_path=reasoning_paths  # 推理路径列表
        )

    def test_vector_similarity(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """
        测试向量相似度计算，用于调试和优化

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            相似度测试结果，包含查询、相似度列表和阈值
        """
        # 生成查询向量
        query_embedding = self.embedding_model.encode(query)
        # 生成文档向量
        doc_embeddings = [self.embedding_model.encode(doc) for doc in documents]

        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            # 计算余弦相似度
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append({
                'document_index': i,
                'document_preview': documents[i][:100] + "..." if len(documents[i]) > 100 else documents[i],
                'similarity_score': float(similarity)
            })

        # 按相似度降序排序
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)

        return {
            'query': query,
            'similarities': similarities,
            'threshold': self.config['confidence_threshold']
        }

    def calculate_joint_score(self, vector_results: List[RetrievalResult],
                            keyword_results: List[RetrievalResult],
                            graph_result: GraphResult) -> List[RetrievalResult]:
        """
        计算联合分数：结合向量、关键词和图谱推理结果

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            graph_result: 图谱推理结果

        Returns:
            联合评分结果，按分数降序排列
        """
        # 创建文档分数字典，用于累积各模块得分
        doc_scores = {}

        # 1. 计算向量检索分数：乘以向量权重
        for result in vector_results:
            doc_id = result.document.id
            # 累加向量检索得分，乘以权重
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + result.score * self.config['vector_weight']

        # 2. 计算关键词检索分数：乘以关键词权重
        for result in keyword_results:
            doc_id = result.document.id
            # 累加关键词检索得分，乘以权重
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + result.score * self.config['keyword_weight']

        # 3. 计算图谱推理分数：为包含图谱实体的文档增加分数
        if graph_result.confidence > 0:
            # 遍历所有文档
            for doc in self.documents:
                # 检查文档是否包含图谱中的实体
                for entity in graph_result.entities:
                    if entity.lower() in doc.content.lower():
                        # 累加图谱推理得分，乘以权重
                        doc_scores[doc.id] = doc_scores.get(doc.id, 0) + graph_result.confidence * self.config['graph_weight']

        # 4. 转换为结果列表
        joint_results = []
        for doc_id, score in doc_scores.items():
            # 查找文档对象
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                joint_results.append(RetrievalResult(
                    document=doc,  # 文档对象
                    score=score,  # 联合评分
                    source='joint'  # 检索来源
                ))

        # 5. 按分数降序排序，确保最相关的结果排在前面
        joint_results.sort(key=lambda x: x.score, reverse=True)
        return joint_results

    def error_propagation_guard(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        错误传播防护：过滤低质量结果

        Args:
            results: 检索结果列表

        Returns:
            过滤后的结果列表
        """
        if not results:
            return results  # 无结果时直接返回

        # 计算分数统计
        scores = [r.score for r in results]  # 提取所有结果的分数
        avg_score = sum(scores) / len(scores)  # 计算平均分数

        # 过滤低质量结果：
        # 1. 分数必须高于错误传播阈值
        # 2. 分数必须不低于平均分数的一半
        # 这样可以确保保留高质量结果，同时避免错误信息的传播
        filtered_results = [
            r for r in results
            if r.score >= self.config['error_propagation_threshold'] and r.score >= avg_score * 0.5
        ]

        logger.info(f"错误传播防护: {len(results)} -> {len(filtered_results)} 个结果")
        return filtered_results

    async def multi_hop_qa(self, question: str, max_hops: int = 3) -> Dict[str, Any]:
        """
        多跳问答：结合向量检索、关键词匹配和图谱推理

        Args:
            question: 问题
            max_hops: 最大跳数，默认为3

        Returns:
            多跳问答结果，包含问题、答案、实体、检索统计等信息
        """
        logger.info(f"🔍 开始多跳问答: {question}")

        # 1. 实体提取：从问题中提取公司实体
        entities = await self.extract_entities_from_query(question)
        logger.info(f"提取到 {len(entities)} 个实体")

        # 2. 多源检索：执行向量检索、关键词检索和图谱推理
        vector_results = self.vector_search(question, top_k=self.config['max_retrieval_results'])
        keyword_results = self.keyword_search(question, top_k=self.config['max_retrieval_results'])
        graph_result = self.graph_reasoning(entities, question)

        # 3. 联合评分：结合三个检索模块的结果，计算综合分数
        joint_results = self.calculate_joint_score(vector_results, keyword_results, graph_result)

        # 4. 错误传播防护：过滤低质量结果，避免错误信息的传播
        final_results = self.error_propagation_guard(joint_results)

        # 5. 生成答案：基于过滤后的结果生成最终答案
        if final_results:
            # 取前5个结果作为上下文
            context = [r.document.content for r in final_results[:5]]
            answer = self.generate_final_answer(question, context)
        else:
            answer = "抱歉，我无法找到相关信息来回答您的问题。"

        # 返回多跳问答结果，包含详细的检索统计信息
        return {
            'question': question,  # 原始问题
            'answer': answer,  # 生成的答案
            'entities': [{'name': e.name, 'type': e.type, 'confidence': e.confidence} for e in entities],  # 提取的实体
            'vector_results': len(vector_results),  # 向量检索结果数量
            'keyword_results': len(keyword_results),  # 关键词检索结果数量
            'graph_confidence': graph_result.confidence,  # 图谱推理置信度
            'final_results': len(final_results),  # 最终过滤后的结果数量
            'reasoning_path': graph_result.reasoning_path  # 图谱推理路径
        }

    def generate_final_answer(self, question: str, context: List[str]) -> str:
        """
        生成最终答案

        Args:
            question: 问题
            context: 上下文信息

        Returns:
            最终答案
        """
        if not context:
            return "抱歉，我无法找到相关信息来回答您的问题。"

        # 构建提示词：将上下文信息格式化为提示词
        context_text = "\n\n".join([f"文档{i+1}: {ctx}" for i, ctx in enumerate(context)])

        # 构建详细的提示词，指导LLM基于上下文生成准确的答案
        prompt = f"""
        基于以下文档信息，回答用户问题。请确保答案准确、完整，并基于提供的文档内容。

        问题：{question}

        相关文档：
        {context_text}

        请提供详细的答案：
        """

        try:
            # 使用同步调用
            if hasattr(self.llm_text, 'invoke'):
                response = self.llm_text.invoke(prompt)
            else:
                # 如果是异步LLM，需要在异步环境中调用
                import asyncio
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(self.llm_text.ainvoke(prompt))

            # 提取答案内容
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            # 失败时返回基于上下文的简单回答
            return f"基于检索到的信息，我找到了以下相关内容：\n\n{context_text[:500]}..."


async def demo_improved():
    """改进版演示函数"""
    logger.info("🚀 启动改进版混合RAG系统演示")

    # Neo4j连接
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    # LLM配置
    llm_json = OpenAILLM(
        model_name="gpt-4o-mini",
        model_params={
            "response_format": {"type": "json_object"},  # JSON格式输出
            "temperature": 0.0,  # 降低随机性，确保结构化输出
        },
    )

    llm_text = OpenAILLM(
        model_name="gpt-4o-mini",
        model_params={
            "temperature": 0.3,  # 适当的随机性，生成自然的回答
        },
    )

    # 初始化系统
    rag_system = ImprovedHybridRAGSystem(driver, llm_json, llm_text, use_qwen_embedding=True)

    try:
        # 清理现有数据
        rag_system.clear_all_databases()

        # 加载数据
        file_path = "company.txt"  # 使用相对路径，确保跨平台兼容性
        raw_text = rag_system.load_data_from_file(file_path)

        if not raw_text:
            logger.error("❌ 无法加载数据文件")
            return

        # 处理文档
        documents = await rag_system.process_text_to_documents(raw_text)
        rag_system.add_documents(documents)

        # 从文本中提取关系并构建知识图谱
        relationships = await rag_system.extract_relationships_from_text(raw_text)
        await build_sample_graph_from_relationships(driver, relationships)

        # 测试问答
        questions = [
            "A集团的最大股东是谁？",
            "B资本控制哪些公司？",
            "A集团有多少层级的控股关系？"
        ]

        for question in questions:
            logger.info(f"\n{'='*50}")
            result = await rag_system.multi_hop_qa(question)
            logger.info(f"问题: {result['question']}")
            logger.info(f"答案: {result['answer']}")
            logger.info(f"检索统计: 向量={result['vector_results']}, 关键词={result['keyword_results']}, 图谱置信度={result['graph_confidence']:.2f}")

        # 测试向量相似度
        test_docs = [doc['content'] for doc in documents[:3]]
        similarity_result = rag_system.test_vector_similarity("A集团", test_docs)
        logger.info(f"\n向量相似度测试结果: {similarity_result}")

    finally:
        # 清理资源
        rag_system.clear_all_databases()
        driver.close()
        logger.info("✅ 系统演示完成，资源已清理")


async def build_sample_graph_from_relationships(driver, relationships: List[Tuple[str, str]]):
    """构建示例知识图谱

    Args:
        driver: Neo4j驱动
        relationships: 关系列表，每个关系是(source, target)元组
    """
    logger.info("🔧 开始构建示例知识图谱...")

    with driver.session() as session:
        # 清空现有数据
        session.run("MATCH (n) DETACH DELETE n")

        # 添加关系
        for source, target in relationships:
            # 执行Cypher查询，创建节点和关系
            session.run("""
                MERGE (a:Company {name: $source})
                MERGE (b:Company {name: $target})
                MERGE (a)-[:RELATED_TO]->(b)
            """, source=source, target=target)

    logger.info(f"✅ 示例图构建完成: {len(relationships)} 个关系")


if __name__ == "__main__":
    """执行演示函数"""
    asyncio.run(demo_improved())