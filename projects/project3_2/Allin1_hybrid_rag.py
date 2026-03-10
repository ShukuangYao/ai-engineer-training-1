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

logging.basicConfig(level=logging.INFO)  # 配置日志级别
logger = logging.getLogger(__name__)  # 初始化日志记录器

# 设置通义千问API密钥，优先从环境变量获取，否则使用默认值
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")

@dataclass
class Entity:
    """实体类，表示知识图谱中的实体"""
    name: str  # 实体名称
    type: str  # 实体类型
    confidence: float = 1.0  # 实体识别置信度

@dataclass
class Relationship:
    """关系类，表示知识图谱中实体之间的关系"""
    source: str  # 源实体
    target: str  # 目标实体
    type: str  # 关系类型
    confidence: float = 1.0  # 关系识别置信度

@dataclass
class Document:
    """文档类，表示检索系统中的文档"""
    id: str  # 文档ID
    content: str  # 文档内容
    metadata: Dict[str, Any]  # 文档元数据
    embedding: Optional[np.ndarray] = None  # 文档嵌入向量

@dataclass
class RetrievalResult:
    """检索结果类，表示检索系统返回的结果"""
    document: Document  # 检索到的文档
    score: float  # 检索分数
    source: str  # 检索来源: 'vector' or 'keyword' or 'graph'

@dataclass
class GraphResult:
    """图谱结果类，表示知识图谱推理的结果"""
    entities: List[str]  # 识别出的实体列表
    relationships: List[Dict]  # 识别出的关系列表
    confidence: float  # 结果置信度
    reasoning_path: List[str]  # 推理路径

class QwenEmbedding:
    """通义千问文本嵌入服务"""

    def __init__(self, model_name="text-embedding-v3"):
        """初始化嵌入服务"""
        self.model_name = model_name  # 存储模型名称

    def encode(self, texts):
        """编码文本为向量"""
        # 确保输入是列表形式
        if isinstance(texts, str):
            texts = [texts]

        try:
            from dashscope import TextEmbedding  # 动态导入，减少初始化时的依赖

            # 调用通义千问文本嵌入API
            response = TextEmbedding.call(
                model=self.model_name,
                input=texts
            )

            if response.status_code == 200:  # API调用成功
                embeddings = []
                # 处理API返回的嵌入结果
                for output in response.output['embeddings']:
                    embeddings.append(np.array(output['embedding']))

                # 根据输入返回相应格式的结果
                return embeddings[0] if len(embeddings) == 1 else embeddings
            else:
                # API调用失败，记录错误并降级到本地模型
                logger.error(f"通义千问embedding调用失败: {response}")
                fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
                return fallback_model.encode(texts)

        except Exception as e:
            # 发生异常，记录错误并降级到本地模型
            logger.error(f"通义千问embedding异常: {e}")
            fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
            return fallback_model.encode(texts)

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
        """初始化改进的混合RAG系统"""
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
        """从文件加载数据"""
        try:
            # 打开文件并读取内容
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
            return ""  # 读取失败返回空字符串

    async def process_text_to_documents(self, raw_text: str) -> List[Dict[str, Any]]:
        """将原始文本处理为文档"""
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
        """从文本中提取关系"""
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
        """使用jieba提取关键词"""
        # 使用TF-IDF提取关键词，取前10个
        keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)

        # 添加基础分词，获取更多可能的关键词
        words = jieba.cut(text)
        basic_words = [w for w in words if len(w) > 1 and w.isalnum()]

        # 合并并去重，确保关键词唯一性
        all_keywords = list(set(keywords + basic_words))
        return all_keywords

    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档到向量数据库"""
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
        """
        if not self.documents:
            return []  # 无文档时返回空列表

        # 查询向量化
        query_embedding = self.embedding_model.encode(query)

        # 计算相似度
        doc_embeddings = np.array([doc.embedding for doc in self.documents])
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # 使用更低的阈值，提升召回率
        results = []
        for i, score in enumerate(similarities):
            if score > self.config['confidence_threshold']:
                results.append(RetrievalResult(
                    document=self.documents[i],
                    score=float(score),
                    source='vector'
                ))

        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"向量检索: {len(results)} 个结果")
        return results[:top_k]  # 返回前top_k个结果

    def keyword_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        改进的关键词检索：使用jieba分词和更灵活的匹配
        """
        if not self.documents:
            return []  # 无文档时返回空列表

        # 提取查询关键词
        query_keywords = self.extract_keywords(query)

        results = []
        for i, doc in enumerate(self.documents):
            doc_keywords = self.keyword_index['doc_terms'].get(doc.id, [])

            # 计算关键词匹配度
            if query_keywords and doc_keywords:
                # 计算交集和并集
                intersection = set(query_keywords).intersection(set(doc_keywords))
                union = set(query_keywords).union(set(doc_keywords))

                if intersection:
                    # Jaccard相似度
                    jaccard_score = len(intersection) / len(union)

                    # 考虑关键词在文档中的重要性
                    importance_score = len(intersection) / len(query_keywords)

                    # 综合分数
                    final_score = (jaccard_score + importance_score) / 2

                    if final_score > self.config['keyword_threshold']:
                        results.append(RetrievalResult(
                            document=doc,
                            score=final_score,
                            source='keyword'
                        ))

        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"关键词检索: {len(results)} 个结果")
        return results[:top_k]  # 返回前top_k个结果

    async def extract_entities_from_query(self, query: str) -> List[Entity]:
        """
        改进的实体提取
        """
        # 构建提示词，要求LLM提取实体
        prompt = f"""
        从问题中提取所有相关的实体（公司名、人名等）：

        问题：{query}

        返回JSON格式：
        {{
            "entities": [
                {{"name": "实体名", "type": "Company|Person", "confidence": 0.9}}
            ]
        }}

        注意：尽可能提取所有可能相关的实体，包括简称和全称。
        """

        try:
            # 调用LLM获取实体提取结果
            response = await self.llm_json.ainvoke(prompt)
            # 解析JSON结果
            result = json.loads(response.content)

            entities = []
            # 提取实体列表
            for e in result.get("entities", []):
                entities.append(Entity(
                    name=e["name"],
                    type=e["type"],
                    confidence=e.get("confidence", 0.8)
                ))

            logger.info(f"提取到 {len(entities)} 个实体")
            return entities
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []  # 提取失败返回空列表

    def graph_reasoning(self, entities: List[Entity], query: str) -> GraphResult:
        """
        改进的图谱推理：支持多跳查询和更灵活的匹配
        """
        if not entities:
            return GraphResult([], [], 0.0, [])  # 无实体时返回空结果

        entity_names = [e.name for e in entities]  # 提取实体名称

        # 构建更灵活的查询，支持模糊匹配
        cypher_queries = []

        # 1. 直接关系查询
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

        with self.driver.session() as session:
            for cypher in cypher_queries:
                try:
                    # 执行Cypher查询
                    result = session.run(cypher)
                    # 处理查询结果
                    for record in result:
                        relationship = {
                            'source': record['source'],
                            'relation': record['relation'],
                            'target': record['target'],
                            'confidence': record['confidence']
                        }
                        all_relationships.append(relationship)
                        reasoning_paths.append(f"{record['source']} -> {record['relation']} -> {record['target']}")
                except Exception as e:
                    logger.error(f"图谱查询失败: {e}")

        # 去重，避免重复关系
        unique_relationships = []
        seen = set()
        for rel in all_relationships:
            key = (rel['source'], rel['relation'], rel['target'])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        # 计算整体置信度
        if unique_relationships:
            avg_confidence = sum(r['confidence'] for r in unique_relationships) / len(unique_relationships)
        else:
            avg_confidence = 0.0

        logger.info(f"图谱推理: {len(unique_relationships)} 个关系")

        return GraphResult(
            entities=entity_names,
            relationships=unique_relationships,
            confidence=avg_confidence,
            reasoning_path=reasoning_paths
        )

    def test_vector_similarity(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """
        测试向量相似度计算，用于调试和优化
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
                            graph_result: GraphResult) -> Dict[str, Any]:
        """
        改进的联合评分机制

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            graph_result: 图谱推理结果

        Returns:
            联合评分结果，包含综合得分、置信度和各模块统计信息
        """
        # 收集所有文档，避免重复计算
        all_docs = {}

        # 1. 处理向量检索结果
        for result in vector_results:
            doc_id = result.document.id
            if doc_id not in all_docs:
                # 初始化文档评分结构
                all_docs[doc_id] = {
                    'document': result.document,
                    'vector_score': 0.0,
                    'keyword_score': 0.0,
                    'graph_score': 0.0
                }
            # 更新向量检索得分
            all_docs[doc_id]['vector_score'] = result.score

        # 2. 处理关键词检索结果
        for result in keyword_results:
            doc_id = result.document.id
            if doc_id not in all_docs:
                # 初始化文档评分结构
                all_docs[doc_id] = {
                    'document': result.document,
                    'vector_score': 0.0,
                    'keyword_score': 0.0,
                    'graph_score': 0.0
                }
            # 更新关键词检索得分
            all_docs[doc_id]['keyword_score'] = result.score

        # 3. 处理图谱推理分数（基于文档内容与图谱关系的匹配度）
        graph_confidence = graph_result.confidence
        for doc_id in all_docs:
            # 简单的图谱相关性评分，使用图谱结果的置信度
            all_docs[doc_id]['graph_score'] = graph_confidence

        # 4. 计算综合分数
        final_results = []
        for doc_id, scores in all_docs.items():
            # 根据权重计算联合得分
            joint_score = (
                scores['vector_score'] * self.config['vector_weight'] +
                scores['keyword_score'] * self.config['keyword_weight'] +
                scores['graph_score'] * self.config['graph_weight']
            )

            # 添加到最终结果列表
            final_results.append({
                'document': scores['document'],
                'joint_score': joint_score,
                'vector_score': scores['vector_score'],
                'keyword_score': scores['keyword_score'],
                'graph_score': scores['graph_score']
            })

        # 5. 按综合分数降序排序
        final_results.sort(key=lambda x: x['joint_score'], reverse=True)

        # 6. 计算整体置信度
        if final_results:
            max_score = final_results[0]['joint_score']  # 获取最高得分
            overall_confidence = min(max_score, 1.0)  # 确保置信度不超过1.0
        else:
            overall_confidence = 0.0  # 无结果时置信度为0

        # 返回联合评分结果
        return {
            'results': final_results,  # 最终评分结果列表
            'overall_confidence': overall_confidence,  # 整体置信度
            'vector_count': len(vector_results),  # 向量检索结果数量
            'keyword_count': len(keyword_results),  # 关键词检索结果数量
            'graph_confidence': graph_result.confidence  # 图谱推理置信度
        }

    def error_propagation_guard(self, results: Dict[str, Any],
                              vector_results: List[RetrievalResult],
                              graph_result: GraphResult) -> Dict[str, Any]:
        """
        改进的错误传播防护

        Args:
            results: 联合评分结果
            vector_results: 向量检索结果
            graph_result: 图谱推理结果

        Returns:
            带有错误传播防护的结果，包含置信度等级和警告信息
        """
        warnings = []  # 存储警告信息

        # 1. 检查整体置信度
        if results['overall_confidence'] < self.config['error_propagation_threshold']:
            warnings.append("整体置信度过低，可能存在错误传播风险")

        # 2. 检查各模块置信度
        if not vector_results:
            warnings.append("向量检索无结果，建议检查embedding质量")

        if results['keyword_count'] == 0:
            warnings.append("关键词检索无结果，建议优化分词策略")

        if graph_result.confidence < 0.3:
            warnings.append("图谱推理置信度过低，建议人工验证")

        # 3. 确定置信度等级
        confidence = results['overall_confidence']
        if confidence >= 0.7:
            confidence_level = "high"
        elif confidence >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # 4. 返回带有错误传播防护的结果
        return {
            **results,  # 保留原结果
            'confidence_level': confidence_level,  # 添加置信度等级
            'warnings': warnings  # 添加警告信息
        }

    async def multi_hop_qa(self, question: str) -> Dict[str, Any]:
        """
        改进的多跳问答

        Args:
            question: 用户问题

        Returns:
            多跳问答结果，包含答案、置信度、警告信息和检索统计
        """
        logger.info(f"开始处理问题: {question}")

        # 1. 实体提取：从问题中提取相关实体
        entities = await self.extract_entities_from_query(question)

        # 2. 多源检索：执行向量检索、关键词检索和图谱推理
        vector_results = self.vector_search(question, top_k=self.config['max_retrieval_results'])
        keyword_results = self.keyword_search(question, top_k=self.config['max_retrieval_results'])
        graph_result = self.graph_reasoning(entities, question)

        # 3. 联合评分：结合三个检索模块的结果，计算综合分数
        scoring_results = self.calculate_joint_score(vector_results, keyword_results, graph_result)

        # 4. 错误传播防护：检测并添加警告信息，确定置信度等级
        final_results = self.error_propagation_guard(scoring_results, vector_results, graph_result)

        # 5. 生成最终答案：基于检索结果和图谱推理生成答案
        final_answer = await self.generate_final_answer(
            question, vector_results, keyword_results, graph_result, final_results
        )

        # 返回多跳问答结果
        return {
            'question': question,  # 原始问题
            'answer': final_answer,  # 生成的答案
            'confidence': final_results['overall_confidence'],  # 整体置信度
            'confidence_level': final_results['confidence_level'],  # 置信度等级
            'warnings': final_results['warnings'],  # 警告信息
            'vector_count': len(vector_results),  # 向量检索结果数量
            'keyword_count': len(keyword_results),  # 关键词检索结果数量
            'graph_relationships': len(graph_result.relationships)  # 图谱关系数量
        }

    async def generate_final_answer(self, question: str,
                                  vector_results: List[RetrievalResult],
                                  keyword_results: List[RetrievalResult],
                                  graph_result: GraphResult,
                                  scoring_results: Dict[str, Any]) -> str:
        """
        改进的答案生成

        Args:
            question: 用户问题
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            graph_result: 图谱推理结果
            scoring_results: 联合评分结果

        Returns:
            生成的最终答案
        """
        # 1. 收集上下文信息
        contexts = []

        # 向量检索上下文，取前3个结果
        for result in vector_results[:3]:
            contexts.append(f"文档内容: {result.document.content}")

        # 图谱推理上下文，取前5个关系
        if graph_result.relationships:
            graph_context = "图谱关系:\n"
            for rel in graph_result.relationships[:5]:
                graph_context += f"- {rel['source']} {rel['relation']} {rel['target']}\n"
            contexts.append(graph_context)

        # 2. 组合上下文文本
        context_text = "\n\n".join(contexts)

        # 3. 构建提示词
        prompt = f"""
        基于以下信息回答问题：

        问题：{question}

        上下文信息：
        {context_text}

        请根据提供的信息给出准确、详细的答案。如果信息不足，请说明不确定性。
        """

        # 4. 调用LLM生成答案
        try:
            response = await self.llm_text.ainvoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return "抱歉，无法生成答案。"

# 测试和演示函数
async def demo_improved():
    """改进版本的演示"""
    # Neo4j连接
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    # LLM配置
    llm_json = OpenAILLM(
        model_name="gpt-4o-mini",
        model_params={"response_format": {"type": "json_object"}}  # JSON格式输出
    )
    llm_text = OpenAILLM(model_name="gpt-4o-mini")  # 文本输出

    # 创建改进的系统
    system = ImprovedHybridRAGSystem(driver, llm_json, llm_text, use_qwen_embedding=True)

    # 清理数据库
    system.clear_all_databases()

    # 加载数据
    raw_text = system.load_data_from_file("company.txt")

    # 处理文档
    documents = await system.process_text_to_documents(raw_text)
    system.add_documents(documents)
    print(f"✅ 成功添加 {len(documents)} 个文档到向量数据库")

    # 构建图谱
    print("🔗 提取控股关系并构建知识图谱...")
    relationships = await system.extract_relationships_from_text(raw_text)
    await build_sample_graph_from_relationships(driver, relationships)
    print(f"✅ 成功构建包含 {len(relationships)} 个关系的知识图谱")

    # 测试问题
    questions = [
        "A集团的最大股东是谁？",
        "B资本控制哪些公司？",
        "A集团有多少层级的控股关系？"
    ]

    for question in questions:
        print(f"\n📋 问题: {question}")
        print("-" * 40)

        result = await system.multi_hop_qa(question)

        print(f"🎯 最终答案: {result['answer']}")
        print(f"📊 整体置信度: {result['confidence']:.2f} ({result['confidence_level']})")

        if result['warnings']:
            print(f"⚠️  警告: {'; '.join(result['warnings'])}")

        print(f"🔍 检索到 {result['vector_count']} 个向量结果")
        print(f"🔗 图谱推理找到 {result['graph_relationships']} 个关系")

    # 向量相似度测试
    print("\n🧪 向量相似度测试:")
    test_docs = [doc['content'] for doc in documents[:5]]
    similarity_test = system.test_vector_similarity("B资本控制哪些公司", test_docs)

    print(f"查询: {similarity_test['query']}")
    print(f"阈值: {similarity_test['threshold']}")
    for sim in similarity_test['similarities'][:3]:
        print(f"  相似度 {sim['similarity_score']:.3f}: {sim['document_preview']}")

    # 关闭Neo4j驱动
    driver.close()

async def build_sample_graph_from_relationships(driver, relationships: List[Tuple[str, str]]):
    """从关系列表构建图谱"""
    with driver.session() as session:
        for source, target in relationships:
            # 执行Cypher查询，创建节点和关系
            session.run("""
                MERGE (a:Company {name: $source})
                MERGE (b:Company {name: $target})
                MERGE (a)-[:CONTROLS]->(b)
            """, source=source, target=target)

if __name__ == "__main__":
    # 执行演示函数
    asyncio.run(demo_improved())