"""
检索过程模块
包含混合RAG系统的检索相关功能
实现了向量检索、关键词检索、图谱检索和联合评分机制
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import *
from models import Document, RetrievalResult, GraphResult
from embedding import QwenEmbedding


class HybridRetrievalSystem:
    """混合检索系统

    融合向量检索、关键词检索和图谱检索的混合检索系统
    通过联合评分机制和错误传播防护提升检索质量
    """

    def __init__(self, embedding_model: QwenEmbedding):
        """
        初始化混合检索系统

        Args:
            embedding_model: 嵌入模型，用于将文本转换为向量表示
        """
        self.embedding_model = embedding_model  # 存储嵌入模型
        self.logger = logging.getLogger(__name__)  # 初始化日志记录器
        # 模拟文档存储（实际项目中可能使用数据库或向量存储）
        self.documents = []  # 存储文档列表

    def vector_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[RetrievalResult]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认为RETRIEVAL_TOP_K

        Returns:
            检索结果列表，按相似度降序排列
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(query)

            # 计算相似度
            results = []
            for doc in self.documents:
                if doc.embedding is not None:  # 确保文档有嵌入向量
                    # 计算查询向量与文档向量的余弦相似度
                    similarity = cosine_similarity([query_embedding], [doc.embedding])[0][0]
                    # 创建检索结果对象
                    results.append(RetrievalResult(
                        document=doc,
                        score=similarity,
                        source='vector'
                    ))

            # 按相似度降序排序并返回top_k结果
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            return []  # 出错时返回空列表

    def keyword_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[RetrievalResult]:
        """
        关键词检索

        Args:
            query: 查询文本
            top_k: 返回结果数量，默认为RETRIEVAL_TOP_K

        Returns:
            检索结果列表，按关键词匹配度降序排列
        """
        try:
            # 简单的关键词匹配
            results = []
            # 将查询文本转换为小写并分词，存储为集合以提高匹配效率
            query_tokens = set(query.lower().split())

            for doc in self.documents:
                # 将文档内容转换为小写并分词
                doc_tokens = set(doc.content.lower().split())
                # 计算关键词匹配度：共同词数除以查询词数
                common_tokens = query_tokens.intersection(doc_tokens)
                score = len(common_tokens) / max(len(query_tokens), 1)  # 避免除以零

                if score > 0:  # 只添加有匹配的结果
                    results.append(RetrievalResult(
                        document=doc,
                        score=score,
                        source='keyword'
                    ))

            # 按匹配度降序排序并返回top_k结果
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            self.logger.error(f"关键词检索失败: {e}")
            return []  # 出错时返回空列表

    def graph_search(self, query: str, max_depth: int = GRAPH_MAX_DEPTH) -> List[GraphResult]:
        """
        图检索

        Args:
            query: 查询文本
            max_depth: 最大搜索深度，默认为GRAPH_MAX_DEPTH

        Returns:
            图检索结果列表
        """
        try:
            # 模拟图谱检索（实际项目中应使用Neo4j等图数据库）
            # 这里返回一个空列表，实际实现需要连接图数据库
            return []
        except Exception as e:
            self.logger.error(f"图检索失败: {e}")
            return []  # 出错时返回空列表

    def calculate_joint_score(self, vector_results: List[RetrievalResult],
                            keyword_results: List[RetrievalResult],
                            graph_result: GraphResult) -> Dict[str, Any]:
        """
        改进的联合评分机制

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            graph_result: 图谱检索结果

        Returns:
            联合评分结果，包含综合评分和各模块得分
        """
        # 收集所有文档，避免重复计算
        all_docs = {}

        # 处理向量检索结果
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

        # 处理关键词检索结果
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

        # 处理图谱推理分数
        graph_confidence = graph_result.confidence
        for doc_id in all_docs:
            # 简单的图谱相关性评分，使用图谱结果的置信度
            all_docs[doc_id]['graph_score'] = graph_confidence

        # 计算综合分数
        final_results = []
        for doc_id, scores in all_docs.items():
            # 根据权重计算联合得分
            joint_score = (
                scores['vector_score'] * RETRIEVAL_VECTOR_WEIGHT +
                scores['keyword_score'] * RETRIEVAL_KEYWORD_WEIGHT +
                scores['graph_score'] * RETRIEVAL_GRAPH_WEIGHT
            )

            # 添加到最终结果列表
            final_results.append({
                'document': scores['document'],
                'joint_score': joint_score,
                'vector_score': scores['vector_score'],
                'keyword_score': scores['keyword_score'],
                'graph_score': scores['graph_score']
            })

        # 按综合分数降序排序
        final_results.sort(key=lambda x: x['joint_score'], reverse=True)

        # 计算整体置信度
        if final_results:
            max_score = final_results[0]['joint_score']  # 获取最高得分
            overall_confidence = min(max_score, 1.0)  # 确保置信度不超过1.0
        else:
            overall_confidence = 0.0  # 无结果时置信度为0

        # 返回联合评分结果
        return {
            'results': final_results,
            'overall_confidence': overall_confidence,
            'vector_count': len(vector_results),
            'keyword_count': len(keyword_results),
            'graph_confidence': graph_result.confidence
        }

    def error_propagation_guard(self, results: Dict[str, Any],
                              vector_results: List[RetrievalResult],
                              graph_result: GraphResult) -> Dict[str, Any]:
        """
        改进的错误传播防护

        Args:
            results: 联合评分结果
            vector_results: 向量检索结果
            graph_result: 图谱检索结果

        Returns:
            带有错误传播防护的结果，包含警告信息和置信度等级
        """
        warnings = []  # 存储警告信息

        # 检查整体置信度
        if results['overall_confidence'] < ERROR_PROPAGATION_THRESHOLD:
            warnings.append("整体置信度过低，可能存在错误传播风险")

        # 检查各模块置信度
        if not vector_results:
            warnings.append("向量检索无结果，建议检查embedding质量")

        if results['keyword_count'] == 0:
            warnings.append("关键词检索无结果，建议优化分词策略")

        if graph_result.confidence < ERROR_PROPAGATION_THRESHOLD:
            warnings.append("图谱推理置信度过低，建议人工验证")

        # 确定置信度等级
        confidence = results['overall_confidence']
        if confidence >= 0.7:
            confidence_level = "high"
        elif confidence >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # 返回带有错误传播防护的结果
        return {
            **results,  # 保留原结果
            'confidence_level': confidence_level,  # 添加置信度等级
            'warnings': warnings  # 添加警告信息
        }

    def hybrid_search(self, query: str,
                     vector_weight: float = RETRIEVAL_VECTOR_WEIGHT,
                     keyword_weight: float = RETRIEVAL_KEYWORD_WEIGHT,
                     graph_weight: float = RETRIEVAL_GRAPH_WEIGHT,
                     top_k: int = RETRIEVAL_TOP_K) -> Dict[str, Any]:
        """
        混合检索

        Args:
            query: 查询文本
            vector_weight: 向量检索权重，默认为RETRIEVAL_VECTOR_WEIGHT
            keyword_weight: 关键词检索权重，默认为RETRIEVAL_KEYWORD_WEIGHT
            graph_weight: 图检索权重，默认为RETRIEVAL_GRAPH_WEIGHT
            top_k: 返回结果数量，默认为RETRIEVAL_TOP_K

        Returns:
            混合检索结果，包含检索结果、各模块结果数量、置信度和警告信息
        """
        try:
            # 执行各种检索
            vector_results = self.vector_search(query, top_k)
            keyword_results = self.keyword_search(query, top_k)
            graph_results = self.graph_search(query)

            # 创建GraphResult对象用于联合评分
            if graph_results:
                # 合并图谱结果
                entities = []
                relationships = []
                total_score = 0
                for result in graph_results:
                    entities.extend(result.entities)
                    relationships.extend(result.relationships)
                    total_score += result.confidence

                # 创建合并后的GraphResult对象
                graph_result = GraphResult(
                    entities=entities,
                    relationships=relationships,
                    confidence=total_score / len(graph_results) if graph_results else 0.0,
                    reasoning_path=[]
                )
            else:
                # 无图谱结果时创建空的GraphResult对象
                graph_result = GraphResult(entities=[], relationships=[], confidence=0.0, reasoning_path=[])

            # 使用联合评分机制计算综合得分
            scoring_results = self.calculate_joint_score(vector_results, keyword_results, graph_result)

            # 错误传播防护
            final_results = self.error_propagation_guard(scoring_results, vector_results, graph_result)

            # 转换为标准格式
            results = []
            for result in final_results['results'][:top_k]:
                results.append({
                    'content': result['document'].content,
                    'score': result['joint_score'],
                    'source': 'hybrid'
                })

            # 返回混合检索结果
            return {
                'results': results,
                'vector_count': len(vector_results),
                'keyword_count': len(keyword_results),
                'graph_count': len(graph_results),
                'total_count': len(results),
                'confidence': final_results['overall_confidence'],
                'confidence_level': final_results['confidence_level'],
                'warnings': final_results['warnings']
            }

        except Exception as e:
            self.logger.error(f"混合检索失败: {e}")
            # 出错时返回错误信息
            return {
                'results': [],
                'vector_count': 0,
                'keyword_count': 0,
                'graph_count': 0,
                'total_count': 0,
                'confidence': 0.0,
                'confidence_level': 'low',
                'warnings': ['检索过程中出现错误']
            }

    def multi_hop_qa(self, question: str, max_hops: int = GRAPH_MAX_HOPS) -> Dict[str, Any]:
        """
        多跳问答

        Args:
            question: 问题
            max_hops: 最大跳数，默认为GRAPH_MAX_HOPS

        Returns:
            多跳问答结果，包含问题、上下文、检索结果和推理结果
        """
        try:
            # 首先进行混合检索获取初始上下文
            initial_results = self.hybrid_search(question)

            # 模拟多跳推理（实际项目中应使用图数据库）
            graph_reasoning_results = []

            # 合并结果
            all_context = []

            # 添加检索结果到上下文
            for result in initial_results['results']:
                all_context.append(result['content'])

            # 添加图推理结果到上下文
            for result in graph_reasoning_results:
                context = f"推理路径: {' -> '.join(result['path'])}, 置信度: {result['confidence']:.3f}"
                all_context.append(context)

            # 返回多跳问答结果
            return {
                'question': question,
                'context': all_context,
                'retrieval_results': initial_results,
                'reasoning_results': graph_reasoning_results,
                'total_context_items': len(all_context)
            }

        except Exception as e:
            self.logger.error(f"多跳问答失败: {e}")
            # 出错时返回错误信息
            return {
                'question': question,
                'context': [],
                'retrieval_results': {'results': [], 'vector_count': 0, 'keyword_count': 0, 'graph_count': 0, 'total_count': 0},
                'reasoning_results': [],
                'total_context_items': 0
            }

    def add_document(self, document: Document):
        """
        添加文档到检索系统

        Args:
            document: 文档对象
        """
        # 如果文档没有嵌入向量，生成一个
        if document.embedding is None:
            document.embedding = self.embedding_model.encode(document.content)
        # 将文档添加到文档列表
        self.documents.append(document)