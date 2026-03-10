# 导入必要的库
import asyncio  # 异步IO支持
import json  # JSON处理
import logging  # 日志记录
import os  # 操作系统功能
from typing import List, Dict, Any, Tuple, Optional  # 类型提示
from dataclasses import dataclass  # 数据类装饰器
from neo4j import GraphDatabase  # Neo4j数据库驱动
from openai import AsyncOpenAI  # OpenAI异步客户端
import nest_asyncio  # 嵌套异步支持
import re  # 正则表达式

# 应用嵌套异步支持，解决在事件循环中运行异步代码的问题
nest_asyncio.apply()

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查LlamaIndex是否可用
LLAMA_INDEX_AVAILABLE = False
try:
    from llama_index import Document, VectorStoreIndex, ServiceContext
    LLAMA_INDEX_AVAILABLE = True
    logger.info("✅ Successfully imported LlamaIndex")
except ImportError as e:
    logger.warning(f"⚠️ LlamaIndex not available: {e}")
    logger.warning("⚠️ Using keyword matching fallback")

# 数据类定义

@dataclass
class Entity:
    """实体类，用于存储公司实体信息"""
    name: str  # 实体名称
    type: str  # 实体类型
    confidence: float = 1.0  # 置信度评分
    properties: Dict[str, Any] = None  # 实体属性

    def __post_init__(self):
        """初始化后处理，确保properties不为None"""
        if self.properties is None:
            self.properties = {}

@dataclass
class Relationship:
    """关系类，用于存储控股关系信息"""
    source: str  # 源实体
    target: str  # 目标实体
    type: str  # 关系类型
    confidence: float = 1.0  # 置信度评分
    evidence_source: str = ""  # 证据来源
    properties: Dict[str, Any] = None  # 关系属性

    def __post_init__(self):
        """初始化后处理，确保properties不为None"""
        if self.properties is None:
            self.properties = {}

@dataclass
class ReasoningStep:
    """推理步骤类，用于记录推理过程"""
    step: int  # 步骤编号
    query: str  # 查询内容
    source: str  # 信息来源
    result: str  # 结果
    confidence: float = 1.0  # 置信度
    evidence: str = ""  # 证据

@dataclass
class QueryResult:
    """查询结果类，用于存储最终查询结果"""
    answer: str  # 答案
    rag_score: float  # RAG评分
    kg_score: float  # 图谱评分
    combined_score: float  # 综合评分
    confidence_score: float  # 整体置信度
    reasoning_path: List[ReasoningStep]  # 推理路径
    source_type: str  # 信息来源类型
    cypher_queries: List[str]  # Cypher查询
    warnings: List[str]  # 警告信息
    evidence_chain: List[str]  # 证据链

class SimpleLLM:
    """简单LLM类，封装OpenAI API调用"""

    def __init__(self, model_name: str = "qwen-plus", api_key: str = None, base_url: str = None):
        """初始化LLM客户端

        Args:
            model_name: 模型名称
            api_key: API密钥
            base_url: API基础URL
        """
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),  # 优先使用传入的API密钥，否则从环境变量获取
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 默认使用DashScope API
        )
        self.model_name = model_name  # 保存模型名称

    async def invoke(self, prompt: str, json_mode: bool = False, temperature: float = 0):
        """调用LLM生成回复

        Args:
            prompt: 提示词
            json_mode: 是否返回JSON格式
            temperature: 生成温度

        Returns:
            模型生成的回复内容
        """
        messages = [{"role": "user", "content": prompt}]  # 构建消息
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}  # 设置JSON格式输出

        response = await self.client.chat.completions.create(**kwargs)  # 调用API
        return response.choices[0].message.content  # 返回生成的内容

class DocumentRetriever:
    """文档检索类，支持LlamaIndex向量检索和关键词匹配回退"""

    def __init__(self, documents: List[str]):
        """初始化文档检索器

        Args:
            documents: 文档列表
        """
        self.documents = documents  # 存储文档列表
        self.index = None  # LlamaIndex向量索引
        self.query_engine = None  # 查询引擎
        self.use_fallback = not LLAMA_INDEX_AVAILABLE  # 是否使用回退机制
        self.doc_source_map = {i: f"文档_{i+1}" for i in range(len(documents))}  # 文档来源映射

        if LLAMA_INDEX_AVAILABLE:
            self._build_index()  # 构建LlamaIndex索引
        if self.index is None:
            self.use_fallback = True
            logger.warning("⚠️ Falling back to keyword matching")

    def _build_index(self):
        """构建LlamaIndex向量索引"""
        try:
            logger.info("🔨 Building LlamaIndex index...")

            # 创建LlamaIndex文档
            llama_docs = [Document(text=doc, doc_id=f"doc_{i}") for i, doc in enumerate(self.documents)]
            logger.info(f"✅ Created {len(llama_docs)} documents")

            # 创建服务上下文
            service_context = ServiceContext.from_defaults(llm=None)
            logger.info("✅ Service context created")

            # 构建向量索引
            self.index = VectorStoreIndex.from_documents(
                llama_docs,
                service_context=service_context
            )
            logger.info("✅ VectorStoreIndex built")

            # 创建检索器
            self.query_engine = self.index.as_retriever(similarity_top_k=3)
            self.use_fallback = False
            logger.info("✅ LlamaIndex index built successfully!")
        except Exception as e:
            logger.error(f"❌ LlamaIndex index build failed: {e}")
            self.index = None
            self.use_fallback = True

    def retrieve_with_source(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[str, str, float]], float]:
        """检索相关文档

        Args:
            query: 查询语句
            top_k: 返回前k个结果

        Returns:
            (文档列表, 平均评分)
        """
        if not self.use_fallback and self.index is not None:
            return self._llama_retrieve(query, top_k)  # 使用LlamaIndex检索
        else:
            return self._fallback_retrieve(query, top_k)  # 使用关键词匹配回退

    def _llama_retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[str, str, float]], float]:
        """使用LlamaIndex进行语义检索

        Args:
            query: 查询语句
            top_k: 返回前k个结果

        Returns:
            (文档列表, 平均评分)
        """
        try:
            nodes = self.query_engine.retrieve(query)  # 检索相关文档

            results = []
            scores = []
            for node in nodes[:top_k]:
                doc_text = node.text  # 文档文本
                doc_source = node.metadata.get("doc_id", "unknown")  # 文档来源
                score = float(node.score)  # 相似度评分
                results.append((doc_text, doc_source, score))
                scores.append(score)

            avg_score = float(sum(scores) / len(scores)) if scores else 0.0  # 计算平均评分
            logger.info(f"✅ LlamaIndex retrieved {len(results)} docs, avg score: {avg_score:.3f}")
            return results, avg_score
        except Exception as e:
            logger.error(f"❌ LlamaIndex retrieval failed: {e}, using fallback")
            return self._fallback_retrieve(query, top_k)  # 失败时回退

    def _fallback_retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[str, str, float]], float]:
        """使用关键词匹配进行回退检索

        Args:
            query: 查询语句
            top_k: 返回前k个结果

        Returns:
            (文档列表, 平均评分)
        """
        # 提取查询中的关键词
        query_words = set(re.findall(r'[\w\u4e00-\u9fff]+', query.lower()))

        scores = []
        for i, doc in enumerate(self.documents):
            # 提取文档中的关键词
            doc_words = set(re.findall(r'[\w\u4e00-\u9fff]+', doc.lower()))
            # 计算关键词重叠度
            overlap = len(query_words & doc_words)
            # 计算评分
            score = min(1.0, overlap / max(len(query_words), 1))
            scores.append((i, score))

        # 按评分排序
        scores.sort(key=lambda x: x[1], reverse=True)
        # 取前k个结果
        top_indices = [i for i, _ in scores[:top_k]]

        results = []
        for i in top_indices:
            results.append((self.documents[i], self.doc_source_map[i], float(scores[top_indices.index(i)][1])))

        # 计算平均评分
        avg_score = float(sum(s for _, s in scores[:top_k]) / len(scores[:top_k])) if scores[:top_k] else 0.0
        return results, avg_score

class ConsistencyChecker:
    """一致性检查器，防止错误传播"""

    def __init__(self, driver):
        """初始化一致性检查器

        Args:
            driver: Neo4j驱动
        """
        self.driver = driver  # Neo4j驱动
        self.warnings = []  # 警告信息

    def check_relationship_consistency(self, source: str, target: str, rel_type: str) -> Tuple[bool, str, float]:
        """检查关系一致性

        Args:
            source: 源实体
            target: 目标实体
            rel_type: 关系类型

        Returns:
            (是否一致, 消息, 置信度)
        """
        query = """
        MATCH (s {name: $source})-[r]->(t {name: $target})
        RETURN type(r) as rel_type, r as properties
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, source=source, target=target)
                record = result.single()
                if record:
                    if record["rel_type"] == rel_type:
                        return True, "关系一致", 0.95
                    else:
                        return False, f"关系类型冲突：期望 {rel_type}，实际 {record['rel_type']}", 0.1
                return True, "关系不存在", 0.8
        except Exception as e:
            return False, f"一致性检查失败: {e}", 0.0

    def check_cyclic_relationship(self, source: str, target: str) -> Tuple[bool, str, float]:
        """检查循环关系

        Args:
            source: 源实体
            target: 目标实体

        Returns:
            (是否无循环, 消息, 置信度)
        """
        query = """
        MATCH path = (s {name: $target})-[:CONTROLS*1..]->(t {name: $source})
        RETURN path
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, source=source, target=target)
                if result.peek():
                    return False, "检测到循环控股关系", 0.0
                return True, "无循环关系", 0.9
        except Exception as e:
            return False, f"循环检查失败: {e}", 0.0

    def cross_validate_with_documents(self, relationship: Relationship, documents: List[str]) -> Tuple[bool, float, List[str]]:
        """使用文档交叉验证关系

        Args:
            relationship: 关系
            documents: 文档列表

        Returns:
            (是否验证通过, 置信度, 证据)
        """
        evidence = []
        for doc in documents:
            source_pattern = re.escape(relationship.source)  # 转义源实体名称
            target_pattern = re.escape(relationship.target)  # 转义目标实体名称

            # 检查文档中是否同时包含源实体和目标实体，以及控股相关词汇
            if source_pattern in doc and target_pattern in doc:
                if "控股" in doc or "持股" in doc:
                    evidence.append(doc[:200])  # 提取证据

        if evidence:
            # 计算置信度，基于证据数量
            confidence = min(0.95, 0.7 + 0.1 * len(evidence))
            return True, confidence, evidence
        else:
            return False, 0.3, []

    def get_warnings(self) -> List[str]:
        """获取警告信息

        Returns:
            警告信息列表
        """
        return self.warnings

class GraphBuilder:
    """图谱构建器，从文档中提取实体和关系并构建知识图谱"""

    def __init__(self, driver, llm: SimpleLLM, documents: List[str]):
        """初始化图谱构建器

        Args:
            driver: Neo4j驱动
            llm: LLM实例
            documents: 文档列表
        """
        self.driver = driver  # Neo4j驱动
        self.llm = llm  # LLM实例
        self.documents = documents  # 文档列表
        self.consistency_checker = ConsistencyChecker(driver)  # 一致性检查器
        self.evidence_chain = []  # 证据链

    async def extract_entities_and_relationships(self, text: str, doc_source: str = "") -> Tuple[List[Entity], List[Relationship]]:
        """从文本中提取实体和关系

        Args:
            text: 文本内容
            doc_source: 文档来源

        Returns:
            (实体列表, 关系列表)
        """
        prompt = f"""
        从以下财报文本中提取公司实体和控股关系，并给出置信度评分(0-1)：
        {text}

        返回JSON格式：
        {{
            "entities": [
                {{
                    "name": "公司名称",
                    "type": "Company",
                    "confidence": 0.95,
                    "properties": {{"业务": "电子产品制造", "营收": "150亿元"}}
                }}
            ],
            "relationships": [
                {{
                    "source": "母公司",
                    "target": "子公司",
                    "type": "CONTROLS",
                    "confidence": 0.9,
                    "properties": {{"持股比例": "60%"}}
                }}
            ]
        }}
        """

        try:
            response = await self.llm.invoke(prompt, json_mode=True, temperature=0)  # 调用LLM提取信息
            result = json.loads(response)  # 解析JSON结果

            entities = []
            for entity_data in result.get("entities", []):
                entities.append(Entity(
                    name=entity_data["name"],
                    type=entity_data["type"],
                    confidence=entity_data.get("confidence", 0.8),
                    properties=entity_data.get("properties", {})
                ))

            relationships = []
            for rel_data in result.get("relationships", []):
                rel = Relationship(
                    source=rel_data["source"],
                    target=rel_data["target"],
                    type=rel_data["type"],
                    confidence=rel_data.get("confidence", 0.8),
                    evidence_source=doc_source,
                    properties=rel_data.get("properties", {})
                )
                relationships.append(rel)

            logger.info(f"✅ Extracted {len(entities)} entities, {len(relationships)} relationships")
            return entities, relationships
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return [], []

    async def build_graph_with_validation(self, documents: List[str]):
        """构建图谱并进行一致性验证

        Args:
            documents: 文档列表

        Returns:
            (警告信息, 证据链)
        """
        all_entities = []
        all_relationships = []
        valid_relationships = []

        # 从所有文档中提取实体和关系
        for idx, doc in enumerate(documents):
            entities, relationships = await self.extract_entities_and_relationships(doc, f"文档_{idx+1}")
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # 写入实体
        await self._write_entities(all_entities)

        # 验证并写入关系
        for rel in all_relationships:
            # 检查关系一致性
            consistent, msg, cons_conf = self.consistency_checker.check_relationship_consistency(rel.source, rel.target, rel.type)
            # 检查循环关系
            acyclic, cyc_msg, cyc_conf = self.consistency_checker.check_cyclic_relationship(rel.source, rel.target)
            # 文档交叉验证
            doc_valid, doc_conf, evidence = self.consistency_checker.cross_validate_with_documents(rel, documents)

            # 计算最终置信度
            final_confidence = (rel.confidence * 0.3 + cons_conf * 0.3 + cyc_conf * 0.2 + doc_conf * 0.2)

            # 只保留一致、无循环且置信度高的关系
            if consistent and acyclic and final_confidence > 0.5:
                rel.confidence = final_confidence
                if evidence:
                    self.evidence_chain.extend(evidence)
                valid_relationships.append(rel)
                logger.info(f"✅ Accepted relationship: {rel.source}->{rel.target} (confidence: {final_confidence:.2f})")
            else:
                warning = f"❌ Rejected relationship: {rel.source}->{rel.target} - consistent:{consistent}, acyclic:{acyclic}, confidence:{final_confidence:.2f}"
                self.consistency_checker.warnings.append(warning)
                logger.warning(warning)

        # 写入验证通过的关系
        await self._write_relationships(valid_relationships)
        return self.consistency_checker.get_warnings(), self.evidence_chain

    async def _write_entities(self, entities: List[Entity]):
        """写入实体到Neo4j

        Args:
            entities: 实体列表
        """
        if not entities:
            return

        query = """
        UNWIND $entities AS entity
        MERGE (n:Company {name: entity.name})
        SET n += entity.properties, n.confidence = entity.confidence
        RETURN count(n) as created
        """
        entities_data = [{"name": e.name, "confidence": e.confidence, "properties": e.properties} for e in entities]

        with self.driver.session() as session:
            result = session.run(query, entities=entities_data)
            count = result.single()["created"]
            logger.info(f"✅ Written {count} entities")

    async def _write_relationships(self, relationships: List[Relationship]):
        """写入关系到Neo4j

        Args:
            relationships: 关系列表
        """
        if not relationships:
            return

        query = """
        UNWIND $rels AS rel
        MATCH (source:Company {name: rel.source})
        MATCH (target:Company {name: rel.target})
        MERGE (source)-[r:CONTROLS]->(target)
        SET r += rel.properties, r.confidence = rel.confidence, r.evidence_source = rel.evidence_source
        RETURN count(r) as created
        """
        rels_data = [{
            "source": r.source,
            "target": r.target,
            "confidence": r.confidence,
            "evidence_source": r.evidence_source,
            "properties": r.properties
        } for r in relationships]

        with self.driver.session() as session:
            result = session.run(query, rels=rels_data)
            count = result.single()["created"]
            logger.info(f"✅ Written {count} relationships")

class HybridScoringEngine:
    """混合评分引擎，融合RAG和图谱推理结果"""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """初始化评分引擎

        Args:
            alpha: RAG权重
            beta: 图谱推理权重
        """
        self.alpha = alpha  # RAG权重
        self.beta = beta  # 图谱推理权重

    def calculate_combined_score(self, rag_score: float, kg_score: float,
                                   rag_confidence: float = 1.0, kg_confidence: float = 1.0) -> Tuple[float, str, Dict[str, float]]:
        """计算综合评分

        Args:
            rag_score: RAG评分
            kg_score: 图谱评分
            rag_confidence: RAG置信度
            kg_confidence: 图谱置信度

        Returns:
            (综合评分, 信息来源类型, 评分详情)
        """
        # 计算调整后的评分
        adjusted_rag = rag_score * rag_confidence
        adjusted_kg = kg_score * kg_confidence

        # 动态调整权重
        if adjusted_rag < 0.2 and adjusted_kg > 0.5:
            # RAG评分低，图谱评分高，增加图谱权重
            weight_rag = 0.2
            weight_kg = 0.8
        elif adjusted_kg < 0.2 and adjusted_rag > 0.5:
            # 图谱评分低，RAG评分高，增加RAG权重
            weight_rag = 0.8
            weight_kg = 0.2
        else:
            # 使用默认权重
            weight_rag = self.alpha
            weight_kg = self.beta

        # 计算综合评分
        combined_score = weight_rag * adjusted_rag + weight_kg * adjusted_kg

        # 判断信息来源类型
        if adjusted_rag > 0.3 and adjusted_kg > 0.3:
            source_type = "混合"
        elif adjusted_rag > adjusted_kg:
            source_type = "文档检索"
        else:
            source_type = "图谱推理"

        # 构建评分详情
        details = {
            "rag_score": rag_score,
            "kg_score": kg_score,
            "rag_confidence": rag_confidence,
            "kg_confidence": kg_confidence,
            "adjusted_rag": adjusted_rag,
            "adjusted_kg": adjusted_kg,
            "weight_rag": weight_rag,
            "weight_kg": weight_kg
        }

        return combined_score, source_type, details

class MultiHopQueryEngine:
    """多跳查询引擎，实现多层级推理"""

    def __init__(self, driver, llm: SimpleLLM, doc_retriever: DocumentRetriever,
                 scoring_engine: HybridScoringEngine):
        """初始化多跳查询引擎

        Args:
            driver: Neo4j驱动
            llm: LLM实例
            doc_retriever: 文档检索器
            scoring_engine: 评分引擎
        """
        self.driver = driver  # Neo4j驱动
        self.llm = llm  # LLM实例
        self.doc_retriever = doc_retriever  # 文档检索器
        self.scoring_engine = scoring_engine  # 评分引擎
        self.cypher_queries = []  # Cypher查询
        self.warnings = []  # 警告信息

    async def query(self, question: str) -> QueryResult:
        """执行多跳查询

        Args:
            question: 查询问题

        Returns:
            查询结果
        """
        reasoning_steps = []  # 推理步骤
        self.cypher_queries = []  # 重置Cypher查询
        self.warnings = []  # 重置警告信息
        evidence_chain = []  # 证据链

        # 文档检索
        rag_results, rag_score = self.doc_retriever.retrieve_with_source(question)
        rag_docs = [r[0] for r in rag_results]  # 文档文本
        rag_sources = [r[1] for r in rag_results]  # 文档来源

        # 记录文档检索步骤
        source_label = "LlamaIndex文档检索" if not self.doc_retriever.use_fallback else "简单文档检索"
        reasoning_steps.append(ReasoningStep(
            step=1,
            query=question,
            source=source_label,
            result=f"找到 {len(rag_docs)} 个相关文档，相似度: {rag_score:.3f}",
            confidence=rag_score,
            evidence=", ".join(rag_sources)
        ))

        # 多跳图谱查询
        kg_result, kg_score, kg_confidence, kg_steps, kg_evidence = await self._multi_hop_graph_query(question)
        reasoning_steps.extend(kg_steps)  # 添加图谱推理步骤
        evidence_chain.extend(kg_evidence)  # 添加证据

        # 计算综合评分
        combined_score, source_type, score_details = self.scoring_engine.calculate_combined_score(
            rag_score, kg_score, 1.0, kg_confidence
        )

        # 计算整体置信度
        overall_confidence = min(1.0, (rag_score + kg_score + kg_confidence) / 3)

        # 生成答案
        answer = await self._synthesize_answer(question, rag_docs, kg_result, score_details)

        # 返回查询结果
        return QueryResult(
            answer=answer,
            rag_score=rag_score,
            kg_score=kg_score,
            combined_score=combined_score,
            confidence_score=overall_confidence,
            reasoning_path=reasoning_steps,
            source_type=source_type,
            cypher_queries=self.cypher_queries,
            warnings=self.warnings,
            evidence_chain=evidence_chain
        )

    async def _multi_hop_graph_query(self, question: str) -> Tuple[str, float, float, List[ReasoningStep], List[str]]:
        """执行多跳图谱查询

        Args:
            question: 查询问题

        Returns:
            (查询结果, 图谱评分, 平均置信度, 推理步骤, 证据链)
        """
        steps = []  # 推理步骤
        all_results = []  # 所有结果
        evidence_chain = []  # 证据链
        total_confidence = 0.0  # 总置信度
        hop_count = 0  # 跳数

        # 从问题中提取实体
        extracted_entities = await self._extract_entities_from_question(question)
        steps.append(ReasoningStep(
            step=2,
            query="实体提取",
            source="LLM",
            result=f"提取到实体: {extracted_entities}",
            confidence=0.9
        ))

        current_entities = extracted_entities[:2]  # 最多处理前2个实体

        # 执行1-2跳查询
        for hop_num in range(1, 3):
            if not current_entities:
                break

            hop_results = []  # 单跳结果
            hop_query = ""  # 单跳查询

            # 对每个实体执行查询
            for entity in current_entities:
                result, query, conf, evidence = await self._execute_hop(entity, hop_num)
                if result:
                    hop_results.append(result)
                    hop_query = query
                    evidence_chain.extend(evidence)
                    total_confidence += conf
                    hop_count += 1

            # 处理单跳结果
            if hop_results:
                self.cypher_queries.append(hop_query)  # 保存Cypher查询
                combined_result = "\n".join(hop_results)  # 合并结果
                all_results.append(combined_result)  # 添加到所有结果

                # 记录推理步骤
                steps.append(ReasoningStep(
                    step=2 + hop_num,
                    query=hop_query,
                    source="Cypher",
                    result=combined_result[:200],
                    confidence=conf if hop_count > 0 else 0
                ))

                # 从结果中提取新的实体，用于下一跳查询
                current_entities = self._parse_entities_from_result(combined_result)[:2]
            else:
                break

        # 合并所有图谱结果
        combined_kg_result = "\n".join(filter(None, all_results))
        # 计算图谱评分
        kg_score = 0.9 if combined_kg_result else 0.3
        # 计算平均置信度
        avg_confidence = total_confidence / hop_count if hop_count > 0 else 0.3

        return combined_kg_result, kg_score, avg_confidence, steps, evidence_chain

    async def _extract_entities_from_question(self, question: str) -> List[str]:
        """从问题中提取实体

        Args:
            question: 查询问题

        Returns:
            实体列表
        """
        prompt = f"""
        从问题中提取公司名称，返回JSON格式：
        问题: {question}
        返回: {"entities": ["A公司", "B公司"]}
        """
        try:
            response = await self.llm.invoke(prompt, json_mode=True, temperature=0)  # 调用LLM提取实体
            result = json.loads(response)  # 解析JSON结果
            return result.get("entities", [])
        except:
            # 失败时使用正则表达式提取
            matches = re.findall(r'([A-Z]公司)', question)
            return matches

    async def _execute_hop(self, entity_name: str, hop_num: int) -> Tuple[str, str, float, List[str]]:
        """执行单跳查询

        Args:
            entity_name: 实体名称
            hop_num: 跳数

        Returns:
            (查询结果, 查询语句, 置信度, 证据)
        """
        # Cypher查询，查找1-2跳控股关系
        query = """
        MATCH path = (c:Company {name: $name})-[:CONTROLS*1..2]->(sub:Company)
        WITH path, sub, length(path) as depth,
             [node in nodes(path) | node.name] as path_names,
             [rel in relationships(path) | rel.confidence] as confidences
        RETURN sub.name as name, depth, path_names as path,
               reduce(total = 1.0, conf in confidences | total * coalesce(conf, 0.8)) as rel_confidence
        ORDER BY depth, name
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, name=entity_name)  # 执行Cypher查询
                records = list(result)  # 获取所有记录

                if records:
                    parts = []  # 结果部分
                    confidences = []  # 置信度列表
                    evidence = []  # 证据列表
                    for record in records:
                        path_str = " → ".join(record['path'])  # 构建路径字符串
                        rel_conf = record.get('rel_confidence', 0.8)  # 获取关系置信度
                        parts.append(f"{record['name']} (第{record['depth']}层: {path_str}, 置信度:{rel_conf:.2f})")
                        confidences.append(rel_conf)
                        evidence.append(path_str)

                    avg_conf = sum(confidences) / len(confidences)  # 计算平均置信度
                    return "\n".join(parts), query, avg_conf, evidence
                else:
                    return "", query, 0.0, []
        except Exception as e:
            logger.error(f"❌ Cypher query failed: {e}")
            self.warnings.append(f"Cypher查询失败: {e}")
            return "", query, 0.0, []

    def _parse_entities_from_result(self, result: str) -> List[str]:
        """从结果中解析实体

        Args:
            result: 查询结果

        Returns:
            实体列表
        """
        entities = re.findall(r'([A-Z]公司)', result)  # 提取公司名称
        return list(set(entities))[:3]  # 去重并最多返回3个

    async def _synthesize_answer(self, question: str, rag_docs: List[str], kg_result: str, score_details: Dict[str, float]) -> str:
        """生成最终答案

        Args:
            question: 查询问题
            rag_docs: 文档检索结果
            kg_result: 图谱推理结果
            score_details: 评分详情

        Returns:
            最终答案
        """
        prompt = f"""
        基于文档检索和图谱推理的结果回答问题：

        问题: {question}

        文档检索结果:
        {chr(10).join(rag_docs)}

        图谱推理结果:
        {kg_result}

        评分详情:
        - RAG评分: {score_details['rag_score']:.3f}
        - KG评分: {score_details['kg_score']:.3f}
        - 调整后RAG: {score_details['adjusted_rag']:.3f}
        - 调整后KG: {score_details['adjusted_kg']:.3f}
        - RAG权重: {score_details['weight_rag']:.2f}
        - KG权重: {score_details['weight_kg']:.2f}

        请准确、简洁地回答问题，使用中文，并说明信息来源。
        """

        try:
            response = await self.llm.invoke(prompt, temperature=0)  # 调用LLM生成答案
            return response.strip()
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return "抱歉，生成答案时出现错误"

class EnterpriseGraphRAG:
    """企业股权图谱 RAG 系统，整合所有组件"""

    def __init__(self, driver, llm: SimpleLLM, documents: List[str], alpha: float = 0.5, beta: float = 0.5):
        """初始化企业股权图谱 RAG 系统

        Args:
            driver: Neo4j驱动
            llm: LLM实例
            documents: 文档列表
            alpha: RAG权重
            beta: 图谱推理权重
        """
        self.driver = driver  # Neo4j驱动
        self.llm = llm  # LLM实例
        self.documents = documents  # 文档列表

        # 初始化各组件
        self.graph_builder = GraphBuilder(driver, llm, documents)  # 图谱构建器
        self.doc_retriever = DocumentRetriever(documents)  # 文档检索器
        self.scoring_engine = HybridScoringEngine(alpha, beta)  # 评分引擎
        self.query_engine = MultiHopQueryEngine(driver, llm, self.doc_retriever, self.scoring_engine)  # 多跳查询引擎

        logger.info("✅ 企业股权图谱 RAG 系统初始化完成")

    async def build_graph(self):
        """构建企业股权图谱

        Returns:
            (警告信息, 证据链)
        """
        logger.info("=" * 70)
        logger.info("开始构建企业股权图谱（含一致性校验）")

        # 构建图谱并进行一致性验证
        warnings, evidence_chain = await self.graph_builder.build_graph_with_validation(self.documents)

        # 输出警告信息
        if warnings:
            logger.warning(f"构建完成，有 {len(warnings)} 个警告:")
            for w in warnings[:5]:
                logger.warning(f"  - {w}")

        logger.info("=" * 70)
        return warnings, evidence_chain

    async def query(self, question: str) -> QueryResult:
        """执行多跳问答

        Args:
            question: 查询问题

        Returns:
            查询结果
        """
        logger.info("=" * 70)
        logger.info(f"查询: {question}")

        # 执行查询
        result = await self.query_engine.query(question)

        # 输出查询结果
        logger.info(f"RAG 评分: {result.rag_score:.3f}")
        logger.info(f"KG 评分: {result.kg_score:.3f}")
        logger.info(f"综合评分: {result.combined_score:.3f}")
        logger.info(f"整体置信度: {result.confidence_score:.3f}")
        logger.info(f"信息来源: {result.source_type}")

        # 输出推理路径
        logger.info("\n推理路径:")
        for step in result.reasoning_path:
            logger.info(f"  [步骤 {step.step}] {step.source}: {step.query[:80]}...")
            logger.info(f"    结果: {step.result[:100]}")
            logger.info(f"    置信度: {step.confidence:.2f}")
            if step.evidence:
                logger.info(f"    证据: {step.evidence}")

        # 输出Cypher查询
        if result.cypher_queries:
            logger.info("\nCypher 查询:")
            for i, q in enumerate(result.cypher_queries, 1):
                logger.info(f"  查询 {i}: {q.strip()[:120]}...")

        # 输出警告
        if result.warnings:
            logger.warning("\n警告:")
            for w in result.warnings:
                logger.warning(f"  - {w}")

        # 输出证据链
        if result.evidence_chain:
            logger.info("\n证据链:")
            for e in result.evidence_chain[:5]:
                logger.info(f"  - {e}")

        logger.info("=" * 70)
        return result

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

async def demo():
    """演示函数，展示系统功能"""
    print("=" * 70)
    print("企业股权图谱多跳问答系统 - LlamaIndex 集成版")
    print("=" * 70)
    print("\n核心技术实现:")
    print("  1. LlamaIndex 文档检索 - 向量索引（可用时）")
    print("  2. RAG与图谱推理融合 - 双路检索 + 动态权重")
    print("  3. 联合评分机制 - 自适应权重 + 置信度调整")
    print("  4. 防止错误传播 - 一致性校验 + 交叉验证 + 溯源")
    print("=" * 70)

    # 示例财报文档
    financial_documents = [
        """A公司是一家大型集团公司，2023年财报显示其营业收入达500亿元。
        A公司控股B公司，持股比例为60%。
        A公司还控股D公司，持股比例为55%。""",

        """B公司2023年财报显示，其主要业务为电子产品制造，
        年营业收入为150亿元，净利润为20亿元。
        B公司控股C公司，持股比例为70%。
        B公司控股E公司，持股比例为80%。""",

        """C公司专注于芯片设计，在行业内具有领先地位。
        C公司控股F公司，持股比例为65%。
        F公司主要负责芯片的生产制造。""",

        """D公司主要从事新能源业务，2023年营收80亿元。
        D公司控股G公司，持股比例为75%。
        G公司在光伏领域有重要布局。""",

        """E公司是B公司的全资子公司，专注于智能硬件研发。
        2023年E公司推出了多款创新产品，市场反响良好。""",

        """G公司2023年财报显示，其光伏组件产量位居全国前列。
        D公司通过控股G公司，成功进入新能源产业链的核心环节。
        A公司的多元化战略取得了显著成效。"""
    ]

    try:
        print("\n[1] 连接 Neo4j 数据库...")
        driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))  # 连接Neo4j数据库

        print("\n[2] 初始化 LLM 模型...")
        llm = SimpleLLM(model_name="qwen-plus")  # 初始化LLM模型

        print("\n[3] 创建企业图谱 RAG 系统（集成 LlamaIndex）...")
        graph_rag = EnterpriseGraphRAG(
            driver=driver,
            llm=llm,
            documents=financial_documents,
            alpha=0.5,
            beta=0.5
        )  # 创建企业图谱RAG系统

        print("\n[4] 构建企业股权图谱（含一致性校验）...")
        warnings, evidence = await graph_rag.build_graph()  # 构建企业股权图谱

        print("\n[5] 多跳问答测试...")
        print("=" * 70)

        # 测试问题
        test_questions = [
            "A公司的子公司有哪些？",
            "B公司的业务是什么？",
            "A公司有多少层控股关系？请列出所有层级的公司。",
            "G公司属于哪个公司？它的业务是什么？",
            "E公司是做什么的？它的母公司是谁？",
            "请详细描述A公司的整个控股架构，包括所有层级的子公司。"
        ]

        # 执行测试
        for question in test_questions:
            print(f"\n{'='*70}")
            print(f"问题: {question}")
            print(f"{'='*70}")

            result = await graph_rag.query(question)  # 执行查询

            # 输出结果
            print(f"\n答案: {result.answer}")
            print(f"\n综合评分: {result.combined_score:.3f}")
            print(f"整体置信度: {result.confidence_score:.3f}")
            print(f"信息来源: {result.source_type}")

            # 输出警告
            if result.warnings:
                print(f"\n⚠️  警告: {len(result.warnings)} 个")
                for w in result.warnings[:3]:
                    print(f"   - {w}")

        print("\n" + "=" * 70)
        print("演示完成！")

        graph_rag.close()  # 关闭数据库连接

    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数，启动演示"""
    asyncio.run(demo())  # 运行演示函数

if __name__ == "__main__":
    main()  # 执行主函数
