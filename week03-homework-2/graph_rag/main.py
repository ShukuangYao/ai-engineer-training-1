import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase
from openai import AsyncOpenAI
import nest_asyncio
import re

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LLAMA_INDEX_AVAILABLE = False
try:
    from llama_index import Document, VectorStoreIndex, ServiceContext
    LLAMA_INDEX_AVAILABLE = True
    logger.info("✅ Successfully imported LlamaIndex")
except ImportError as e:
    logger.warning(f"⚠️ LlamaIndex not available: {e}")
    logger.warning("⚠️ Using keyword matching fallback")

@dataclass
class Entity:
    name: str
    type: str
    confidence: float = 1.0
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class Relationship:
    source: str
    target: str
    type: str
    confidence: float = 1.0
    evidence_source: str = ""
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class ReasoningStep:
    step: int
    query: str
    source: str
    result: str
    confidence: float = 1.0
    evidence: str = ""

@dataclass
class QueryResult:
    answer: str
    rag_score: float
    kg_score: float
    combined_score: float
    confidence_score: float
    reasoning_path: List[ReasoningStep]
    source_type: str
    cypher_queries: List[str]
    warnings: List[str]
    evidence_chain: List[str]

class SimpleLLM:
    def __init__(self, model_name: str = "qwen-plus", api_key: str = None, base_url: str = None):
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_name

    async def invoke(self, prompt: str, json_mode: bool = False, temperature: float = 0):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

class DocumentRetriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.index = None
        self.query_engine = None
        self.use_fallback = not LLAMA_INDEX_AVAILABLE
        self.doc_source_map = {i: f"文档_{i+1}" for i in range(len(documents))}

        if LLAMA_INDEX_AVAILABLE:
            self._build_index()
        if self.index is None:
            self.use_fallback = True
            logger.warning("⚠️ Falling back to keyword matching")

    def _build_index(self):
        try:
            logger.info("🔨 Building LlamaIndex index...")

            llama_docs = [Document(text=doc, doc_id=f"doc_{i}") for i, doc in enumerate(self.documents)]
            logger.info(f"✅ Created {len(llama_docs)} documents")

            service_context = ServiceContext.from_defaults(llm=None)
            logger.info("✅ Service context created")

            self.index = VectorStoreIndex.from_documents(
                llama_docs,
                service_context=service_context
            )
            logger.info("✅ VectorStoreIndex built")

            self.query_engine = self.index.as_retriever(similarity_top_k=3)
            self.use_fallback = False
            logger.info("✅ LlamaIndex index built successfully!")
        except Exception as e:
            logger.error(f"❌ LlamaIndex index build failed: {e}")
            self.index = None
            self.use_fallback = True

    def retrieve_with_source(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[str, str, float]], float]:
        if not self.use_fallback and self.index is not None:
            return self._llama_retrieve(query, top_k)
        else:
            return self._fallback_retrieve(query, top_k)

    def _llama_retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[str, str, float]], float]:
        try:
            nodes = self.query_engine.retrieve(query)

            results = []
            scores = []
            for node in nodes[:top_k]:
                doc_text = node.text
                doc_source = node.metadata.get("doc_id", "unknown")
                score = float(node.score)
                results.append((doc_text, doc_source, score))
                scores.append(score)

            avg_score = float(sum(scores) / len(scores)) if scores else 0.0
            logger.info(f"✅ LlamaIndex retrieved {len(results)} docs, avg score: {avg_score:.3f}")
            return results, avg_score
        except Exception as e:
            logger.error(f"❌ LlamaIndex retrieval failed: {e}, using fallback")
            return self._fallback_retrieve(query, top_k)

    def _fallback_retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Tuple[str, str, float]], float]:
        query_words = set(re.findall(r'[\w\u4e00-\u9fff]+', query.lower()))

        scores = []
        for i, doc in enumerate(self.documents):
            doc_words = set(re.findall(r'[\w\u4e00-\u9fff]+', doc.lower()))
            overlap = len(query_words & doc_words)
            score = min(1.0, overlap / max(len(query_words), 1))
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:top_k]]

        results = []
        for i in top_indices:
            results.append((self.documents[i], self.doc_source_map[i], float(scores[top_indices.index(i)][1])))

        avg_score = float(sum(s for _, s in scores[:top_k]) / len(scores[:top_k])) if scores[:top_k] else 0.0
        return results, avg_score

class ConsistencyChecker:
    def __init__(self, driver):
        self.driver = driver
        self.warnings = []

    def check_relationship_consistency(self, source: str, target: str, rel_type: str) -> Tuple[bool, str, float]:
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
        evidence = []
        for doc in documents:
            source_pattern = re.escape(relationship.source)
            target_pattern = re.escape(relationship.target)

            if source_pattern in doc and target_pattern in doc:
                if "控股" in doc or "持股" in doc:
                    evidence.append(doc[:200])

        if evidence:
            confidence = min(0.95, 0.7 + 0.1 * len(evidence))
            return True, confidence, evidence
        else:
            return False, 0.3, []

    def get_warnings(self) -> List[str]:
        return self.warnings

class GraphBuilder:
    def __init__(self, driver, llm: SimpleLLM, documents: List[str]):
        self.driver = driver
        self.llm = llm
        self.documents = documents
        self.consistency_checker = ConsistencyChecker(driver)
        self.evidence_chain = []

    async def extract_entities_and_relationships(self, text: str, doc_source: str = "") -> Tuple[List[Entity], List[Relationship]]:
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
            response = await self.llm.invoke(prompt, json_mode=True, temperature=0)
            result = json.loads(response)

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
        all_entities = []
        all_relationships = []
        valid_relationships = []

        for idx, doc in enumerate(documents):
            entities, relationships = await self.extract_entities_and_relationships(doc, f"文档_{idx+1}")
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        await self._write_entities(all_entities)

        for rel in all_relationships:
            consistent, msg, cons_conf = self.consistency_checker.check_relationship_consistency(rel.source, rel.target, rel.type)
            acyclic, cyc_msg, cyc_conf = self.consistency_checker.check_cyclic_relationship(rel.source, rel.target)
            doc_valid, doc_conf, evidence = self.consistency_checker.cross_validate_with_documents(rel, documents)

            final_confidence = (rel.confidence * 0.3 + cons_conf * 0.3 + cyc_conf * 0.2 + doc_conf * 0.2)

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

        await self._write_relationships(valid_relationships)
        return self.consistency_checker.get_warnings(), self.evidence_chain

    async def _write_entities(self, entities: List[Entity]):
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
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def calculate_combined_score(self, rag_score: float, kg_score: float,
                                   rag_confidence: float = 1.0, kg_confidence: float = 1.0) -> Tuple[float, str, Dict[str, float]]:
        adjusted_rag = rag_score * rag_confidence
        adjusted_kg = kg_score * kg_confidence

        if adjusted_rag < 0.2 and adjusted_kg > 0.5:
            weight_rag = 0.2
            weight_kg = 0.8
        elif adjusted_kg < 0.2 and adjusted_rag > 0.5:
            weight_rag = 0.8
            weight_kg = 0.2
        else:
            weight_rag = self.alpha
            weight_kg = self.beta

        combined_score = weight_rag * adjusted_rag + weight_kg * adjusted_kg

        if adjusted_rag > 0.3 and adjusted_kg > 0.3:
            source_type = "混合"
        elif adjusted_rag > adjusted_kg:
            source_type = "文档检索"
        else:
            source_type = "图谱推理"

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
    def __init__(self, driver, llm: SimpleLLM, doc_retriever: DocumentRetriever,
                 scoring_engine: HybridScoringEngine):
        self.driver = driver
        self.llm = llm
        self.doc_retriever = doc_retriever
        self.scoring_engine = scoring_engine
        self.cypher_queries = []
        self.warnings = []

    async def query(self, question: str) -> QueryResult:
        reasoning_steps = []
        self.cypher_queries = []
        self.warnings = []
        evidence_chain = []

        rag_results, rag_score = self.doc_retriever.retrieve_with_source(question)
        rag_docs = [r[0] for r in rag_results]
        rag_sources = [r[1] for r in rag_results]

        source_label = "LlamaIndex文档检索" if not self.doc_retriever.use_fallback else "简单文档检索"
        reasoning_steps.append(ReasoningStep(
            step=1,
            query=question,
            source=source_label,
            result=f"找到 {len(rag_docs)} 个相关文档，相似度: {rag_score:.3f}",
            confidence=rag_score,
            evidence=", ".join(rag_sources)
        ))

        kg_result, kg_score, kg_confidence, kg_steps, kg_evidence = await self._multi_hop_graph_query(question)
        reasoning_steps.extend(kg_steps)
        evidence_chain.extend(kg_evidence)

        combined_score, source_type, score_details = self.scoring_engine.calculate_combined_score(
            rag_score, kg_score, 1.0, kg_confidence
        )

        overall_confidence = min(1.0, (rag_score + kg_score + kg_confidence) / 3)

        answer = await self._synthesize_answer(question, rag_docs, kg_result, score_details)

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
        steps = []
        all_results = []
        evidence_chain = []
        total_confidence = 0.0
        hop_count = 0

        extracted_entities = await self._extract_entities_from_question(question)
        steps.append(ReasoningStep(
            step=2,
            query="实体提取",
            source="LLM",
            result=f"提取到实体: {extracted_entities}",
            confidence=0.9
        ))

        current_entities = extracted_entities[:2]

        for hop_num in range(1, 3):
            if not current_entities:
                break

            hop_results = []
            hop_query = ""

            for entity in current_entities:
                result, query, conf, evidence = await self._execute_hop(entity, hop_num)
                if result:
                    hop_results.append(result)
                    hop_query = query
                    evidence_chain.extend(evidence)
                    total_confidence += conf
                    hop_count += 1

            if hop_results:
                self.cypher_queries.append(hop_query)
                combined_result = "\n".join(hop_results)
                all_results.append(combined_result)
                steps.append(ReasoningStep(
                    step=2 + hop_num,
                    query=hop_query,
                    source="Cypher",
                    result=combined_result[:200],
                    confidence=conf if hop_count > 0 else 0
                ))

                current_entities = self._parse_entities_from_result(combined_result)[:2]
            else:
                break

        combined_kg_result = "\n".join(filter(None, all_results))
        kg_score = 0.9 if combined_kg_result else 0.3
        avg_confidence = total_confidence / hop_count if hop_count > 0 else 0.3

        return combined_kg_result, kg_score, avg_confidence, steps, evidence_chain

    async def _extract_entities_from_question(self, question: str) -> List[str]:
        prompt = f"""
        从问题中提取公司名称，返回JSON格式：
        问题: {question}
        返回: {{"entities": ["A公司", "B公司"]}}
        """
        try:
            response = await self.llm.invoke(prompt, json_mode=True, temperature=0)
            result = json.loads(response)
            return result.get("entities", [])
        except:
            matches = re.findall(r'([A-Z]公司)', question)
            return matches

    async def _execute_hop(self, entity_name: str, hop_num: int) -> Tuple[str, str, float, List[str]]:
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
                result = session.run(query, name=entity_name)
                records = list(result)

                if records:
                    parts = []
                    confidences = []
                    evidence = []
                    for record in records:
                        path_str = " → ".join(record['path'])
                        rel_conf = record.get('rel_confidence', 0.8)
                        parts.append(f"{record['name']} (第{record['depth']}层: {path_str}, 置信度:{rel_conf:.2f})")
                        confidences.append(rel_conf)
                        evidence.append(path_str)

                    avg_conf = sum(confidences) / len(confidences)
                    return "\n".join(parts), query, avg_conf, evidence
                else:
                    return "", query, 0.0, []
        except Exception as e:
            logger.error(f"❌ Cypher query failed: {e}")
            self.warnings.append(f"Cypher查询失败: {e}")
            return "", query, 0.0, []

    def _parse_entities_from_result(self, result: str) -> List[str]:
        entities = re.findall(r'([A-Z]公司)', result)
        return list(set(entities))[:3]

    async def _synthesize_answer(self, question: str, rag_docs: List[str], kg_result: str, score_details: Dict[str, float]) -> str:
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
            response = await self.llm.invoke(prompt, temperature=0)
            return response.strip()
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return "抱歉，生成答案时出现错误"

class EnterpriseGraphRAG:
    def __init__(self, driver, llm: SimpleLLM, documents: List[str], alpha: float = 0.5, beta: float = 0.5):
        self.driver = driver
        self.llm = llm
        self.documents = documents

        self.graph_builder = GraphBuilder(driver, llm, documents)
        self.doc_retriever = DocumentRetriever(documents)
        self.scoring_engine = HybridScoringEngine(alpha, beta)
        self.query_engine = MultiHopQueryEngine(driver, llm, self.doc_retriever, self.scoring_engine)

        logger.info("✅ 企业股权图谱 RAG 系统初始化完成")

    async def build_graph(self):
        logger.info("=" * 70)
        logger.info("开始构建企业股权图谱（含一致性校验）")

        warnings, evidence_chain = await self.graph_builder.build_graph_with_validation(self.documents)

        if warnings:
            logger.warning(f"构建完成，有 {len(warnings)} 个警告:")
            for w in warnings[:5]:
                logger.warning(f"  - {w}")

        logger.info("=" * 70)
        return warnings, evidence_chain

    async def query(self, question: str) -> QueryResult:
        logger.info("=" * 70)
        logger.info(f"查询: {question}")

        result = await self.query_engine.query(question)

        logger.info(f"RAG 评分: {result.rag_score:.3f}")
        logger.info(f"KG 评分: {result.kg_score:.3f}")
        logger.info(f"综合评分: {result.combined_score:.3f}")
        logger.info(f"整体置信度: {result.confidence_score:.3f}")
        logger.info(f"信息来源: {result.source_type}")

        logger.info("\n推理路径:")
        for step in result.reasoning_path:
            logger.info(f"  [步骤 {step.step}] {step.source}: {step.query[:80]}...")
            logger.info(f"    结果: {step.result[:100]}")
            logger.info(f"    置信度: {step.confidence:.2f}")
            if step.evidence:
                logger.info(f"    证据: {step.evidence}")

        if result.cypher_queries:
            logger.info("\nCypher 查询:")
            for i, q in enumerate(result.cypher_queries, 1):
                logger.info(f"  查询 {i}: {q.strip()[:120]}...")

        if result.warnings:
            logger.warning("\n警告:")
            for w in result.warnings:
                logger.warning(f"  - {w}")

        if result.evidence_chain:
            logger.info("\n证据链:")
            for e in result.evidence_chain[:5]:
                logger.info(f"  - {e}")

        logger.info("=" * 70)
        return result

    def close(self):
        self.driver.close()

async def demo():
    print("=" * 70)
    print("企业股权图谱多跳问答系统 - LlamaIndex 集成版")
    print("=" * 70)
    print("\n核心技术实现:")
    print("  1. LlamaIndex 文档检索 - 向量索引（可用时）")
    print("  2. RAG与图谱推理融合 - 双路检索 + 动态权重")
    print("  3. 联合评分机制 - 自适应权重 + 置信度调整")
    print("  4. 防止错误传播 - 一致性校验 + 交叉验证 + 溯源")
    print("=" * 70)

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
        driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))

        print("\n[2] 初始化 LLM 模型...")
        llm = SimpleLLM(model_name="qwen-plus")

        print("\n[3] 创建企业图谱 RAG 系统（集成 LlamaIndex）...")
        graph_rag = EnterpriseGraphRAG(
            driver=driver,
            llm=llm,
            documents=financial_documents,
            alpha=0.5,
            beta=0.5
        )

        print("\n[4] 构建企业股权图谱（含一致性校验）...")
        warnings, evidence = await graph_rag.build_graph()

        print("\n[5] 多跳问答测试...")
        print("=" * 70)

        test_questions = [
            "A公司的子公司有哪些？",
            "B公司的业务是什么？",
            "A公司有多少层控股关系？请列出所有层级的公司。",
            "G公司属于哪个公司？它的业务是什么？",
            "E公司是做什么的？它的母公司是谁？",
            "请详细描述A公司的整个控股架构，包括所有层级的子公司。"
        ]

        for question in test_questions:
            print(f"\n{'='*70}")
            print(f"问题: {question}")
            print(f"{'='*70}")

            result = await graph_rag.query(question)

            print(f"\n答案: {result.answer}")
            print(f"\n综合评分: {result.combined_score:.3f}")
            print(f"整体置信度: {result.confidence_score:.3f}")
            print(f"信息来源: {result.source_type}")

            if result.warnings:
                print(f"\n⚠️  警告: {len(result.warnings)} 个")
                for w in result.warnings[:3]:
                    print(f"   - {w}")

        print("\n" + "=" * 70)
        print("演示完成！")

        graph_rag.close()

    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    asyncio.run(demo())

if __name__ == "__main__":
    main()
