import os
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
    MarkdownNodeParser
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from pathlib import Path
from typing import List, Dict, Any

class AdaptiveChunkingStrategy:
    """自适应分块策略"""

    def __init__(self):
        self.strategies = {
            "token": TokenTextSplitter(
                chunk_size=128,
                chunk_overlap=16,
                separator="\n"
            ),
            "sentence": SentenceSplitter(
                chunk_size=512,  # chunk_size 表示字符数，不是句子数
                chunk_overlap=50  # 块之间的重叠字符数
            ),
            "sentence_window": SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            ),
            "markdown": MarkdownNodeParser()
        }

    def select_strategy(self, query_complexity, doc_type):
        """根据查询复杂度和文档类型选择策略"""
        if doc_type == "structured":
            return self.strategies["sentence_window"]
        elif query_complexity == "high":
            return self.strategies["sentence_window"]
        elif query_complexity == "medium":
            return self.strategies["sentence"]
        else:
            return self.strategies["token"]

    def get_node_parser(self, strategy_name: str):
        """根据策略名称获取对应的节点解析器"""
        if strategy_name not in self.strategies:
            raise ValueError(f"未知的策略: {strategy_name}。可用策略: {list(self.strategies.keys())}")
        return self.strategies[strategy_name]

    def create_query_engine(self, index, strategy_name: str, similarity_top_k: int = 5):
        """
        创建带后处理的查询引擎

        参数:
            index: 向量索引对象
            strategy_name: 策略名称 ("token", "sentence", "sentence_window", "markdown")
            similarity_top_k: 检索的top-k数量
        """
        if strategy_name == "sentence_window":
            # sentence_window 策略需要特殊的后处理器
            return index.as_query_engine(
                similarity_top_k=similarity_top_k,
                node_postprocessors=[
                    MetadataReplacementPostProcessor(
                        target_metadata_key="window"
                    )
                ]
            )
        else:
            # token、sentence 和 markdown 策略使用标准查询引擎
            return index.as_query_engine(
                similarity_top_k=similarity_top_k
            )

def evaluate_vector_search_precision(
    query_engine,
    query: str,
    ground_truth: str = None,
    top_k: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    评估向量查询的精准度

    参数:
        query_engine: 查询引擎对象
        query: 查询问题
        ground_truth: 标准答案（可选，用于评估检索质量）
        top_k: 检索的top-k数量
        verbose: 是否打印详细信息（默认True）

    返回:
        包含评估结果的字典
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"向量查询精准度评估")
        print(f"{'='*60}")
        print(f"查询问题: {query}")
        print(f"检索Top-K: {top_k}")

    # 1. 检索相关节点
    retriever = query_engine.retriever
    retrieved_nodes = retriever.retrieve(query)

    # 2. 提取检索结果信息
    retrieval_results = []
    for i, node in enumerate(retrieved_nodes[:top_k], 1):
        score = node.score if hasattr(node, 'score') and node.score is not None else 0.0
        content = node.get_content()[:200] + "..." if len(node.get_content()) > 200 else node.get_content()

        retrieval_results.append({
            "rank": i,
            "score": round(score, 4),
            "content_preview": content,
            "node_id": node.node_id if hasattr(node, 'node_id') else None
        })

        if verbose:
            print(f"\n--- 检索结果 {i} ---")
            print(f"相似度分数: {score:.4f}")
            print(f"内容预览: {content}")

    # 3. 计算统计信息
    if retrieved_nodes:
        scores = [node.score for node in retrieved_nodes[:top_k]
                 if hasattr(node, 'score') and node.score is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
        else:
            avg_score = max_score = min_score = 0.0
    else:
        avg_score = max_score = min_score = 0.0

    # 4. 评估检索质量（如果有标准答案）
    context_quality = {}
    if ground_truth:
        # 合并所有检索到的上下文
        retrieved_context = "\n\n".join([node.get_content() for node in retrieved_nodes[:top_k]])

        # 检查上下文是否包含标准答案的关键信息
        # 使用标准答案的前20个字符作为关键信息
        key_info = ground_truth[:20] if len(ground_truth) >= 20 else ground_truth
        context_contains_answer = key_info in retrieved_context

        context_quality = {
            "context_contains_answer": "是" if context_contains_answer else "否",
            "key_info": key_info,
            "retrieved_context_length": len(retrieved_context)
        }

        if verbose:
            print(f"\n--- 检索质量评估 ---")
            print(f"上下文是否包含答案关键信息: {context_quality['context_contains_answer']}")
            print(f"检索上下文总长度: {context_quality['retrieved_context_length']} 字符")

    # 5. 生成完整回答
    if verbose:
        print(f"\n--- 生成完整回答 ---")
    response = query_engine.query(query)
    generated_answer = str(response)
    if verbose:
        print(f"回答: {generated_answer[:300]}..." if len(generated_answer) > 300 else f"回答: {generated_answer}")

    # 6. 汇总评估结果
    evaluation_result = {
        "query": query,
        "top_k": top_k,
        "retrieval_count": len(retrieved_nodes),
        "retrieval_results": retrieval_results,
        "statistics": {
            "average_score": round(avg_score, 4),
            "max_score": round(max_score, 4),
            "min_score": round(min_score, 4),
            "score_range": round(max_score - min_score, 4) if scores else 0.0
        },
        "context_quality": context_quality,
        "generated_answer": generated_answer
    }

    if verbose:
        print(f"\n--- 统计信息 ---")
        print(f"平均相似度分数: {evaluation_result['statistics']['average_score']:.4f}")
        print(f"最高相似度分数: {evaluation_result['statistics']['max_score']:.4f}")
        print(f"最低相似度分数: {evaluation_result['statistics']['min_score']:.4f}")
        print(f"分数范围: {evaluation_result['statistics']['score_range']:.4f}")
        print(f"{'='*60}\n")

    return evaluation_result


def compare_strategies(
    documents,
    query: str,
    ground_truth: str,
    strategies: List[str] = None,
    similarity_top_k: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    对比评估多种切片策略

    参数:
        documents: 文档列表
        query: 查询问题
        ground_truth: 标准答案
        strategies: 要测试的策略列表，如果为None则测试所有策略
        similarity_top_k: 检索的top-k数量

    返回:
        包含所有策略评估结果的字典
    """
    strategy_manager = AdaptiveChunkingStrategy()

    # 如果没有指定策略，测试所有策略
    if strategies is None:
        strategies = list(strategy_manager.strategies.keys())

    print(f"\n{'#'*80}")
    print(f"开始对比评估 {len(strategies)} 种切片策略")
    print(f"策略列表: {strategies}")
    print(f"{'#'*80}\n")

    results = {}

    for strategy_name in strategies:
        print(f"\n{'='*80}")
        print(f"策略 {strategies.index(strategy_name) + 1}/{len(strategies)}: {strategy_name.upper()}")
        print(f"{'='*80}")

        try:
            # 使用对应策略构建索引
            node_parser = strategy_manager.get_node_parser(strategy_name)
            nodes = node_parser.get_nodes_from_documents(documents)
            index = VectorStoreIndex(nodes)

            # 创建查询引擎
            query_engine = strategy_manager.create_query_engine(
                index,
                strategy_name=strategy_name,
                similarity_top_k=similarity_top_k
            )

            # 评估（verbose=True 显示详细信息）
            result = evaluate_vector_search_precision(
                query_engine=query_engine,
                query=query,
                ground_truth=ground_truth,
                top_k=similarity_top_k,
                verbose=True
            )

            # 添加策略名称和节点数量信息
            result["strategy_name"] = strategy_name
            result["node_count"] = len(nodes)
            results[strategy_name] = result

        except Exception as e:
            print(f"策略 {strategy_name} 评估失败: {e}")
            results[strategy_name] = {
                "strategy_name": strategy_name,
                "error": str(e)
            }

    # 打印对比结果汇总
    print_comparison_summary(results)

    return results


def print_comparison_summary(results: Dict[str, Dict[str, Any]]):
    """
    打印策略对比汇总表

    参数:
        results: 所有策略的评估结果字典
    """
    print(f"\n{'#'*80}")
    print(f"策略对比汇总表")
    print(f"{'#'*80}\n")

    # 定义列宽（使用固定宽度，确保对齐）
    # 策略名称 | 节点数 | 平均分数 | 最高分数 | 包含答案 | 上下文长度
    w1, w2, w3, w4, w5, w6 = 22, 10, 12, 12, 10, 14

    # 使用 | 分隔符来确保列对齐
    # 表头
    header = f"| {'策略名称':<{w1}} | {'节点数':>{w2}} | {'平均分数':>{w3}} | {'最高分数':>{w4}} | {'包含答案':<{w5}} | {'上下文长度':>{w6}} |"
    print(header)
    # 分隔线
    separator = f"|{'-'*(w1+2)}|{'-'*(w2+2)}|{'-'*(w3+2)}|{'-'*(w4+2)}|{'-'*(w5+2)}|{'-'*(w6+2)}|"
    print(separator)

    # 数据行（使用与表头完全相同的格式化方式）
    for strategy_name, result in results.items():
        if "error" in result:
            error_row = f"| {strategy_name.replace('_', ' ').title():<{w1}} | {'错误':>{w2}} | {'N/A':>{w3}} | {'N/A':>{w4}} | {'N/A':<{w5}} | {'N/A':>{w6}} |"
            print(error_row)
        else:
            stats = result.get("statistics", {})
            context_quality = result.get("context_quality", {})

            strategy_display = strategy_name.replace("_", " ").title()
            node_count = result.get("node_count", 0)
            avg_score = stats.get("average_score", 0.0)
            max_score = stats.get("max_score", 0.0)
            contains_answer = context_quality.get("context_contains_answer", "未知")
            context_length = context_quality.get("retrieved_context_length", 0)

            # 格式化数据行，确保与表头对齐（使用相同的格式）
            data_row = f"| {strategy_display:<{w1}} | {node_count:>{w2}} | {avg_score:>{w3}.4f} | {max_score:>{w4}.4f} | {contains_answer:<{w5}} | {context_length:>{w6}} |"
            print(data_row)

    print(f"\n{'#'*80}\n")


def main():
    # 1. 加载文档
    # 使用 SimpleDirectoryReader 加载 markdown 文件
    # 获取脚本所在目录，构建正确的文件路径
    script_dir = Path(__file__).parent  # chunking_research 目录
    project_root = script_dir.parent    # week03-homework 目录
    md_file_path = project_root / "data" / "量子计算：开启未来计算新纪元.md"

    # 检查文件是否存在
    if not md_file_path.exists():
        print(f"错误：文件不存在: {md_file_path}")
        print(f"当前脚本目录: {script_dir}")
        print(f"项目根目录: {project_root}")
        return

    documents = SimpleDirectoryReader(
        input_files=[str(md_file_path)]
    ).load_data()

    print(f"成功加载 {len(documents)} 个文档")
    print(f"文档路径: {md_file_path}")

    # 配置 LLM（用于生成回答）
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True
    )

    # 配置嵌入模型（用于向量化）
    # 注意：DashScope API 要求批处理大小不能超过 10
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,  # 设置批处理大小，必须 <= 10
        embed_input_length=8192  # 设置最大输入长度
    )

    # 2. 定义测试查询和标准答案（根据实际文档内容调整）
    test_query = "量子计算的基本原理是什么？"
    ground_truth = "量子计算基于量子力学原理，与传统的经典计算有着本质的区别。在经典计算机中，信息的基本单位是比特（Bit），它只有两种状态，即 0 或 1，就像普通的开关，要么开，要么关。而量子计算的基本信息单元是量子比特（Qubit），它具有独特的量子特性，不仅可以处于 0 或 1 的状态，还可以处于这两种状态的叠加态。这意味着一个量子比特能够同时表示 0 和 1，从而使量子计算机在同一时刻能够处理多种信息，具备了并行计算的能力。"

    # 3. 同时评估所有四种切片策略
    all_strategies = ["token", "sentence", "sentence_window", "markdown"]

    # 执行对比评估
    comparison_results = compare_strategies(
        documents=documents,
        query=test_query,
        ground_truth=ground_truth,
        strategies=all_strategies,
        similarity_top_k=5
    )

    print(f"\n评估完成！共评估了 {len(comparison_results)} 种策略。")

if __name__ == "__main__":
    main()