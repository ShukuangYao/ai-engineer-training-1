"""
YouDao 重排序模型客户端包装器 - 通过 HTTP 调用本地重排序服务

主要功能：
1. 对检索到的文档进行重新排序，提升检索精度
2. 使用专门的重排序模型，比纯向量相似度更准确
3. 支持异步批量处理，提升大量文档重排序的效率
4. 通过 HTTP 请求调用本地重排序服务，实现模型服务的解耦

重排序在 RAG 系统中的作用：
- 向量检索阶段：使用向量相似度快速召回相关文档（粗排）
- 重排序阶段：使用专门的重排序模型对召回结果进行精确排序（精排）
- 为什么需要重排序：
  1. 向量相似度可能受语义偏差影响，不够精确
  2. 重排序模型考虑查询与文档的深层交互关系
  3. 基于 Transformer 的交互式编码，准确性更高

技术架构：
- 客户端-服务端架构：客户端通过 HTTP 调用独立的重排序服务
- 异步并发处理：使用 aiohttp 和 asyncio 实现高效的批量重排序
- 批量处理优化：将大量文档分批处理，每批并发请求

使用场景：
- RAG 系统：在向量检索后对文档进行精确排序
- 检索优化：提升检索结果的相关性和准确性
- 文档过滤：基于重排序分数过滤低质量文档
"""

import asyncio  # 异步编程支持
import aiohttp  # 异步 HTTP 客户端
from typing import List  # 类型提示：列表类型
from qanything_kernel.utils.custom_log import debug_logger  # 自定义日志记录器
from qanything_kernel.utils.general_utils import get_time_async  # 异步性能计时装饰器
from qanything_kernel.configs.model_config import LOCAL_RERANK_SERVICE_URL, LOCAL_RERANK_BATCH  # 配置常量
from langchain.schema import Document  # LangChain 文档数据结构
import traceback  # 异常追踪


class YouDaoRerank:
    """
    YouDao 重排序模型客户端类

    通过 HTTP 请求调用本地重排序服务，对检索到的文档进行重新排序。
    重排序模型能够更准确地评估查询与文档的相关性，提升检索精度。

    主要特性：
    1. 异步批量处理：支持大量文档的并发重排序
    2. 保持原始顺序：重排序后仍能追踪文档的原始位置
    3. 分数记录：将重排序分数写入文档元数据，便于后续过滤
    4. 错误容错：重排序失败时返回原始文档，保证系统稳定性

    工作流程：
    1. 将文档列表分批处理
    2. 每批并发请求重排序服务
    3. 收集所有批次的分数
    4. 将分数写入文档元数据
    5. 按分数降序排序文档

    使用方式：
        reranker = YouDaoRerank()
        reranked_docs = await reranker.arerank_documents(query, documents)
    """

    def __init__(self):
        """
        初始化 YouDao 重排序客户端

        配置项：
        - url: 重排序服务的 HTTP 接口地址
        """
        self.url = f"http://{LOCAL_RERANK_SERVICE_URL}/rerank"  # 构建重排序服务 URL

    async def _get_rerank_res(self, query: str, passages: List[str]) -> List[float]:
        """
        异步重排序请求内部方法

        功能：
        - 向重排序服务发送异步 HTTP POST 请求
        - 将查询和文档段落发送给重排序模型
        - 获取每个文档段落的相关性分数

        重排序模型的工作原理：
        - 输入：查询文本 + 文档段落列表
        - 处理：使用 Transformer 模型计算查询与每个段落的交互分数
        - 输出：每个段落的相关性分数列表（0-1 之间的浮点数）

        参数:
            query: 用户查询文本
            passages: 待重排序的文档段落列表

        返回:
            List[float]: 每个段落对应的相关性分数列表
            None: 如果请求失败，返回 None

        注意：
        - 这是内部方法，不直接对外暴露
        - 使用 async with 确保响应资源正确释放
        - 完善的错误处理和日志记录
        """
        # 构建请求体：包含查询和文档段落
        data = {
            'query': query,        # 用户查询文本
            'passages': passages   # 待重排序的文档段落列表
        }
        headers = {"content-type": "application/json"}  # 设置请求头

        try:
            async with aiohttp.ClientSession() as session:  # 创建异步 HTTP 会话
                async with session.post(self.url, json=data, headers=headers) as response:
                    if response.status == 200:  # 请求成功
                        scores = await response.json()  # 解析 JSON 响应，获取分数列表
                        return scores
                    else:
                        # 请求失败，记录错误状态码
                        debug_logger.error(f'Rerank request failed with status {response.status}')
                        return None
        except Exception as e:
            # 捕获所有异常，记录详细的错误信息
            debug_logger.info(f'rerank query: {query}, rerank passages length: {len(passages)}')
            debug_logger.error(f'rerank error: {traceback.format_exc()}')
            return None  # 返回 None 表示请求失败

    @get_time_async
    async def arerank_documents(self, query: str, source_documents: List[Document]) -> List[Document]:
        """
        异步批量重排序文档

        功能：
        - 将大量文档分批处理，每批并发请求重排序服务
        - 使用 asyncio.create_task 实现并发请求，提升处理速度
        - 收集所有批次的分数，写入文档元数据
        - 按分数降序排序文档，返回重排序后的文档列表

        性能优化：
        - 批量处理：减少网络请求次数，提升吞吐量
        - 并发请求：多个批次同时处理，充分利用 I/O 等待时间
        - 保持顺序：通过索引映射确保分数正确对应到原始文档

        参数:
            query: 用户查询文本，用于计算文档相关性
            source_documents: 待重排序的文档列表（LangChain Document 对象）

        返回:
            List[Document]: 按相关性分数降序排序的文档列表
            - 每个文档的 metadata['score'] 字段包含重排序分数
            - 如果重排序失败，返回原始文档列表（保证系统稳定性）

        工作流程：
        1. 提取文档内容，准备批量处理
        2. 将文档分批，创建并发任务
        3. 等待所有批次完成，收集分数
        4. 将分数写入文档元数据
        5. 按分数降序排序文档

        性能说明：
        - 对于大量文档，异步并发处理比同步顺序处理快数倍
        - 批处理大小由 LOCAL_RERANK_BATCH 配置决定
        - 使用 @get_time_async 装饰器自动记录处理时间

        示例:
            >>> reranker = YouDaoRerank()
            >>> query = "什么是机器学习？"
            >>> documents = [Document(page_content="内容1"), Document(page_content="内容2")]
            >>> reranked_docs = await reranker.arerank_documents(query, documents)
            >>> # 文档已按相关性分数排序，可通过 doc.metadata['score'] 访问分数
        """
        batch_size = LOCAL_RERANK_BATCH  # 批处理大小，从配置读取

        # 初始化分数列表，默认分数为 0（如果重排序失败，保持原始顺序）
        all_scores = [0 for _ in range(len(source_documents))]

        # 提取所有文档的文本内容，准备发送给重排序服务
        passages = [doc.page_content for doc in source_documents]

        # 创建并发任务列表
        # 每个任务包含起始索引和异步任务对象，用于后续正确映射分数
        tasks = []
        for i in range(0, len(passages), batch_size):
            # 为每个批次创建异步任务
            # 任务返回该批次的分数列表
            task = asyncio.create_task(self._get_rerank_res(query, passages[i:i + batch_size]))
            # 保存起始索引和任务，用于后续将分数映射到正确位置
            tasks.append((i, task))

        # 等待所有批次完成，收集分数
        for start_index, task in tasks:
            res = await task  # 等待当前批次完成
            if res is None:
                # 如果重排序失败，返回原始文档列表（保证系统稳定性）
                # 这样即使重排序服务不可用，系统仍能正常工作
                return source_documents
            # 将分数映射到正确的位置
            # 例如：如果 start_index=0, batch_size=3，则将 res 的前 3 个分数写入 all_scores[0:3]
            all_scores[start_index:start_index + batch_size] = res

        # 将分数写入文档元数据，并保留两位小数
        for idx, score in enumerate(all_scores):
            source_documents[idx].metadata['score'] = round(float(score), 2)

        # 按分数降序排序文档（分数越高，相关性越高）
        # reverse=True 表示降序排序，最相关的文档排在前面
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)

        return source_documents


# ============================================================================
# 使用示例代码（已注释）
# ============================================================================
#
# 异步使用示例：
# async def main():
#     """异步重排序示例"""
#     reranker = YouDaoRerank()
#     query = "什么是人工智能？"
#     # 创建示例文档列表
#     documents = [
#         Document(page_content="人工智能是计算机科学的一个分支..."),
#         Document(page_content="机器学习是人工智能的核心技术..."),
#         Document(page_content="深度学习是机器学习的一个子领域...")
#     ]
#     # 对文档进行重排序
#     reranked_docs = await reranker.arerank_documents(query, documents)
#
#     # 查看重排序结果
#     for doc in reranked_docs:
#         print(f"分数: {doc.metadata['score']}, 内容: {doc.page_content[:50]}...")
#
#     return reranked_docs
#
# # 运行异步主函数
# if __name__ == "__main__":
#     reranked_docs = asyncio.run(main())
#
# ============================================================================