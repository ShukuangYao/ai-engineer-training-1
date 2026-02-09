"""
YouDao 嵌入模型客户端包装器 - 通过 HTTP 调用本地嵌入服务

主要功能：
1. 实现 LangChain Embeddings 接口，提供标准化的嵌入服务接口
2. 支持同步和异步两种调用方式，满足不同场景的性能需求
3. 通过 HTTP 请求调用本地嵌入服务，实现模型服务的解耦
4. 支持批量处理，提升大量文档嵌入的效率
5. 自动过滤图片和公式标记，确保文本嵌入的准确性

技术架构：
- 客户端-服务端架构：客户端通过 HTTP 调用独立的嵌入服务
- 异步并发处理：使用 aiohttp 和 asyncio 实现高效的批量嵌入
- 会话复用：使用 requests.Session 复用连接，提升同步调用性能

使用场景：
- 文档向量化：将文档转换为向量用于向量检索
- 查询向量化：将用户查询转换为向量用于相似度搜索
- RAG 系统：作为检索增强生成系统的向量化组件
"""

from typing import List  # 类型提示：列表类型
from qanything_kernel.utils.custom_log import debug_logger, embed_logger  # 自定义日志记录器
from qanything_kernel.utils.general_utils import get_time_async, get_time  # 性能计时装饰器
from langchain_core.embeddings import Embeddings  # LangChain 嵌入接口基类
from qanything_kernel.configs.model_config import LOCAL_EMBED_SERVICE_URL, LOCAL_RERANK_BATCH  # 配置常量
import traceback  # 异常追踪
import aiohttp  # 异步 HTTP 客户端
import asyncio  # 异步编程支持
import requests  # 同步 HTTP 客户端


def _process_query(query: str) -> str:
    """
    查询文本预处理函数 - 过滤图片和公式标记

    功能：
    - 移除包含图片标记的行（![figure]）
    - 移除包含公式标记的行（![equation]）
    - 保留其他文本内容，确保嵌入模型只处理纯文本

    为什么需要过滤：
    - 图片和公式标记不是实际文本内容，不应该参与向量化
    - 这些标记可能干扰嵌入模型的语义理解
    - 过滤后可以提升嵌入质量和检索准确性

    参数:
        query: 原始查询文本，可能包含图片和公式标记

    返回:
        str: 过滤后的纯文本内容

    示例:
        >>> _process_query("文本内容\\n![figure]\\n更多文本")
        '文本内容\\n更多文本'
    """
    return '\n'.join([line for line in query.split('\n') if
                      not line.strip().startswith('![figure]') and
                      not line.strip().startswith('![equation]')])


class YouDaoEmbeddings(Embeddings):
    """
    YouDao 嵌入模型客户端类

    继承自 LangChain 的 Embeddings 基类，提供标准化的嵌入服务接口。
    通过 HTTP 请求调用本地嵌入服务，实现模型服务的解耦和独立部署。

    主要特性：
    1. 同步和异步双模式：支持同步阻塞调用和异步并发调用
    2. 批量处理优化：自动将大量文本分批处理，提升效率
    3. 连接复用：使用 Session 复用 HTTP 连接，减少连接开销
    4. 错误处理：完善的异常捕获和日志记录

    使用方式：
        # 同步调用
        embedder = YouDaoEmbeddings()
        embeddings = embedder.embed_documents(["文本1", "文本2"])
        query_embedding = embedder.embed_query("查询文本")

        # 异步调用
        embeddings = await embedder.aembed_documents(["文本1", "文本2"])
        query_embedding = await embedder.aembed_query("查询文本")
    """

    def __init__(self):
        """
        初始化 YouDao 嵌入客户端

        配置项：
        - model_version: 模型版本标识，用于版本管理和兼容性检查
        - url: 嵌入服务的 HTTP 接口地址
        - session: requests Session 对象，用于复用连接提升性能
        """
        self.model_version = 'local_v20240725'  # 模型版本号，标识嵌入模型的版本
        self.url = f"http://{LOCAL_EMBED_SERVICE_URL}/embedding"  # 构建嵌入服务 URL
        self.session = requests.Session()  # 创建 HTTP 会话，复用连接提升性能
        super().__init__()  # 调用父类初始化

    async def _get_embedding_async(self, session: aiohttp.ClientSession, queries: List[str]) -> List[List[float]]:
        """
        异步嵌入请求内部方法

        功能：
        - 向嵌入服务发送异步 HTTP POST 请求
        - 将文本列表转换为向量列表
        - 使用 aiohttp 实现非阻塞的异步请求

        参数:
            session: aiohttp 客户端会话对象，用于发送异步请求
            queries: 待嵌入的文本列表

        返回:
            List[List[float]]: 文本对应的向量列表，每个向量是一个浮点数列表

        注意：
        - 这是内部方法，不直接对外暴露
        - 使用 async with 确保响应资源正确释放
        """
        data = {'texts': queries}  # 构建请求体，包含待嵌入的文本列表
        async with session.post(self.url, json=data) as response:  # 发送异步 POST 请求
            return await response.json()  # 解析 JSON 响应并返回向量列表

    @get_time_async
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量嵌入文档

        功能：
        - 将大量文本分批处理，每批并发请求嵌入服务
        - 使用 asyncio.gather 实现并发请求，大幅提升处理速度
        - 自动合并所有批次的嵌入结果，保持原始顺序

        性能优化：
        - 批量处理：减少网络请求次数，提升吞吐量
        - 并发请求：多个批次同时处理，充分利用 I/O 等待时间
        - 连接复用：使用 aiohttp.ClientSession 复用连接

        参数:
            texts: 待嵌入的文本列表，可以是任意长度的列表

        返回:
            List[List[float]]: 每个文本对应的向量列表，顺序与输入文本一致

        性能说明：
        - 对于大量文本，异步并发处理比同步顺序处理快数倍
        - 批处理大小由 LOCAL_RERANK_BATCH 配置决定
        - 使用 @get_time_async 装饰器自动记录处理时间

        示例:
            >>> embedder = YouDaoEmbeddings()
            >>> texts = ["文本1", "文本2", "文本3"]
            >>> embeddings = await embedder.aembed_documents(texts)
            >>> len(embeddings)  # 输出: 3
        """
        batch_size = LOCAL_RERANK_BATCH  # 批处理大小，从配置读取（注意：虽然变量名是 RERANK_BATCH，但这里用作嵌入批处理大小）
        # 计算需要处理的批次数（向上取整）
        # 公式说明：(len(texts) + batch_size - 1) // batch_size
        # - // 是整数除法（向下取整），例如：10 // 3 = 3, 11 // 3 = 3, 12 // 3 = 4
        # - 这个公式实现了向上取整的效果：
        #   例如：10个文本，batch_size=3
        #   正常除法：10 / 3 = 3.333...，需要4个批次
        #   使用公式：(10 + 3 - 1) // 3 = 12 // 3 = 4 ✓
        #   等价于：math.ceil(len(texts) / batch_size)，但不需要导入 math 模块
        batch_count = (len(texts) + batch_size - 1) // batch_size
        embed_logger.info(f'embedding texts number: {batch_count} batches (total {len(texts)} texts)')

        all_embeddings = []  # 存储所有嵌入结果
        async with aiohttp.ClientSession() as session:  # 创建异步 HTTP 会话
            # 创建所有批次的异步任务
            # 将文本列表按 batch_size 切分，每个批次创建一个异步任务
            tasks = [self._get_embedding_async(session, texts[i:i + batch_size])
                     for i in range(0, len(texts), batch_size)]
            # 并发执行所有任务，等待所有批次完成
            results = await asyncio.gather(*tasks)
            # 合并所有批次的嵌入结果
            for result in results:
                all_embeddings.extend(result)

        debug_logger.info(f'success embedding number: {len(all_embeddings)}')
        return all_embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """
        异步嵌入单个查询文本

        功能：
        - 将单个查询文本转换为向量
        - 复用 aembed_documents 方法，保持代码简洁
        - 返回单个向量而非向量列表

        使用场景：
        - 用户查询向量化：将用户输入的问题转换为向量用于检索
        - 实时查询：需要快速响应的单次查询场景

        参数:
            text: 待嵌入的查询文本（单个字符串）

        返回:
            List[float]: 查询文本对应的向量

        注意：
        - 虽然内部调用批量方法，但只处理单个文本，性能开销很小
        - 对于单个查询，异步调用可以避免阻塞其他操作

        示例:
            >>> embedder = YouDaoEmbeddings()
            >>> query = "什么是人工智能？"
            >>> vector = await embedder.aembed_query(query)
            >>> len(vector)  # 输出向量维度，如 768 或 1024
        """
        # 将单个文本包装成列表，调用批量方法，然后取第一个结果
        return (await self.aembed_documents([text]))[0]

    def _get_embedding_sync(self, texts: List[str]) -> List[List[float]]:
        """
        同步嵌入请求内部方法

        功能：
        - 向嵌入服务发送同步 HTTP POST 请求
        - 对每个文本进行预处理，过滤图片和公式标记
        - 使用 requests.Session 复用连接，提升性能
        - 完善的错误处理和日志记录

        与异步方法的区别：
        - 同步方法会阻塞当前线程，直到请求完成
        - 适合少量文本或需要顺序处理的场景
        - 异步方法适合大量文本的并发处理

        参数:
            texts: 待嵌入的文本列表

        返回:
            List[List[float]]: 文本对应的向量列表
            None: 如果请求失败，返回 None

        错误处理：
        - 捕获所有异常并记录详细错误信息
        - 返回 None 表示请求失败，调用方需要处理
        - 使用 traceback 记录完整的异常堆栈信息
        """
        # 对每个文本进行预处理，过滤图片和公式标记
        data = {'texts': [_process_query(text) for text in texts]}
        try:
            # 发送同步 POST 请求
            response = self.session.post(self.url, json=data)
            # 检查 HTTP 状态码，如果不是 2xx 会抛出异常
            response.raise_for_status()
            # 解析 JSON 响应
            result = response.json()
            return result
        except Exception as e:
            # 记录详细的错误信息，包括完整的堆栈跟踪
            debug_logger.error(f'sync embedding error: {traceback.format_exc()}')
            return None  # 返回 None 表示请求失败

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        同步批量嵌入文档（LangChain 标准接口）

        功能：
        - 实现 LangChain Embeddings 接口的 embed_documents 方法
        - 同步阻塞调用，适合少量文本或需要顺序处理的场景
        - 自动处理文本预处理和错误情况

        与异步方法的对比：
        - 同步方法：简单直接，但会阻塞线程
        - 异步方法：性能更好，适合大量文本的并发处理

        参数:
            texts: 待嵌入的文本列表

        返回:
            List[List[float]]: 每个文本对应的向量列表

        注意：
        - 如果请求失败，可能返回 None，调用方需要检查
        - 对于大量文本，建议使用异步方法 aembed_documents

        示例:
            >>> embedder = YouDaoEmbeddings()
            >>> texts = ["文档1", "文档2"]
            >>> embeddings = embedder.embed_documents(texts)
        """
        # 注意：@get_time 装饰器被注释掉了，可能是为了避免频繁的性能日志
        return self._get_embedding_sync(texts)

    @get_time
    def embed_query(self, text: str) -> List[float]:
        """
        同步嵌入单个查询文本（LangChain 标准接口）

        功能：
        - 实现 LangChain Embeddings 接口的 embed_query 方法
        - 将单个查询文本转换为向量
        - 使用 @get_time 装饰器记录处理时间

        使用场景：
        - 用户查询向量化：将用户输入的问题转换为向量用于检索
        - 实时查询：需要快速响应的单次查询场景
        - 同步调用：适合在同步代码中使用

        参数:
            text: 待嵌入的查询文本（单个字符串）

        返回:
            List[float]: 查询文本对应的向量

        注意：
        - 如果请求失败，可能抛出异常或返回 None
        - 对于单个查询，性能开销很小
        - 使用 @get_time 装饰器自动记录处理时间

        示例:
            >>> embedder = YouDaoEmbeddings()
            >>> query = "什么是机器学习？"
            >>> vector = embedder.embed_query(query)
            >>> len(vector)  # 输出向量维度
        """
        # 注释掉的代码是旧版本的实现方式
        # return self._get_embedding([text])['embeddings'][0]
        # 新版本：将单个文本包装成列表，调用同步方法，然后取第一个结果
        return self._get_embedding_sync([text])[0]

    @property
    def embed_version(self) -> str:
        """
        获取嵌入模型版本号（属性方法）

        功能：
        - 返回当前使用的嵌入模型版本标识
        - 用于版本管理和兼容性检查
        - 作为属性方法，可以通过 .embed_version 直接访问

        返回:
            str: 模型版本字符串，如 'local_v20240725'

        使用场景：
        - 版本检查：确认使用的模型版本
        - 兼容性验证：检查向量索引是否与模型版本匹配
        - 日志记录：在日志中记录模型版本信息

        示例:
            >>> embedder = YouDaoEmbeddings()
            >>> version = embedder.embed_version
            >>> print(version)  # 输出: 'local_v20240725'
        """
        return self.model_version

# ============================================================================
# 使用示例代码（已注释）
# ============================================================================
#
# 异步使用示例：
# async def main():
#     """异步嵌入示例"""
#     embedder = YouDaoEmbeddings()
#     query = "Your query here"
#     texts = ["text1", "text2"]  # 示例文本
#     embeddings = await embedder.aembed_documents(texts)
#     return embeddings
#
# if __name__ == '__main__':
#     embeddings = asyncio.run(main())
#
# 同步使用示例：
# embedder = YouDaoEmbeddings()
# texts = ["文档1", "文档2", "文档3"]
# embeddings = embedder.embed_documents(texts)  # 同步批量嵌入
# query_vector = embedder.embed_query("查询文本")  # 同步单文本嵌入
#
# ============================================================================