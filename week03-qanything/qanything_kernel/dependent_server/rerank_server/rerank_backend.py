from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
from qanything_kernel.configs.model_config import LOCAL_RERANK_MAX_LENGTH, \
    LOCAL_RERANK_BATCH, LOCAL_RERANK_PATH, LOCAL_RERANK_THREADS
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import get_time
import concurrent.futures
from abc import ABC, abstractmethod


class RerankBackend(ABC):
    """
    重排序模型后端抽象基类

    定义了重排序的核心接口和方法，具体的推理实现由子类完成
    支持批量处理、长文本分段、多线程并发等功能
    """
    def __init__(self, use_cpu: bool = False):
        """
        初始化重排序后端

        Args:
            use_cpu: 是否使用 CPU 进行推理
        """
        self.use_cpu = use_cpu
        # 加载分词器
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH)
        # 获取分隔符 ID
        self.spe_id = self._tokenizer.sep_token_id
        # 文本分段的重叠 token 数
        self.overlap_tokens = 80
        # 批量处理大小
        self.batch_size = LOCAL_RERANK_BATCH
        # 模型最大输入长度
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        # 返回张量类型（由子类设置）
        self.return_tensors = None
        # 线程池大小
        self.workers = LOCAL_RERANK_THREADS

    @abstractmethod
    def inference(self, batch) -> List:
        """
        模型推理抽象方法

        由子类实现具体的模型推理逻辑

        Args:
            batch: 批量输入数据

        Returns:
            List: 每个输入的相关性分数列表
        """
        pass

    def merge_inputs(self, chunk1_raw, chunk2):
        """
        合并两个输入块（通常是查询和文档）

        Args:
            chunk1_raw: 第一个输入块（通常是查询）
            chunk2: 第二个输入块（通常是文档）

        Returns:
            dict: 合并后的输入，包含 input_ids、attention_mask 等
        """
        # 深拷贝第一个输入块，避免修改原始数据
        chunk1 = deepcopy(chunk1_raw)

        # 在 chunk1 的末尾添加分隔符
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].append(1)  # 为分隔符添加 attention mask

        # 添加 chunk2 的内容
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])

        # 在整个序列的末尾再添加一个分隔符
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].append(1)  # 为最后的分隔符添加 attention mask

        '''
        token_type_ids 是 Transformer 模型（如 BERT）的重要输入，用于：
            - 区分不同序列 ：标记 token 属于哪个句子/序列
            - 辅助模型理解 ：帮助模型处理多序列输入（如查询-文档对）
            - 影响注意力计算 ：模型会根据 token 类型调整注意力权重 序列标记规则
            - 0 ：表示属于第一个序列（查询部分）
            - 1 ：表示属于第二个序列（文档部分）
            - 分隔符 ：属于第二个序列（标记为 1）
        '''
        if 'token_type_ids' in chunk1:
            # 为 chunk2 和两个分隔符添加 token_type_ids
            # 1 表示属于第二个序列（文档）
            # 获取 chunk2 的 token_type_ids的长度，并且长度加2都填充为1，扩展到 chunk1 的 token_type_ids 中
            # + 2 ：为两个分隔符（ [SEP] ）添加 token 类型 ID
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 2)]
            chunk1['token_type_ids'].extend(token_type_ids)

        return chunk1

    def tokenize_preproc(self,
                         query: str,
                         passages: List[str],
                         ):
        """
        对查询和文档进行分词预处理

        Args:
            query: 用户查询文本
            passages: 文档段落列表

        Returns:
            tuple: (merge_inputs, merge_inputs_idxs)
                merge_inputs: 合并后的输入列表
                merge_inputs_idxs: 每个输入对应的原始文档索引
        """
        # 对查询进行编码，不截断、不填充
        query_inputs = self._tokenizer.encode_plus(query, truncation=False, padding=False)

        # 计算文档的最大允许长度
        # 减2是因为添加了两个分隔符 [SEP]
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 2

        # 确保文档有足够的长度
        assert max_passage_inputs_length > 10

        # 计算重叠 token 数，确保分段之间有足够的重叠
        # 重叠 token 数不能超过最大允许长度的 2/7
        # //7 表明除以7向下取整
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        # 组[query, passage]对
        merge_inputs = []  # 存储合并后的输入
        merge_inputs_idxs = []  # 存储每个输入对应的原始文档索引

        for pid, passage in enumerate(passages):
            # 对文档进行编码，不添加特殊标记
            passage_inputs = self._tokenizer.encode_plus(passage, truncation=False, padding=False,
                                                         add_special_tokens=False)
            # 计算文档 token 数
            # passage_inputs['input_ids'] 是一个 整数列表 ，表示文档经过分词后每个 token 在词汇表中的 ID。
            # 由 Hugging Face 的 AutoTokenizer.encode_plus() 方法生成
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                # 文档长度在允许范围内，直接处理
                if passage_inputs['attention_mask'] is None or len(passage_inputs['attention_mask']) == 0:
                    continue  # 跳过空文档

                # 合并查询和文档
                # Returns:
                # dict: 合并后的输入，包含 input_ids、attention_mask 等
                qp_merge_inputs = self.merge_inputs(query_inputs, passage_inputs)
                merge_inputs.append(qp_merge_inputs)
                merge_inputs_idxs.append(pid)
            else:
                # 文档过长，需要分段处理
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    # 截取文档片段
                    # passage_inputs字段有可能如下：
                    # {'input_ids': [101, 2023, 2003, 1037, 2084, 102], 'attention_mask': [1, 1, 1, 1, 1, 1]}
                    sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                    # 更新起始位置，保留重叠部分
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id

                    # 合并查询和文档片段
                    qp_merge_inputs = self.merge_inputs(query_inputs, sub_passage_inputs)
                    merge_inputs.append(qp_merge_inputs)
                    merge_inputs_idxs.append(pid)

        return merge_inputs, merge_inputs_idxs

    @get_time
    def get_rerank(self, query: str, passages: List[str]):
        """
        执行重排序，计算查询与每个文档的相关性分数

        Args:
            query: 用户查询文本
            passages: 文档段落列表

        Returns:
            List[float]: 每个文档的相关性分数列表
        """
        # 预处理输入，获取合并后的输入和索引
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        tot_scores = []  # 存储所有批次的分数

        # 使用线程池并发处理批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            # 按批次处理输入
            for k in range(0, len(tot_batches), self.batch_size):
                # 获取当前批次
                batch_inputs = tot_batches[k:k + self.batch_size]
                # 对批次进行填充，使所有输入长度一致
                batch = self._tokenizer.pad(
                    batch_inputs,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors=self.return_tensors
                )
                # 提交推理任务
                future = executor.submit(self.inference, batch)
                futures.append(future)

            # 收集所有批次的结果
            for future in futures:
                scores = future.result()
                tot_scores.extend(scores)

        # 合并分数：对于分段处理的文档，取最高分
        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            # 对每个文档，保留最高的相关性分数
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)

        return merge_tot_scores
