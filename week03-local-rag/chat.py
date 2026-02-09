"""
RAG 对话系统核心引擎 - 集成检索增强生成技术

核心功能：
1. 向量检索：基于查询语义检索相关文档片段
2. 重排序：使用专门的重排序模型提升检索精度
3. 上下文构建：将检索到的文档片段整合为 LLM 输入
4. 流式生成：实时返回模型生成的内容，提升用户体验
5. 多轮对话：支持上下文记忆，实现连贯的对话体验

技术架构：
- 两阶段检索：粗排（向量相似度）+ 精排（重排序模型）
- 动态知识库切换：支持临时文件和持久化知识库
- 异常容错：检索失败时降级为纯 LLM 对话
- 参数可配置：支持调整模型、温度、召回数量等参数

RAG 流程：
用户查询 -> 向量检索 -> 重排序 -> 上下文构建 -> LLM 生成 -> 流式输出
"""

import os  # 操作系统接口，用于环境变量和路径操作
from openai import OpenAI  # OpenAI 兼容客户端，用于调用 DashScope API
from llama_index.core import StorageContext, load_index_from_storage, Settings  # LlamaIndex 核心组件
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,                    # DashScope 嵌入模型
    DashScopeTextEmbeddingModels,          # 模型名称枚举
    DashScopeTextEmbeddingType,            # 文本类型枚举
)

# 重排序模块导入 - 采用优雅降级策略
# 重排序是可选功能，如果导入失败，系统仍能正常工作（仅使用向量相似度排序）
try:
    from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
except ImportError:
    print("Warning: DashScopeRerank not found, will skip reranking")
    DashScopeRerank = None
    # 设计理念：系统在缺少重排序组件时仍能正常工作
    # 通过 None 值标记和条件检查实现功能的优雅降级
    # 重排序可以提升检索精度，但不是必需功能

from create_kb import *

# 系统配置常量
DB_PATH = "VectorStore"  # 向量数据库根路径，存储所有持久化知识库
TMP_NAME = "tmp_abcd"    # 临时知识库标识符，用于即时文档问答

# 嵌入模型配置 - 与知识库构建保持一致
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)

# 本地嵌入模型备选方案（已注释）
# 如果需要离线部署或数据隐私保护，可以使用本地嵌入模型
# 取消注释下面的代码并注释掉上面的 DashScopeEmbedding 即可切换
# from langchain_community.embeddings import ModelScopeEmbeddings
# from llama_index.embeddings.langchain import LangchainEmbedding
# embeddings = ModelScopeEmbeddings(model_id="modelscope/iic/nlp_gte_sentence-embedding_chinese-large")
# EMBED_MODEL = LangchainEmbedding(embeddings)

# 全局嵌入模型设置 - 确保检索和构建使用相同的向量空间
Settings.embed_model = EMBED_MODEL

def get_model_response(multi_modal_input, history, model, temperature, max_tokens, history_round, db_name, similarity_threshold, chunk_cnt):
    """
    RAG对话系统核心响应生成器

    技术架构设计：
    1. 多模态输入处理：支持文本+文件混合输入模式
    2. 动态知识库切换：临时文件优先级高于预设知识库
    3. 两阶段检索策略：粗排（向量相似度）+ 精排（重排序模型）
    4. 流式响应生成：实时输出提升用户体验
    5. 异常容错机制：检索失败时降级为纯LLM对话

    核心算法流程：
    输入解析 -> 知识库选择 -> 向量检索 -> 重排序 -> 上下文构建 -> LLM生成
    """
    # 提取用户查询 - 从对话历史获取最新用户输入
    # history 格式：[[用户消息1, 助手回复1], [用户消息2, 助手回复2], ...]
    prompt = history[-1][0]  # 获取最后一条用户消息
    tmp_files = multi_modal_input['files']  # 获取上传的文件列表

    # 动态知识库选择策略
    # 优先级：临时上传文件 > 用户选择的知识库
    # 这样用户可以直接上传文件进行问答，无需预先创建知识库
    if os.path.exists(os.path.join("File", TMP_NAME)):
        # 如果临时知识库已存在，优先使用临时知识库
        db_name = TMP_NAME
    else:
        if tmp_files:
            # 如果有新上传的文件，实时构建临时知识库
            # 支持即时文档问答，提升用户体验
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME

    print(f"prompt:{prompt},tmp_files:{tmp_files},db_name:{db_name}")

    try:
        # 重排序器初始化 - 采用条件性实例化避免导入错误
        if DashScopeRerank is not None:
            dashscope_rerank = DashScopeRerank(
                top_n=chunk_cnt,           # 重排序后保留的文档数量
                return_documents=True      # 返回完整文档而非仅ID
            )
        else:
            dashscope_rerank = None

        # 向量索引加载 - 从持久化存储恢复索引结构
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(DB_PATH, db_name)
        )
        index = load_index_from_storage(storage_context)
        print("index获取完成")

        # 检索器配置 - 第一阶段粗排检索
        # similarity_top_k=20 表示召回 20 个候选文档
        # 这个数量大于最终需要的 chunk_cnt，为后续重排序提供候选池
        retriever_engine = index.as_retriever(
            similarity_top_k=20,  # 粗排阶段召回更多候选文档，提高召回率
        )

        # 向量相似度检索 - 基于查询向量的语义匹配
        # 将用户查询转换为向量，在向量空间中查找最相似的文档片段
        retrieve_chunk = retriever_engine.retrieve(prompt)
        print(f"原始chunk为：{retrieve_chunk}")

        # 第二阶段精排处理 - 使用专门的重排序模型
        try:
            if dashscope_rerank is not None:
                # 重排序的技术优势：
                # 1. 考虑查询与文档的深层语义关系
                # 2. 基于Transformer的交互式编码
                # 3. 相比纯向量相似度有更高的准确性
                results = dashscope_rerank.postprocess_nodes(retrieve_chunk, query_str=prompt)
                print(f"rerank成功，重排后的chunk为：{results}")
            else:
                # 降级策略：直接使用向量相似度排序结果
                results = retrieve_chunk[:chunk_cnt]
                print(f"未使用rerank，chunk为：{results}")
        except Exception as rerank_error:
            # 重排序失败的容错处理
            results = retrieve_chunk[:chunk_cnt]
            print(f"rerank失败，chunk为：{results}")

        # 上下文文本构建 - 基于相似度阈值过滤
        # 将检索到的文档片段整合为 LLM 可以理解的上下文格式
        chunk_text = ""      # 用于 LLM 输入的上下文（简洁格式）
        chunk_show = ""      # 用于用户界面显示的召回文本（包含相似度分数）

        for i in range(len(results)):
            # 相似度阈值过滤 - 排除低相关性文档减少噪声
            # 只保留相似度分数高于阈值的文档片段，提高上下文质量
            if results[i].score >= similarity_threshold:
                # 格式化文档片段，添加序号便于 LLM 引用
                chunk_text += f"## {i+1}:\n {results[i].text}\n"
                # 显示格式包含相似度分数，增强可解释性
                chunk_show += f"## {i+1}:\n {results[i].text}\nscore: {round(results[i].score, 2)}\n"

        print(f"已获取chunk：{chunk_text}")

        # RAG 提示词模板构建 - 结合检索内容和用户查询
        # 将检索到的文档片段和用户查询组合成完整的提示词
        # 这样 LLM 可以基于检索到的上下文生成准确的回答
        prompt_template = f"请参考以下内容：{chunk_text}，以合适的语气回答用户的问题：{prompt}。如果参考内容中有图片链接也请直接返回。"

    except Exception as e:
        # 检索系统异常时的降级策略 - 退化为纯LLM对话
        print(f"异常信息：{e}")
        prompt_template = prompt  # 直接使用原始查询
        chunk_show = ""

    # 对话历史初始化 - 为流式响应预留位置
    # 将最后一条消息的助手回复设为空字符串，后续通过流式生成逐步填充
    history[-1][-1] = ""

    # OpenAI 兼容客户端初始化 - 支持 DashScope API
    # DashScope 提供了 OpenAI 兼容的 API 接口，可以直接使用 OpenAI SDK
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取 API 密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # DashScope 兼容接口地址
    )

    # 对话上下文构建 - 实现多轮对话记忆
    # 系统消息：定义助手的行为和角色
    system_message = {'role': 'system', 'content': 'You are a helpful assistant.'}
    messages = []

    # 上下文窗口管理 - 控制 token 消耗和对话连贯性
    # 只保留最近 N 轮对话，避免上下文过长导致 token 消耗过大
    history_round = min(len(history), history_round)  # 确保不超过实际历史长度
    for i in range(history_round):
        # 从后往前遍历历史，获取最近的对话轮次
        messages.append({'role': 'user', 'content': history[-history_round+i][0]})
        messages.append({'role': 'assistant', 'content': history[-history_round+i][1]})

    # 当前查询添加到消息列表
    messages.append({'role': 'user', 'content': prompt_template})
    # 将系统消息放在最前面
    messages = [system_message] + messages

    # 流式响应生成 - 实时输出提升用户体验
    # 流式输出可以让用户看到模型逐步生成的内容，而不是等待完整响应
    completion = client.chat.completions.create(
        model=model,              # 选择的模型（如 qwen-max, qwen-plus）
        messages=messages,         # 构建好的对话上下文
        temperature=temperature,  # 控制生成的随机性（0-2，越高越随机）
        max_tokens=max_tokens,    # 限制响应长度，避免过长输出
        stream=True               # 启用流式输出，实时返回生成的内容
    )

    # 流式响应处理 - 逐步构建完整回答
    # 每次收到新的 token，就更新对话历史并返回，实现实时显示
    assistant_response = ""
    for chunk in completion:
        # chunk.choices[0].delta.content 包含新生成的文本片段
        if chunk.choices[0].delta.content:
            assistant_response += chunk.choices[0].delta.content  # 累积完整回答
            history[-1][-1] = assistant_response  # 更新对话历史
            # 生成器模式 - 实时返回更新的对话历史和召回文本
            # 这样前端可以实时显示生成的内容，提升用户体验
            yield history, chunk_show