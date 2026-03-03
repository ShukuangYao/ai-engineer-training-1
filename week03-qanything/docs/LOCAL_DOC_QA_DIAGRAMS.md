# local_doc_qa.py 与各组件流程图、架构图与脑图

本文档基于 `qanything_kernel/core/local_doc_qa.py` 绘制其与各组件的关系、主流程与方法脑图。所有图均为 Mermaid 语法，可在支持 Mermaid 的 Markdown 预览或 [Mermaid Live](https://mermaid.live/) 中渲染。

---

## 一、架构图：LocalDocQA 与各组件关系

### 1.1 组件依赖架构（谁依赖谁）

```mermaid
flowchart TB
    subgraph 调用方
        HANDLER[handler.py<br/>local_doc_chat / get_rerank_results 等]
    end

    subgraph LocalDocQA["core/local_doc_qa.py - LocalDocQA"]
        CHAT[get_knowledge_based_answer]
        GET_SRC[get_source_documents]
        RERANK_API[get_rerank_results]
        REPROCESS[reprocess_source_documents]
        GEN_PROMPT[generate_prompt]
        PREPARE[prepare_source_documents]
        WEB[web_page_search]
        REL[calculate_relevance_optimized]
        GEN_RESP[generate_response]
        COMPLETE[get_completed_document]
        AGG[aggregate_documents]
        INCOMPLETE[incomplete_table]
    end

    subgraph 连接器 connector
        EMB[YouDaoEmbeddings]
        RERANK_CLIENT[YouDaoRerank]
        LLM[OpenAILLM]
        MYSQL[KnowledgeBaseManager]
    end

    subgraph 检索与存储 core/retriever
        RET[ParentRetriever]
        MILVUS[VectorStoreMilvusClient]
        ES[StoreElasticSearchClient]
    end

    subgraph 配置与工具
        CFG[model_config]
        CHAINS[condense_q_chain<br/>RewriteQuestionChain]
        WEB_TOOL[web_search_tool<br/>duckduckgo_search]
        UTIL[general_utils]
    end

    HANDLER --> CHAT
    HANDLER --> RERANK_API

    CHAT --> GET_SRC
    CHAT --> WEB
    CHAT --> REPROCESS
    CHAT --> PREPARE
    CHAT --> GEN_PROMPT
    CHAT --> GEN_RESP
    CHAT --> REL
    CHAT --> RERANK_CLIENT
    CHAT --> LLM
    CHAT --> CHAINS
    CHAT --> MYSQL

    GET_SRC --> RET
    GET_SRC --> EMB
    RERANK_API --> MYSQL
    RERANK_API --> EMB
    RERANK_API --> RERANK_CLIENT
    REPROCESS --> LLM
    PREPARE --> AGG
    AGG --> COMPLETE
    COMPLETE --> MYSQL
    REL --> EMB
    REL --> CFG
    WEB --> WEB_TOOL
    WEB --> EMB
    GEN_PROMPT --> CFG

    RET --> MILVUS
    RET --> ES
    RET --> MYSQL
    MILVUS --> MYSQL
```

### 1.2 数据流架构（数据在组件间如何流动）

```mermaid
flowchart LR
    subgraph 输入
        Q[query]
        KH[chat_history]
        KB[kb_ids]
    end

    subgraph LocalDocQA
        REWRITE[多轮改写]
        RETRIEVE[检索]
        MERGE[去重/合并]
        RERANK[重排]
        CUT[token 裁切]
        PROMPT[生成 prompt]
        GEN[LLM 生成]
        IMG[带图选图]
    end

    subgraph 外部存储与模型
        MIL[(Milvus)]
        ES[(ES)]
        MS[(MySQL)]
        EMB_SVC[Embedding API]
        RERANK_SVC[Rerank API]
        LLM_SVC[LLM API]
    end

    Q --> REWRITE
    KH --> REWRITE
    REWRITE --> RETRIEVE
    KB --> RETRIEVE
    RETRIEVE --> MIL
    RETRIEVE --> ES
    RETRIEVE --> MS
    MERGE --> RERANK
    RERANK --> RERANK_SVC
    RERANK --> CUT
    CUT --> PROMPT
    PROMPT --> GEN
    GEN --> LLM_SVC
    GEN --> IMG
    IMG --> EMB_SVC
```

---

## 二、流程图：LocalDocQA 主流程与子流程

### 2.1 主入口 get_knowledge_based_answer 总览

```mermaid
flowchart TB
    START([请求进入]) --> INIT[构造 OpenAILLM]
    INIT --> HAS_HISTORY{"有 chat_history?"}
    HAS_HISTORY -->|是| REWRITE[RewriteQuestionChain 多轮改写]
    HAS_HISTORY -->|否| SKIP_REWRITE[retrieval_query = query]
    REWRITE --> COND[condense_question]
    SKIP_REWRITE --> KB_CHECK{"有 kb_ids?"}
    COND --> KB_CHECK

    KB_CHECK -->|是| GET_SRC[get_source_documents]
    KB_CHECK -->|否| EMPTY[source_documents 置空]
    GET_SRC --> WEB_CHECK{"need_web_search?"}
    EMPTY --> WEB_CHECK

    WEB_CHECK -->|是| WEB[web_page_search 分片 add_document]
    WEB_CHECK -->|否| DEDUP[deduplicate_documents]
    WEB --> DEDUP

    DEDUP --> RERANK_CHECK{"rerank 且 doc>1?"}
    RERANK_CHECK -->|是| RERANK[rerank 过滤 score 相对分]
    RERANK_CHECK -->|否| TOP_K[取 top_k]
    RERANK --> TOP_K

    TOP_K --> STRIP[去掉 headers 行]
    STRIP --> FAQ_HIGH{"高分 FAQ 0.9?"}
    FAQ_HIGH -->|是| FAQ_ONLY[仅保留 FAQ 文档]
    FAQ_HIGH -->|否| FAQ_MATCH{"FAQ 完全匹配?"}
    FAQ_ONLY --> FAQ_MATCH
    FAQ_MATCH -->|是| RETURN_FAQ[generate_response 直接返回 answer]
    FAQ_MATCH -->|否| TEMPLATE[选 prompt 模板]

    TEMPLATE --> REPROCESS[reprocess_source_documents]
    REPROCESS --> TOKEN_ZERO{"裁切后 0 条?"}
    TOKEN_ZERO -->|是| RETURN_ERR[返回 token 不足提示]
    TOKEN_ZERO -->|否| PREPARE[prepare_source_documents]
    PREPARE --> IMG_REF[replace_image_references]
    IMG_REF --> ONLY_SRCH{"only_need_search_results?"}
    ONLY_SRCH -->|是| YIELD_SRC[yield source_documents]
    ONLY_SRCH -->|否| GEN_PROMPT[generate_prompt]

    GEN_PROMPT --> LLM_LOOP[LLM generatorAnswer 流式或非流式]
    LLM_LOOP --> DONE{"流式收到 DONE?"}
    DONE -->|是| IMG_CHECK{"有图片?"}
    DONE -->|否| LLM_LOOP
    IMG_CHECK -->|是| CALC_REL[calculate_relevance_optimized]
    IMG_CHECK -->|否| YIELD_RESP[yield response history]
    CALC_REL --> YIELD_RESP
    RETURN_FAQ --> END([结束])
    RETURN_ERR --> END
    YIELD_SRC --> END
    YIELD_RESP --> END
```

### 2.2 get_source_documents 与检索组件

```mermaid
flowchart LR
    subgraph LocalDocQA
        GET_SRC[get_source_documents]
    end

    subgraph ParentRetriever
        SET_KW[set_search_kwargs]
        GET_RET[get_retrieved_documents]
    end

    subgraph 存储
        MIL[(Milvus)]
        ES[(ES)]
        MS[(MySQL)]
    end

    GET_SRC -->|query, kb_ids, hybrid_search, top_k| GET_RET
    GET_RET -->|expr=kb_id in partition_keys| MIL
    GET_RET -->|hybrid_search| ES
    GET_RET -->|docstore.amget| MS
    GET_SRC -->|len==0 时重启| MIL
    GET_SRC -->|is_deleted_file| MS
    GET_SRC -->|embed_version| EMB[embeddings]
```

### 2.3 reprocess_source_documents 与 token 裁切

```mermaid
flowchart TB
    IN[source_docs, query, history, prompt_template] --> TOKEN_QUERY[query token × 4]
    TOKEN_QUERY --> TOKEN_HIST[history token]
    TOKEN_HIST --> TOKEN_TMPL[template token]
    TOKEN_TMPL --> TOKEN_REF[reference 标签 token]
    TOKEN_REF --> LIMITED[limited_token_nums = window - max - offcut - 上述]
    LIMITED --> LOOP[按 doc 顺序累加 token]
    LOOP --> FILL{总 token ≤ limited?}
    FILL -->|是| APPEND[加入 new_source_docs]
    FILL -->|否| BREAK[停止]
    APPEND --> LOOP
    BREAK --> OUT[(new_source_docs, limited_token_nums, tokens_msg)]
```

### 2.4 get_rerank_results 与 Embedding/Rerank 组件

```mermaid
flowchart TB
    IN[query, doc_ids 或 doc_strs] --> BUILD[构造 docs 列表]
    BUILD --> FROM_MYSQL[doc_ids: milvus_summary.get_document_by_doc_id]
    BUILD --> FROM_STR[doc_strs: Document 直接构造]
    FROM_MYSQL --> CHECK{len>1 且 query token≤300?}
    FROM_STR --> CHECK
    CHECK -->|是| RERANK_CALL[rerank.arerank_documents]
    CHECK -->|否| EMB_QUERY[embeddings.aembed_query]
    RERANK_CALL --> FILTER[score ≥ 0.28]
    RERANK_CALL -->|异常| EMB_QUERY
    EMB_QUERY --> COS[cosine_similarity 打分]
    FILTER --> OUT[带 score 的 docs]
    COS --> OUT
```

### 2.5 calculate_relevance_optimized 与 Embedding/KD 树

```mermaid
flowchart TB
    IN[question, llm_answer, reference_docs, top_k] --> SCORES[question_scores]
    SCORES --> EMB_ANS[embeddings.aembed_query llm_answer]
    EMB_ANS --> SPLIT[doc_splitter.split_documents reference_docs]
    SPLIT --> EMB_SEG[embeddings.aembed_documents segments]
    EMB_SEG --> KDTREE[cKDTree]
    KDTREE --> QUERY[tree.query top_k]
    QUERY --> CUM[cumulative_lengths 反推 doc_id]
    CUM --> GEOMEAN[加权几何平均 similarity_llm + question_score]
    GEOMEAN --> SORT[按 combined_score 降序]
    SORT --> OUT[relevant_docs]
```

---

## 三、脑图：LocalDocQA 类结构与方法关系

### 3.1 类属性与初始化依赖

```mermaid
mindmap
  root((LocalDocQA))
    初始化 __init__ port
      milvus_cache
      embeddings YouDaoEmbeddings
      rerank YouDaoRerank
      chunk_conent
      score_threshold
      milvus_kb VectorStoreMilvusClient
      retriever ParentRetriever
      milvus_summary KnowledgeBaseManager
      es_client StoreElasticSearchClient
      session create_retry_session
      doc_splitter CharacterTextSplitter
    init_cfg
      embeddings = YouDaoEmbeddings
      rerank = YouDaoRerank
      milvus_summary = KnowledgeBaseManager
      milvus_kb = VectorStoreMilvusClient
      es_client = StoreElasticSearchClient
      retriever = ParentRetriever
```

### 3.2 方法分类脑图（按职责）

```mermaid
mindmap
  root((LocalDocQA方法))
    检索与来源
      get_source_documents
        retriever 检索
        Milvus空时重启
        mysql is_deleted_file
      get_rerank_results
        get_document_by_doc_id
        rerank arerank_documents
        embeddings相似度兜底
      web_page_search
        get_web_search
        duckduckgo_search
    上下文与Prompt
      reprocess_source_documents
        num_tokens_from_messages
        limited_token_nums装填
      generate_prompt
        source_docs 转 context
        context 与 question 占位符
      prepare_source_documents
        aggregate_documents可选
        get_completed_document
    文档聚合与表格
      get_completed_document
        get_document_by_file_id
      aggregate_documents
        get_completed_document
        一或两文件完整或截取
      incomplete_table
        get_document_by_doc_id
        table_doc_id完整表格
    回答与带图
      get_knowledge_based_answer
        主流程入口
      generate_response
        静态 yield response history
      calculate_relevance_optimized
        doc_splitter embeddings cKDTree
        show_images选图
    工具
      create_retry_session
        静态 HTTP 重试
```

### 3.3 方法调用关系脑图（谁调谁）

```mermaid
flowchart TB
    subgraph 入口
        CHAT[get_knowledge_based_answer]
    end

    subgraph 检索与来源
        GET_SRC[get_source_documents]
        RERANK_API[get_rerank_results]
        WEB[web_page_search]
    end

    subgraph 上下文
        REPROCESS[reprocess_source_documents]
        GEN_PROMPT[generate_prompt]
        PREPARE[prepare_source_documents]
    end

    subgraph 文档聚合
        AGG[aggregate_documents]
        COMPLETE[get_completed_document]
        INCOMPLETE[incomplete_table]
    end

    subgraph 回答与带图
        GEN_RESP[generate_response]
        REL[calculate_relevance_optimized]
    end

    CHAT --> GET_SRC
    CHAT --> WEB
    CHAT --> REPROCESS
    CHAT --> PREPARE
    CHAT --> GEN_PROMPT
    CHAT --> GEN_RESP
    CHAT --> REL
    PREPARE --> AGG
    AGG --> COMPLETE
    REPROCESS --> GEN_PROMPT
```

---

## 四、与 handler 的对接关系

```mermaid
flowchart LR
    subgraph handler.py
        LOCAL_CHAT[local_doc_chat]
        RERANK_H[get_rerank_results]
    end

    subgraph local_doc_qa.py
        CHAT[get_knowledge_based_answer]
        RERANK_QA[get_rerank_results]
    end

    LOCAL_CHAT -->|bot_id 或 kb_ids + LLM 参数| CHAT
    LOCAL_CHAT -->|streaming| CHAT
    RERANK_H -->|query, doc_ids/doc_strs| RERANK_QA
    CHAT -->|yield response, history| LOCAL_CHAT
    RERANK_QA -->|docs + score| RERANK_H
```

---

## 五、小结表：方法 ↔ 组件

| LocalDocQA 方法 | 直接依赖组件 | 说明 |
|------------------|--------------|------|
| `__init__` | 无（仅 port） | 属性占位，由 init_cfg 赋值 |
| `init_cfg` | YouDaoEmbeddings, YouDaoRerank, KnowledgeBaseManager, VectorStoreMilvusClient, StoreElasticSearchClient, ParentRetriever | 组装完整 RAG 栈 |
| `create_retry_session` | requests | HTTP 重试会话 |
| `get_web_search` | duckduckgo_search, self.embeddings.embed_version | 联网搜索并统一 Document 格式 |
| `web_page_search` | get_web_search | 对外封装，异常返回 [] |
| `get_source_documents` | ParentRetriever, VectorStoreMilvusClient, self.embeddings.embed_version, retriever.mysql_client | 检索 + 过滤已删 + 写 embed_version/score |
| `reprocess_source_documents` | OpenAILLM.num_tokens_from_messages | 算 token、按 limited 装填文档 |
| `generate_prompt` | 无（仅 config 模板字符串） | source_docs → context，替换 {{context}} {{question}} |
| `get_rerank_results` | milvus_summary, self.embeddings, self.rerank, cosine_similarity | doc 取内容 → 重排或向量打分 |
| `prepare_source_documents` | aggregate_documents（当前被 return 短路） | 可选聚合一/两文件 |
| `calculate_relevance_optimized` | self.embeddings, self.doc_splitter, cKDTree, gmean | 带图回答时选最相关 doc 与 show_images |
| `generate_response` | 无 | 静态，拼 response 与 history 并 yield |
| `get_knowledge_based_answer` | OpenAILLM, RewriteQuestionChain, get_source_documents, web_page_search, milvus_summary.add_document, deduplicate_documents, self.rerank, reprocess_source_documents, prepare_source_documents, generate_prompt, custom_llm.generatorAnswer, calculate_relevance_optimized, generate_response | 主流程入口 |
| `get_completed_document` | milvus_summary.get_document_by_file_id | 按 file_id 取分块并拼成完整文档 |
| `aggregate_documents` | get_completed_document, custom_llm.num_tokens_from_docs | 一/两文件内完整或按 doc_id 范围截取 |
| `incomplete_table` | milvus_summary.get_document_by_doc_id | 表格片段替换为完整表格 doc |

---

以上图表均基于当前 `local_doc_qa.py` 代码整理，便于阅读与维护。若需导出为 PNG/SVG，可使用 [Mermaid Live](https://mermaid.live/) 粘贴对应代码块渲染后导出。
