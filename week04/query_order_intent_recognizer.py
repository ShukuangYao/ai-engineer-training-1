#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查订单意图识别器 - LangChain 多策略融合
========================================
- 正则匹配「查订单」「订单号」等关键词
- HuggingFace 微调 BERT 分类（可选，需设 QUERY_ORDER_BERT_MODEL）
- 大模型 few-shot 兜底（需 OPENAI_API_KEY 或 DASHSCOPE_API_KEY）
- 槽位 order_id 提取（正则 / 模型）
- 输出: {"intent": "query_order", "slots": {"order_id": "123456"}}

依赖: langchain-core, langchain-openai 或 langchain_community (Tongyi);
     可选 transformers, torch 用于 BERT。
"""

import re
import json
import os
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# 1. 正则层：关键词 + order_id 抽取
# ---------------------------------------------------------------------------

QUERY_ORDER_PATTERNS = [
    r"查.*订单",
    r"订单号",
    r"我的订单",
    r"订单.*查",
    r"查.*订单号.*?(\d{6,})",
    r"订单号.*?(\d{6,})",
    r"订单[^\d]*(\d{6,})",
]

ORDER_ID_PATTERNS = [
    r"订单号[^\d]*(\d{6,})",
    r"订单[^\d]*(\d{6,})",
    r"(\d{6,})",  # 6 位以上数字作为候选
]


def regex_match_query_order(text: str) -> tuple[bool, Optional[str]]:
    """正则判断是否为查订单意图，并尝试抽取 order_id。返回 (是否匹配, order_id)。"""
    text = (text or "").strip()
    if not text:
        return False, None
    # 意图：任一查订单相关模式命中即视为 query_order
    for p in QUERY_ORDER_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            break
    else:
        return False, None
    # 槽位：用 order_id 相关模式抽取
    for p in ORDER_ID_PATTERNS:
        m = re.search(p, text)
        if m:
            return True, m.group(1)
    return True, None


def extract_order_id_regex(text: str) -> Optional[str]:
    """仅从文本中抽取 order_id（供其它层未提供时使用）。"""
    for p in ORDER_ID_PATTERNS:
        m = re.search(p, (text or "").strip())
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# 2. BERT 分类层（HuggingFace 微调模型）
# ---------------------------------------------------------------------------

_BERT_PIPELINE = None
# 需为 HuggingFace 上带分类头的模型，如微调后的 bert-base-chinese。不设则跳过 BERT 仅用正则+LLM。
BERT_MODEL_ID = os.environ.get("QUERY_ORDER_BERT_MODEL", "").strip()


def _get_bert_pipeline():
    global _BERT_PIPELINE
    if _BERT_PIPELINE is not None:
        return _BERT_PIPELINE
    if not BERT_MODEL_ID:
        _BERT_PIPELINE = False
        return _BERT_PIPELINE
    try:
        from transformers import pipeline
        _BERT_PIPELINE = pipeline(
            "text-classification",
            model=BERT_MODEL_ID,
            top_k=1,
        )
    except Exception:
        _BERT_PIPELINE = False
    return _BERT_PIPELINE


def bert_classify(text: str) -> tuple[str, float]:
    """BERT 分类。返回 (intent, score)，intent 为 query_order 或 other。"""
    pipe = _get_bert_pipeline()
    if pipe is False or not text.strip():
        return "other", 0.0
    try:
        out = pipe(text.strip(), top_k=1)
        if not out:
            return "other", 0.0
        item = out[0] if isinstance(out[0], dict) else out[0][0]
        label = (item.get("label") or "").lower()
        score = float(item.get("score", 0))
        if "order" in label or "query" in label or label in ("0", "query_order", "positive", "label_0"):
            return "query_order", score
        return "other", score
    except Exception:
        return "other", 0.0


# ---------------------------------------------------------------------------
# 3. LLM 兜底（few-shot）
# ---------------------------------------------------------------------------

FEW_SHOT_PROMPT = """你是一个意图识别器，只判断用户是否在「查订单」并抽取订单号。

## 示例
用户: 帮我查一下订单号123456
输出: {{"intent": "query_order", "slots": {{"order_id": "123456"}}}}

用户: 我的订单什么时候到
输出: {{"intent": "query_order", "slots": {{}}}}

用户: 今天天气怎么样
输出: {{"intent": "other", "slots": {{}}}}

用户: 订单 888888 到哪了
输出: {{"intent": "query_order", "slots": {{"order_id": "888888"}}}}

## 规则
- 只要与查订单、订单状态、订单号相关，intent 为 query_order，否则为 other。
- 能从用户话里看出订单号（数字，通常6位以上）则填入 slots.order_id，否则 slots 为空对象。
- 只输出一行 JSON，不要其他内容。

用户: {input}
输出:"""


def llm_fallback(text: str) -> Dict[str, Any]:
    """大模型 few-shot 兜底，返回 {"intent": ..., "slots": {...}}。"""
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_community.llms import Tongyi
        from langchain_openai import ChatOpenAI
    except ImportError:
        return {"intent": "other", "slots": {}}
    prompt = PromptTemplate.from_template(FEW_SHOT_PROMPT)
    llm = None
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    if llm is None and os.getenv("DASHSCOPE_API_KEY"):
        llm = Tongyi(model="qwen-plus", temperature=0)
    if llm is None:
        return {"intent": "other", "slots": {}}
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"input": (text or "").strip()})
    except Exception:
        return {"intent": "other", "slots": {}}
    raw = (raw or "").strip()
    # 尝试从回复中截取 JSON
    for start in ("{", "输出:"):
        i = raw.find(start)
        if i >= 0:
            raw = raw[i:].replace("输出:", "").strip()
            break
    try:
        obj = json.loads(raw)
        intent = (obj.get("intent") or "other").strip().lower()
        if "query" in intent or "order" in intent:
            intent = "query_order"
        else:
            intent = "other"
        slots = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
        return {"intent": intent, "slots": slots}
    except Exception:
        return {"intent": "other", "slots": {}}


# ---------------------------------------------------------------------------
# 4. LangChain 风格链：串联正则 → BERT → LLM，统一输出
# ---------------------------------------------------------------------------

def recognize(text: str) -> Dict[str, Any]:
    """
    查订单意图识别入口。
    - 先走正则；命中则直接返回 query_order + 正则抽取的 order_id。
    - 未命中则尝试 BERT；若为 query_order 且置信度够高则采用。
    - 否则走 LLM few-shot 兜底。
    - 始终用正则再补一次 order_id（若前面未得到）。
    输出格式: {"intent": "query_order" | "other", "slots": {"order_id": "..."}}
    """
    text = (text or "").strip()
    slots = {}

    # 1) 正则
    matched, order_id = regex_match_query_order(text)
    if matched:
        if order_id:
            slots["order_id"] = order_id
        return {"intent": "query_order", "slots": slots}

    # 2) BERT
    intent_bert, score = bert_classify(text)
    if intent_bert == "query_order" and score >= 0.7:
        order_id = extract_order_id_regex(text)
        if order_id:
            slots["order_id"] = order_id
        return {"intent": "query_order", "slots": slots}

    # 3) LLM 兜底
    out = llm_fallback(text)
    intent = out.get("intent") or "other"
    slots = dict(out.get("slots") or {})
    if intent == "query_order" and not slots.get("order_id"):
        order_id = extract_order_id_regex(text)
        if order_id:
            slots["order_id"] = order_id
    return {"intent": intent, "slots": slots}


# ---------------------------------------------------------------------------
# 5. Runnable 封装（便于与 LangChain 链组合）
# ---------------------------------------------------------------------------

def create_recognizer_chain():
    """返回一个 LangChain Runnable：输入 {"text": "..."}，输出 {"intent": ..., "slots": ...}。"""
    from langchain_core.runnables import RunnableLambda
    def fn(x):
        if isinstance(x, dict):
            text = x.get("text", x.get("input", ""))
        else:
            text = str(x)
        return recognize(text)
    return RunnableLambda(fn)


# ---------------------------------------------------------------------------
# 6. 主函数与示例
# ---------------------------------------------------------------------------

def main():
    examples = [
        "帮我查一下订单号123456",
        "我的订单什么时候到",
        "订单 888888 到哪了",
        "今天天气怎么样",
        "查订单",
    ]
    print("查订单意图识别示例\n" + "=" * 50)
    for s in examples:
        out = recognize(s)
        print(f"输入: {s}")
        print(f"输出: {json.dumps(out, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
