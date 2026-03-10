"""
基于DashScope的多策略融合意图识别流水线
使用通义千问模型替代OpenAI和HuggingFace

整体逻辑：
  1. 用户输入文本 → 并行经过 规则引擎 / DashScope ML 模型 / 通义千问 LLM 路由
  2. 三种方法各自输出一个意图（或 unknown）
  3. 投票逻辑（默认优先级：规则 > ML > LLM）融合得到最终意图
  4. 置信度 = 与最终意图一致的方法数 / 总方法数

流程图与配置说明见同目录 流程图.md
"""

# 屏蔽 transformers 触发的 pytree 弃用警告（依赖库未更新，非本脚本问题）
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*register_pytree_node.*deprecated.*",
    category=UserWarning,
)

import json
import os
import time
import requests
from http import HTTPStatus

# 网络/SSL 瞬时错误时重试（如 SSLEOFError、ConnectionError）
def _retry_on_network_error(func, max_attempts: int = 3, delay: float = 1.5):
    """对 func() 在连接/SSL 类异常时重试，最多 max_attempts 次，间隔 delay 秒。"""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return func()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, OSError) as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
    raise last_err

# 443/SSL 错误常见原因与处理：
# - 端口 443 被防火墙/公司网络拦截 → 换网络（如手机热点）或联系管理员放行 dashscope.aliyuncs.com
# - 本机开了代理但代理对阿里云不稳定 → 临时取消代理：unset HTTP_PROXY HTTPS_PROXY
# - 必须走代理才能访问外网 → 设置代理：export HTTPS_PROXY=http://代理地址:端口
# requests 会自动读取 HTTP_PROXY/HTTPS_PROXY，dashscope SDK 内部请求也会受其影响
def _get_requests_proxies():
    """从环境变量读取代理，供 requests 使用；未设置则返回 None（使用默认行为）。"""
    p = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if p:
        return {"https": p, "http": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or p}
    return None

from typing import Dict, List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# 通义千问调用：优先用 dashscope SDK，避免依赖 langchain_community.llms.Tongyi（pydantic_v1 兼容问题）
try:
    from dashscope import Generation
except ImportError:
    Generation = None


class RuleEngine:
    """
    规则引擎：基于关键词匹配的快速意图识别。
    无需调用 API，延迟低，适合明确关键词场景；无法处理同义、模糊表达。
    """

    def __init__(self, keywords_config: Dict[str, List[str]]):
        # keywords_config: 意图名 -> 该意图对应的关键词列表，如 {"query_order": ["查订单", "订单号", ...]}
        self.keywords = keywords_config

    def predict(self, text: str) -> str:
        """
        基于关键词规则判断意图。
        按配置顺序遍历意图，命中任一关键词即返回该意图；均未命中返回 "unknown"。
        """
        text = text.lower()
        for intent, words in self.keywords.items():
            if any(word in text for word in words):
                return intent
        return "unknown"


class DashScopeMLModel:
    """
    DashScope 文本生成模型封装：通过 HTTP 调用阿里云 DashScope 文本生成接口做意图分类。
    使用固定 prompt 让模型从给定意图列表中选一个；返回结果需从模型输出文本中解析出意图名。
    """

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def predict(self, text: str) -> str:
        """
        调用 DashScope 文本生成 API，用「意图分类」提示得到模型输出，再从输出中匹配合法意图名。
        解析失败或 API 异常时返回 "unknown"。
        """
        try:
            # 构建「只返回意图名称」的分类提示
            prompt = f"""请对以下文本进行意图分类，从这些选项中选择一个：
                        query_order（查询订单）
                        refund_request（退款申请）
                        issue_invoice（开具发票）
                        logistics_inquiry（物流查询）
                        cancel_order（取消订单）

                        文本：{text}

                        请只返回意图名称，不要其他解释。"""

            payload = {
                "model": self.model_name,
                "input": {
                    "messages": [{"role": "user", "content": prompt}],
                },
                "parameters": {
                    "temperature": 0.1,
                    "max_tokens": 50,
                },
            }

            def _do_request():
                return requests.post(
                    f"{self.base_url}/services/aigc/text-generation/generation",
                    headers=self.headers,
                    json=payload,
                    timeout=15,
                    proxies=_get_requests_proxies(),
                )
            response = _retry_on_network_error(_do_request)

            if response.status_code == 200:
                result = response.json()
                if "output" in result and "text" in result["output"]:
                    intent = result["output"]["text"].strip().lower()
                    # 在模型返回文本中查找第一个出现的合法意图名（兼容多字、带解释的返回）
                    for valid_intent in [
                        "query_order",
                        "refund_request",
                        "issue_invoice",
                        "logistics_inquiry",
                        "cancel_order",
                    ]:
                        if valid_intent in intent:
                            return valid_intent
            return "unknown"
        except Exception as e:
            print(f"DashScope API调用失败: {e}")
            _print_443_hint(e)
            return "unknown"


def _print_443_hint(e: Exception):
    """遇到 443/SSL/连接类错误时打印排查提示（仅首次）。"""
    err_str = str(e).lower()
    if "443" in err_str or "ssl" in err_str or "connection" in err_str or "eof" in err_str:
        print("  提示：若为 443/SSL 错误，可尝试：1) 换网络或关闭 VPN  2) 取消代理 unset HTTP_PROXY HTTPS_PROXY  3) 需代理时设置 export HTTPS_PROXY=...")


class TongyiLLMRouter:
    """
    通义千问 LLM 路由：用 dashscope SDK 直接调用通义千问，做少样本意图分类。
    支持配置化意图列表（intents），适合复杂/模糊表述；未安装 dashscope 时直接返回 "unknown"。
    """

    def __init__(self, api_key: str, model_name: str = "qwen-plus"):
        self.api_key = api_key
        self.model_name = model_name
        # LangChain 模板仅用于拼 prompt 字符串，不依赖 LangChain 的 Tongyi LLM
        self.prompt = ChatPromptTemplate.from_template("""
                    你是一个意图分类器，请从以下选项中选择最匹配的意图：
                    {intents}

                    示例：
                    输入：我想查订单 → query_order
                    输入：怎么退款？ → refund_request
                    输入：开发票 → issue_invoice
                    输入：物流在哪里看？ → logistics_inquiry
                    输入：不要这个订单了 → cancel_order

                    用户输入：{input}

                    请只返回意图名称。
        """)

    def predict(self, text: str, intents: List[str]) -> str:
        """
        使用通义千问进行意图预测。
        intents: 当前流水线支持的意图列表，会填入 prompt；返回结果在该列表中做子串匹配。
        """
        if Generation is None:
            print("通义千问预测跳过: 未安装 dashscope，请 pip install dashscope")
            return "unknown"
        try:
            prompt_str = self.prompt.format(
                intents="\n".join([f"- {intent}" for intent in intents]),
                input=text,
            )

            def _do_call():
                return Generation.call(
                    api_key=self.api_key,
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_str}],
                    result_format="message",
                    temperature=0,
                )
            response = _retry_on_network_error(_do_call)

            if response.status_code != HTTPStatus.OK:
                print(
                    f"通义千问 API 错误: {getattr(response, 'code', '')} {getattr(response, 'message', '')}"
                )
                return "unknown"
            # 从 message 格式响应中取首条 content
            result = (
                response.output.choices[0].message.content
                if response.output and response.output.choices
                else ""
            )
            result = (result or "").strip().lower()
            # 在返回文本中查找第一个命中的配置意图（兼容多字或带解释）
            for intent in intents:
                if intent.lower() in result:
                    return intent
            return "unknown"
        except Exception as e:
            print(f"通义千问预测失败: {e}")
            _print_443_hint(e)
            return "unknown"


class VotingLogic:
    """
    投票逻辑：将规则 / ML / LLM 多条预测融合为单一最终意图。
    支持 priority（按优先级取第一个非 unknown）与 majority（取出现次数最多的意图）。
    """

    def __init__(self, strategy: str = "priority"):
        """
        strategy: "priority" 按规则 > ML > LLM 取第一个非 unknown；
                  "majority" 取多数一致；其他回退到 priority。
        """
        self.strategy = strategy

    def vote(self, results: Dict[str, str]) -> str:
        """根据当前策略从 results（如 rule_intent, ml_intent, llm_intent）中选出最终意图。"""
        if self.strategy == "priority":
            return self._priority_vote(results)
        elif self.strategy == "majority":
            return self._majority_vote(results)
        else:
            return self._priority_vote(results)

    def _priority_vote(self, results: Dict[str, str]) -> str:
        """优先级投票：规则 > ML 模型 > LLM，取第一个非 unknown 的结果。"""
        if results.get("rule_intent") != "unknown":
            return results["rule_intent"]
        if results.get("ml_intent") != "unknown":
            return results["ml_intent"]
        return results.get("llm_intent", "unknown")

    def _majority_vote(self, results: Dict[str, str]) -> str:
        """多数投票：忽略 unknown，统计各意图出现次数，返回得票最多的意图。"""
        vote_count = {}
        for method, intent in results.items():
            if intent != "unknown":
                vote_count[intent] = vote_count.get(intent, 0) + 1
        if not vote_count:
            return "unknown"
        return max(vote_count, key=vote_count.get)


# ---------- 默认配置：无 config.json 或缺少字段时使用（仅规则引擎可运行） ----------
DEFAULT_INTENTS = [
    "query_order",
    "refund_request",
    "issue_invoice",
    "logistics_inquiry",
    "cancel_order",
    "unknown",
]
DEFAULT_RULE_KEYWORDS = {
    "query_order": ["查订单", "订单号", "订单状态", "我的订单"],
    "refund_request": ["退钱", "退款", "取消", "申请退款"],
    "issue_invoice": ["开发票", "要发票", "报销", "发票"],
    "logistics_inquiry": ["物流", "快递", "配送", "运输"],
    "cancel_order": ["取消订单", "不要了", "取消"],
}


class DashScopeIntentPipeline:
    """
    基于 DashScope 的多策略融合意图识别流水线。
    根据 config 启用规则引擎 / ML 模型 / LLM 路由，用 RunnableParallel 并行执行，
    再用 VotingLogic 融合结果，输出最终意图、置信度与各方法详情。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 意图列表：用于 LLM 路由的选项与置信度计算
        self.intents = config.get("intents", DEFAULT_INTENTS)

        self._init_rule_engine()
        self._init_ml_model()
        self._init_llm_router()
        self._init_voting_logic()
        self._create_pipeline()

    def _init_rule_engine(self):
        """按 config 的 enable_rule_engine、rule_keywords 初始化规则引擎，否则置为 None。"""
        if self.config.get("enable_rule_engine", True):
            self.rule_engine = RuleEngine(
                self.config.get("rule_keywords", DEFAULT_RULE_KEYWORDS)
            )
        else:
            self.rule_engine = None

    def _init_ml_model(self):
        """按 config 的 enable_ml_model、ml_model（model_name/api_key/base_url）初始化 DashScope ML，否则 None。"""
        # qwen-turbo：只做“从几句话里选一个意图”，任务简单，用 turbo 省时省钱。轻量、高并发、实时场景。够用，适合简单/结构化任务。更快、延迟更低。更便宜（按 token 计费更低。典型用法：意图分类、简单问答、大批量调用
        if self.config.get("enable_ml_model", True):
            ml_config = self.config.get("ml_model", {})
            self.ml_model = DashScopeMLModel(
                model_name=ml_config.get("model_name", "qwen-turbo"),
                api_key=ml_config.get("api_key", ""),
                base_url=ml_config.get(
                    "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
            )
        else:
            self.ml_model = None

    def _init_llm_router(self):
        """按 config 的 enable_llm_router、llm（api_key/model）初始化通义 LLM 路由，否则 None。"""
        # qwen-plus：处理“规则没命中、需要理解同义/模糊说法”的情况，用 plus 效果更稳。效果更好、复杂任务。更强，复杂理解、长文本、推理更好。稍慢一些。更贵。典型用法：复杂对话、长文总结、多步推理。
        if self.config.get("enable_llm_router", True):
            llm_config = self.config.get("llm", {})
            self.llm_router = TongyiLLMRouter(
                api_key=llm_config.get("api_key", ""),
                model_name=llm_config.get("model", "qwen-plus"),
            )
        else:
            self.llm_router = None

    def _init_voting_logic(self):
        """按 config 的 voting_strategy（priority / majority）初始化投票逻辑。"""
        strategy = self.config.get("voting_strategy", "priority")
        self.voting_logic = VotingLogic(strategy)

    def _create_pipeline(self):
        """
        根据已启用的组件构建 LangChain RunnableParallel。
        每个 runnable 接收 {"input": text}，输出对应 key 的意图（rule_intent / ml_intent / llm_intent）。
        """
        runnables = {}

        if self.rule_engine:
            runnables["rule_intent"] = lambda x: self.rule_engine.predict(x["input"])

        if self.ml_model:
            runnables["ml_intent"] = lambda x: self.ml_model.predict(x["input"])

        if self.llm_router:
            runnables["llm_intent"] = lambda x: self.llm_router.predict(
                x["input"], self.intents
            )

        if runnables:
            self.parallel_router = RunnableParallel(**runnables)
        else:
            self.parallel_router = None

    def predict(self, text: str) -> Dict[str, Any]:
        """
        对用户输入 text 做意图识别。
        返回: intent（最终意图）, confidence（与最终意图一致的方法数/总方法数）, details（各方法原始结果）。
        """
        if not self.parallel_router:
            return {"intent": "unknown", "confidence": 0.0, "details": {}}

        try:
            # 并行执行规则 / ML / LLM，得到 {"rule_intent": ..., "ml_intent": ..., "llm_intent": ...}
            results = self.parallel_router.invoke({"input": text})

            # 按配置策略选出最终意图
            final_intent = self.voting_logic.vote(results)

            return {
                "intent": final_intent,
                "confidence": self._calculate_confidence(results, final_intent),
                "details": results,
            }
        except Exception as e:
            print(f"意图识别失败: {e}")
            return {"intent": "unknown", "confidence": 0.0, "details": {}}

    def _calculate_confidence(self, results: Dict[str, str], final_intent: str) -> float:
        """
        置信度 = 与 final_intent 一致的结果数 / 总方法数。
        final_intent 为 unknown 时返回 0.0。
        """
        if final_intent == "unknown":
            return 0.0
        agreement_count = sum(
            1 for intent in results.values() if intent == final_intent
        )
        total_methods = len(results)
        return agreement_count / total_methods if total_methods > 0 else 0.0


def load_config(config_path: str) -> Dict[str, Any]:
    """从 config_path 读取 JSON 配置；文件不存在或解析失败时打印错误并返回空字典。"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


# 占位符：config 中未填真实 Key 时的占位字符串；若同时无环境变量则仅用规则引擎
DASHSCOPE_API_KEY_PLACEHOLDER = "your_dashscope_api_key_here"
DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"


def _resolve_api_key_from_config_or_env(config_section: Dict[str, Any]) -> str:
    """从 config 段取 api_key；若缺失或为占位符则从环境变量 DASHSCOPE_API_KEY 读取。"""
    key = (config_section or {}).get("api_key") or ""
    if not key or key.strip() == DASHSCOPE_API_KEY_PLACEHOLDER:
        key = os.environ.get(DASHSCOPE_API_KEY_ENV, "")
    return key.strip()


def main():
    """
    演示入口：加载 config.json，无有效 API Key 时仅启用规则引擎；
    API Key 优先从 config 读取，否则从环境变量 DASHSCOPE_API_KEY 读取。
    """
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(_script_dir, "config.json")
    config = load_config(config_path)

    # 从 config 或环境变量解析 API Key，并写回 config，供流水线使用
    config.setdefault("ml_model", {})
    config.setdefault("llm", {})
    config["ml_model"]["api_key"] = _resolve_api_key_from_config_or_env(
        config["ml_model"]
    )
    config["llm"]["api_key"] = _resolve_api_key_from_config_or_env(config["llm"])

    # 若仍无有效 Key（空或占位符），则只启用规则引擎
    api_key = config["ml_model"]["api_key"]
    if not api_key or api_key == DASHSCOPE_API_KEY_PLACEHOLDER:
        print("警告: 请在 config.json 或环境变量 DASHSCOPE_API_KEY 中配置 DashScope API 密钥")
        print("当前将只使用规则引擎进行演示")
        config["enable_ml_model"] = False
        config["enable_llm_router"] = False

    pipeline = DashScopeIntentPipeline(config)

    test_cases = [
        "我想查一下我的订单状态",
        "怎么申请退款？",
        "需要开发票",
        "物流信息在哪里看？",
        "取消订单",
        "我要报销，需要发票",
    ]

    print("=== 基于DashScope的多策略融合意图识别演示 ===\n")

    for text in test_cases:
        result = pipeline.predict(text)
        print(f"输入: {text}")
        print(f"预测意图: {result['intent']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"详细结果: {result['details']}")
        print("-" * 50)


if __name__ == "__main__":
    main()