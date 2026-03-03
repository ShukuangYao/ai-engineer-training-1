import traceback
from openai import OpenAI
from typing import List, Optional
import json
from qanything_kernel.connector.llm.base import AnswerResult
from qanything_kernel.utils.custom_log import debug_logger
import tiktoken


class OpenAILLM:
    offcut_token: int = 50
    stop_words: Optional[List[str]] = None

    def __init__(self, model, max_token, api_base, api_key, api_context_length, top_p, temperature):
        base_url = api_base
        api_key = api_key

        if max_token is not None:
            self.max_token = max_token
        if model is not None:
            self.model = model
        if api_context_length is not None:
            self.token_window = api_context_length
        if top_p is not None:
            self.top_p = top_p
        if temperature is not None:
            self.temperature = temperature
        self.use_cl100k_base = False
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except Exception as e:
            debug_logger.warning(f"{model} not found in tiktoken, using cl100k_base!")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_cl100k_base = True


        self.client = OpenAI(base_url=base_url, api_key=api_key)
        debug_logger.info(f"OPENAI_API_KEY = {api_key}")
        debug_logger.info(f"OPENAI_API_BASE = {base_url}")
        debug_logger.info(f"OPENAI_API_MODEL_NAME = {self.model}")
        debug_logger.info(f"OPENAI_API_CONTEXT_LENGTH = {self.token_window}")
        debug_logger.info(f"OPENAI_API_MAX_TOKEN = {self.max_token}")
        debug_logger.info(f"TOP_P = {self.top_p}")
        debug_logger.info(f"TEMPERATURE = {self.temperature}")

    @property
    def _llm_type(self) -> str:
        return "using OpenAI API serve as LLM backend"

    # 定义函数 num_tokens_from_messages，该函数返回由一组消息所使用的token数
    def num_tokens_from_messages(self, messages):
        """
        计算消息列表的 token 数量

        Args:
            messages: 消息列表，元素可以是字典或字符串

        Returns:
            int: 计算得到的 token 数量（包含余量）
        """
        # 初始化总 token 数为 0
        total_tokens = 0

        # 遍历消息列表中的每个消息
        for message in messages:
            if isinstance(message, dict):
                # 对于字典类型的消息，假设它包含 'role' 和 'content' 键
                for key, value in message.items():
                    # 每个键（如 'role'）的开销为 3 token
                    total_tokens += 3  # role的开销(key的开销)
                    # 如果值是字符串类型，计算其 token 数
                    if isinstance(value, str):
                        # 使用 tokenizer 编码字符串并计算 token 数
                        tokens = self.tokenizer.encode(value, disallowed_special=())
                        total_tokens += len(tokens)
            elif isinstance(message, str):
                # 对于字符串类型的消息，直接编码计算 token 数
                tokens = self.tokenizer.encode(message, disallowed_special=())
                total_tokens += len(tokens)
            else:
                # 不支持的消息类型，抛出异常
                raise ValueError(f"Unsupported message type: {type(message)}")

        # 根据使用的 tokenizer 类型添加不同比例的余量
        if self.use_cl100k_base:
            # 使用 cl100k_base tokenizer 时，添加 20% 的余量
            total_tokens *= 1.2
        else:
            # 使用其他 tokenizer 时，添加 10% 的余量
            # 保留一定余量，由于metadata信息的嵌入导致token比计算的会多一些
            total_tokens *= 1.1

        # 返回整数形式的 token 数
        return int(total_tokens)

    def num_tokens_from_docs(self, docs):
        total_tokens = 0
        for doc in docs:
            # 对每个文本进行分词
            tokens = self.tokenizer.encode(doc.page_content, disallowed_special=())
            # 累加tokens数量
            total_tokens += len(tokens)
        if self.use_cl100k_base:
            total_tokens *= 1.2
        else:
            total_tokens *= 1.1  # 保留一定余量，由于metadata信息的嵌入导致token比计算的会多一些
        return int(total_tokens)

    async def _call(self, messages: List[dict], streaming: bool = False) -> str:
        try:

            if streaming:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    max_tokens=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=self.stop_words
                )
                for event in response:
                    if not isinstance(event, dict):
                        event = event.model_dump()

                    if isinstance(event['choices'], List) and len(event['choices']) > 0:
                        event_text = event["choices"][0]['delta']['content']
                        if isinstance(event_text, str) and event_text != "":
                            delta = {'answer': event_text}
                            yield "data: " + json.dumps(delta, ensure_ascii=False)

            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    max_tokens=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=self.stop_words
                )

                event_text = response.choices[0].message.content if response.choices else ""
                delta = {'answer': event_text}
                yield "data: " + json.dumps(delta, ensure_ascii=False)

        except Exception as e:
            debug_logger.info(f"Error calling OpenAI API: {traceback.format_exc()}")
            delta = {'answer': f"{e}"}
            yield "data: " + json.dumps(delta, ensure_ascii=False)

        finally:
            # debug_logger.info("[debug] try-finally")
            yield f"data: [DONE]\n\n"

    async def generatorAnswer(self, prompt: str,
                              history: List[List[str]] = [],
                              streaming: bool = False) -> AnswerResult:
        """
        生成 LLM 回答，支持流式输出

        Args:
            prompt (str): 提示文本，包含用户问题和相关上下文
            history (List[List[str]]): 对话历史，格式为 [[问题1, 回答1], [问题2, 回答2], ...]
            streaming (bool): 是否启用流式输出

        Returns:
            AnswerResult: 包含回答、历史记录和 token 使用情况的结果对象
        """
        # 处理对话历史
        # 如果历史记录为空或 None，初始化一个空的历史记录
        if history is None or len(history) == 0:
            history = [[]]
        else:
            # 为当前对话添加一个新的空条目
            history.append([])

        # 记录 prompt 的 token 数
        debug_logger.info(f"prompt tokens: {self.num_tokens_from_messages([{'content': prompt}])}")

        # 构建消息列表
        # 消息格式遵循 OpenAI API 的要求
        messages = []
        # 遍历历史记录（除了最后一个空条目）
        for pair in history[:-1]:
            question, answer = pair
            # 添加用户消息
            messages.append({"role": "user", "content": question})
            # 添加助手消息
            messages.append({"role": "assistant", "content": answer})
        # 添加当前的用户提示
        messages.append({"role": "user", "content": prompt})

        # 计算 prompt 的 token 数
        prompt_tokens = self.num_tokens_from_messages(messages)
        total_tokens = 0  # 总 token 数
        completion_tokens = 0  # 完成部分的 token 数

        # 调用底层方法获取响应
        response = self._call(messages, streaming)
        complete_answer = ""  # 累积的完整回答

        # 异步迭代流式响应
        async for response_text in response:
            if response_text:
                # 去除响应前缀 'data: '
                chunk_str = response_text[6:]
                # 如果不是结束标记
                if not chunk_str.startswith("[DONE]"):
                    # 解析 JSON 响应
                    chunk_js = json.loads(chunk_str)
                    # 累积回答内容
                    complete_answer += chunk_js["answer"]
                # 计算完成部分的 token 数
                completion_tokens = self.num_tokens_from_messages([complete_answer])
                # 计算总 token 数
                total_tokens = prompt_tokens + completion_tokens

            # 更新对话历史中的最后一个条目
            history[-1] = [prompt, complete_answer]

            # 创建 AnswerResult 对象
            answer_result = AnswerResult()
            answer_result.history = history  # 更新后的对话历史
            answer_result.llm_output = {"answer": response_text}  # LLM 的原始输出
            answer_result.prompt = prompt  # 用户的提示
            answer_result.total_tokens = total_tokens  # 总 token 数
            answer_result.completion_tokens = completion_tokens  # 完成部分的 token 数
            answer_result.prompt_tokens = prompt_tokens  # 提示的 token 数

            # 生成响应
            yield answer_result


if __name__ == "__main__":

    llm = OpenAILLM()
    streaming = True
    chat_history = []
    prompt = """参考信息：
中央纪委国家监委网站讯 据山西省纪委监委消息：山西转型综合改革示范区党工委副书记、管委会副主任董良涉嫌严重违纪违法，目前正接受山西省纪委监委纪律审查和监察调查。\\u3000\\u3000董良简历\\u3000\\u3000董良，男，汉族，1964年8月生，河南鹿邑人，在职研究生学历，邮箱random@xxx.com，联系电话131xxxxx909，1984年3月加入中国共产党，1984年8月参加工作\\u3000\\u3000历任太原经济技术开发区管委会副主任、太原武宿综合保税区专职副主任，山西转型综合改革示范区党工委委员、管委会副主任。2021年8月，任山西转型综合改革示范区党工委副书记、管委会副主任。(山西省纪委监委)
---
我的问题或指令：
帮我提取上述人物的中文名，英文名，性别，国籍，现任职位，最高学历，毕业院校，邮箱，电话
---
请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
你的回复："""
    final_result = ""
    for answer_result in llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
        resp = answer_result.llm_output["answer"]
        if "DONE" not in resp:
            final_result += json.loads(resp[6:])["answer"]
        debug_logger.info(resp)

    debug_logger.info(f"final_result = {final_result}")
