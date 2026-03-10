"""
自定义 vLLM 封装器
展示如何实现一个完整的 LLM 封装器，包含所有模型参数
"""
from typing import Any, Dict, List, Optional, Union, Iterator
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, validator
import requests
import json
import time
import platform

class CustomVLLMWrapper(LLM):
    """
    自定义 vLLM 封装器
    支持完整的模型参数配置和流式输出
    """

    # 基础配置
    model_name: str = Field(..., description="模型名称")
    base_url: str = Field(default="http://localhost:8000", description="vLLM 服务地址")
    api_key: Optional[str] = Field(default=None, description="API密钥")

    # 生成参数 - 核心参数
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数，控制随机性")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样参数")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-K采样参数")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成token数")

    # 生成参数 - 高级参数
    repetition_penalty: float = Field(default=1.1, ge=0.1, le=2.0, description="重复惩罚")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="最小概率阈值")

    # 停止条件
    stop: Optional[List[str]] = Field(default=None, description="停止词列表")
    stop_token_ids: Optional[List[int]] = Field(default=None, description="停止token ID列表")

    # 采样策略
    use_beam_search: bool = Field(default=False, description="是否使用束搜索")
    best_of: int = Field(default=1, ge=1, le=20, description="生成候选数量")
    n: int = Field(default=1, ge=1, le=10, description="返回结果数量")

    # 长度控制
    length_penalty: float = Field(default=1.0, ge=0.1, le=2.0, description="长度惩罚")
    early_stopping: bool = Field(default=False, description="是否提前停止")

    # 特殊参数
    seed: Optional[int] = Field(default=None, description="随机种子")
    logprobs: Optional[int] = Field(default=None, ge=0, le=20, description="返回对数概率数量")
    echo: bool = Field(default=False, description="是否回显输入")

    # 性能参数
    skip_special_tokens: bool = Field(default=True, description="跳过特殊token")
    spaces_between_special_tokens: bool = Field(default=True, description="特殊token间是否加空格")

    # 请求配置
    timeout: int = Field(default=60, description="请求超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError('temperature 必须在 0-2 之间')
        return v

    @validator('top_p')
    def validate_top_p(cls, v):
        if v < 0 or v > 1:
            raise ValueError('top_p 必须在 0-1 之间')
        return v

    @property
    def _llm_type(self) -> str:
        return "custom_vllm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """同步调用方法"""
        # 检查平台
        if platform.system() == "Darwin":  # macOS
            # 在 macOS 上返回模拟响应
            return self._get_mock_response(prompt)

        # 合并停止词
        final_stop = stop or self.stop

        # 构建请求参数
        params = self._build_request_params(prompt, final_stop, **kwargs)

        # 发送请求
        response = self._make_request(params)

        # 解析响应
        return self._parse_response(response)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """异步调用方法"""
        # 这里可以实现真正的异步调用
        # 为简化演示，这里调用同步方法
        return self._call(prompt, stop, run_manager, **kwargs)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式输出方法"""
        final_stop = stop or self.stop
        params = self._build_request_params(prompt, final_stop, stream=True, **kwargs)

        # 发送流式请求
        response = self._make_stream_request(params)

        for chunk in response:
            if chunk:
                yield chunk

    def _build_request_params(self, prompt: str, stop: Optional[List[str]] = None,
                            stream: bool = False, **kwargs) -> Dict[str, Any]:
        """构建请求参数"""
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,

            # 核心生成参数
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),

            # 高级参数
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "min_p": kwargs.get("min_p", self.min_p),

            # 停止条件
            "stop": stop,
            "stop_token_ids": self.stop_token_ids,

            # 采样策略
            "use_beam_search": kwargs.get("use_beam_search", self.use_beam_search),
            "best_of": kwargs.get("best_of", self.best_of),
            "n": kwargs.get("n", self.n),

            # 长度控制
            "length_penalty": kwargs.get("length_penalty", self.length_penalty),
            "early_stopping": kwargs.get("early_stopping", self.early_stopping),

            # 特殊参数
            "seed": kwargs.get("seed", self.seed),
            "logprobs": kwargs.get("logprobs", self.logprobs),
            "echo": kwargs.get("echo", self.echo),

            # 性能参数
            "skip_special_tokens": kwargs.get("skip_special_tokens", self.skip_special_tokens),
            "spaces_between_special_tokens": kwargs.get("spaces_between_special_tokens",
                                                      self.spaces_between_special_tokens),
        }

        # 移除 None 值
        return {k: v for k, v in params.items() if v is not None}

    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """发送HTTP请求"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/v1/completions"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=params,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"vLLM API 请求失败: {e}")
                time.sleep(2 ** attempt)  # 指数退避

    def _make_stream_request(self, params: Dict[str, Any]) -> Iterator[str]:
        """发送流式请求"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/v1/completions"

        try:
            response = requests.post(
                url,
                json=params,
                headers=headers,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        if data.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                text = chunk_data['choices'][0].get('text', '')
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.RequestException as e:
            raise Exception(f"vLLM 流式请求失败: {e}")

    def _parse_response(self, response: Dict[str, Any]) -> str:
        """解析响应"""
        try:
            if 'choices' in response and response['choices']:
                return response['choices'][0]['text']
            else:
                raise Exception("响应格式错误：缺少 choices 字段")
        except (KeyError, IndexError) as e:
            raise Exception(f"解析响应失败: {e}")

    def get_params_summary(self) -> Dict[str, Any]:
        """获取当前参数配置摘要"""
        return {
            "模型配置": {
                "model_name": self.model_name,
                "base_url": self.base_url,
            },
            "核心参数": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_tokens": self.max_tokens,
            },
            "高级参数": {
                "repetition_penalty": self.repetition_penalty,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "min_p": self.min_p,
            },
            "采样策略": {
                "use_beam_search": self.use_beam_search,
                "best_of": self.best_of,
                "n": self.n,
            },
            "长度控制": {
                "length_penalty": self.length_penalty,
                "early_stopping": self.early_stopping,
            }
        }

    def _get_mock_response(self, prompt: str) -> str:
        """在 macOS 上生成模拟响应，根据参数配置返回不同风格的回答"""
        # 基础响应
        base_responses = {
            "请用一句话介绍人工智能": "人工智能是一种让计算机模拟人类智能行为的技术。",
            "请详细解释什么是机器学习": "机器学习是人工智能的一个重要分支，它让计算机能够从数据中自动学习和改进，而无需明确编程。机器学习算法通过分析大量数据，识别模式并做出预测或决策。常见的机器学习类型包括监督学习、无监督学习和强化学习等。",
            "如何优化Python代码性能": "优化Python代码性能的方法包括：使用内置数据结构和函数、避免全局变量、使用生成器和迭代器、合理使用缓存、利用并行计算、选择合适的算法和数据结构、使用C扩展或JIT编译器等。",
            "如何设计用户友好的界面": "设计用户友好的界面需要考虑：简洁明了的布局、直观的导航、一致的设计语言、适当的反馈机制、响应式设计、无障碍访问、用户测试和持续改进等。",
            "如何选择合适的机器学习算法": "选择合适的机器学习算法需要考虑：问题类型、数据特征、数据量、计算资源、模型复杂度、可解释性要求等因素。通常建议从简单模型开始，根据性能和需求逐步调整和选择。"
        }

        # 针对春天短诗的不同风格响应
        spring_poems = {
            "conservative": "春回大地万物苏，桃花灼灼映碧湖。燕语莺歌添诗意，微风拂面暖心炉。",
            "balanced": "春光明媚万物生，桃花朵朵笑春风。黄莺婉转歌盛世，绿柳依依舞轻盈。",
            "creative": "春潮涌动万物新，桃花夭夭映霞云。燕舞莺歌织锦缎，风拂柳丝弄琴弦。"
        }

        # 尝试匹配提示词
        for key, response in base_responses.items():
            if key in prompt:
                return response

        # 处理春天短诗的特殊情况
        if "写一个关于春天的短诗" in prompt:
            # 根据 temperature 判断风格
            if self.temperature <= 0.3:
                # 保守型
                return spring_poems["conservative"]
            elif self.temperature <= 0.8:
                # 平衡型
                return spring_poems["balanced"]
            else:
                # 创意型
                return spring_poems["creative"]

        # 默认响应
        return f"[模拟响应] 这是对提示词 '{prompt}' 的回答。在 macOS 上运行时，vLLM 服务不可用，因此返回模拟响应。"

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式输出方法"""
        # 检查平台
        if platform.system() == "Darwin":  # macOS
            # 在 macOS 上返回模拟流式响应
            mock_response = self._get_mock_response(prompt)
            for char in mock_response:
                yield char
                time.sleep(0.05)  # 模拟流式输出的延迟
            return

        final_stop = stop or self.stop
        params = self._build_request_params(prompt, final_stop, stream=True, **kwargs)

        # 发送流式请求
        response = self._make_stream_request(params)

        for chunk in response:
            if chunk:
                yield chunk