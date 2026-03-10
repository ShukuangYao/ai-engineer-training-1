from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import Tongyi
from typing import List, Dict, Any
import asyncio

class ChatChain:
    def __init__(self):
        self.llm = None
        self.chain = None
        self.parser = StrOutputParser()
        
    async def initialize(self):
        """异步初始化"""
        # 初始化 LLM
        self.llm = Tongyi(
            temperature=0.7,
            model_name="qwen-turbo"  # LangChain 0.3 推荐明确指定模型
        )
        
        # 创建提示模板 - LangChain 0.3 风格
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能客服助手，请遵循以下规则：
                1. 友好、专业地回答用户问题
                2. 如果不确定答案，诚实地说不知道
                3. 保持回答简洁明了
                4. 根据对话历史提供连贯的回复
                5. 用中文回答"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{message}")
        ])
        
        # 构建链 - LangChain 0.3 LCEL 语法
        self.chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self._format_history)
            )
            | self.prompt
            | self.llm
            | self.parser
        )
    
    async def process_message(self, message: str, history: List[Dict] = None) -> str:
        """处理用户消息"""
        try:
            # 准备输入数据
            input_data = {
                "message": message,
                "raw_history": history or []
            }
            
            # 异步调用链
            response = await self.chain.ainvoke(input_data)
            
            return response.strip()
            
        except Exception as e:
            print(f"处理消息时出错: {e}")
            return "抱歉，我现在无法处理您的请求，请稍后再试。"
    
    def _format_history(self, input_data: Dict[str, Any]) -> List:
        """格式化历史消息为 LangChain 消息格式"""
        history = input_data.get("raw_history", [])
        
        if not history:
            return []
        
        messages = []
        # 只保留最近5轮对话
        recent_history = history[-5:] if len(history) > 5 else history
        
        for item in recent_history:
            messages.append(HumanMessage(content=item["user_message"]))
            messages.append(AIMessage(content=item["bot_reply"]))
        
        return messages