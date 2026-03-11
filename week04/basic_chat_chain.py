"""
基础对话系统搭建

使用 LangChain 构建基础 Chain：Prompt → LLM → OutputParser
用户说「我昨天下的单」时，系统结合当前时间推断「昨天」的具体日期并回复。
"""

from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatTongyi


def get_date_context() -> dict:
    """提供当前日期与「昨天」的具体日期，供 prompt 做时间推断。"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return {
        "today": today.strftime("%Y年%m月%d日"),
        "yesterday": yesterday.strftime("%Y年%m月%d日"),
        "today_iso": today.strftime("%Y-%m-%d"),
        "yesterday_iso": yesterday.strftime("%Y-%m-%d"),
    }


def build_chain():
    """构建 Chain：Prompt → LLM → OutputParser。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是订单客服助手。当前日期是 {today}，昨天的日期是 {yesterday}。"
         "当用户提到「今天」「昨天」「前天」等相对时间时，请结合上述日期用具体日期回应。"),
        ("human", "{user_input}"),
    ])
    llm = ChatTongyi(model_name="qwen-turbo", temperature=0)
    parser = StrOutputParser()
    return prompt | llm | parser


def run(user_input: str) -> str:
    """用户说一句话，返回结合当前时间推断后的回复。"""
    chain = build_chain()
    date_ctx = get_date_context()
    return chain.invoke({
        "user_input": user_input,
        **date_ctx,
    })


if __name__ == "__main__":
    # 用户说「我昨天下的单」，系统应结合当前时间把「昨天」换成具体日期
    user = "我昨天下的单"
    print("当前时间上下文:", get_date_context())
    print("用户:", user)
    print("回复:", run(user))
