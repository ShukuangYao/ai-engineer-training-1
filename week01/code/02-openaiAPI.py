import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv('DEEPSEEK_API_KEY')
base_url = os.getenv('DEEPSEEK_API_BASE')
print(f"-- debug -- deepseek api key is {api_key[0:10]}******")

# 使用 LangChain 调用 DeepSeek
# 通过配置 base_url 和 api_key 来使用 DeepSeek API
llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model="deepseek-chat"  # DeepSeek 模型名称
)

# 使用 LangChain 的 invoke 方法调用
response = llm.invoke("Hello world!")

print(response.content)


# 正常会输出结果：Hello! It's great to see you. How can I assist you today?