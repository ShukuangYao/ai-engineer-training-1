# pip install langchain_community datasets ragas

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness

data_samples = {
    'question': [
        '杭州最值得去的景点有哪些？',
        '去成都旅游的话，有哪些特色美食推荐？',
        '在西安游览时，参观兵马俑需要提前预约吗？'
    ],
    'answer': [
        '杭州西湖、灵隐寺和千岛湖是比较受欢迎的景点。',
        '成都有很多好吃的，比如火锅、串串香和担担面。',
        '参观兵马俑不需要预约，现场买票就可以进去。'
    ],
    'ground_truth': [
        '杭州必游景点包括西湖、灵隐寺、雷峰塔、千岛湖和宋城，其中西湖是国家5A级景区，建议清晨游览以避开人流。',
        '成都作为美食之都，推荐品尝火锅、串串香、担担面、龙抄手和钟水饺，宽窄巷子和锦里是集中体验地道小吃的好去处。',
        '参观秦始皇兵马俑博物馆必须通过官方平台提前实名预约购票，旺季时需至少提前3天预约，现场不保证有票。'
    ]
}

dataset = Dataset.from_dict(data_samples)
score = evaluate(
    dataset=dataset,
    metrics=[answer_correctness],
    llm=Tongyi(model_name="deepseek-v3.2"),
    embeddings=DashScopeEmbeddings(model="text-embedding-v4")
)
print(score.to_pandas())