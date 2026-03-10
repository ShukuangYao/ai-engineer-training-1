"""
简洁版自定义 PromptTemplate 演示
展示如何继承 StringPromptTemplate 创建自定义模板
"""
from langchain_core.prompts import StringPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, List, Any

class PersonInfo(BaseModel):
    """人员信息数据模型"""
    name: str
    age: int
    job: str
    skills: List[str] = []

class PersonPromptTemplate(StringPromptTemplate):
    """自定义人员分析模板"""

    # 模板配置
    analysis_type: str = Field(default="basic")
    input_variables: List[str] = ["person_info"]

    def format(self, **kwargs: Any) -> str:
        """核心格式化方法"""
        person_info = kwargs.get("person_info")

        # 转换数据格式
        if isinstance(person_info, dict):
            person_info = PersonInfo(**person_info)

        # 构建提示
        skills_text = ", ".join(person_info.skills) if person_info.skills else "无"

        prompt = f"""请分析以下人员信息：
            姓名：{person_info.name}
            年龄：{person_info.age}岁
            职业：{person_info.job}
            技能：{skills_text}

            请提供简要的{self.analysis_type}分析。"""

        # 添加分析类型
        return prompt

def demo():
    """演示使用"""
    print("=== 自定义 PromptTemplate 演示 ===\n")

    # 1. 创建模板
    template = PersonPromptTemplate(analysis_type="职业发展")

    # 2. 准备数据
    person_data = {
        "name": "张三",
        "age": 28,
        "job": "Python开发工程师",
        "skills": ["Python", "Django", "MySQL"]
    }

    # 3. 生成提示
    prompt = template.format(person_info=person_data)

    print("生成的提示：")
    print("-" * 40)
    print(prompt)
    print("-" * 40)

    # 4. 展示不同分析类型
    print("\n不同分析类型对比：")
    for analysis in ["技能评估", "薪资建议", "转岗建议"]:
        template.analysis_type = analysis
        short_prompt = template.format(person_info=person_data)
        print(f"\n{analysis}:")
        print(short_prompt.split('\n')[-5:])  # 只显示最后五行

if __name__ == "__main__":
    demo()