from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser
from ext_template import PersonInfoPromptTemplate, PersonInfo

def demo_custom_template():
    """演示自定义模板的使用"""

    # 1. 创建自定义模板实例
    template = PersonInfoPromptTemplate(
        include_skills_analysis=True,
        include_career_advice=True,
        output_language="chinese"
    )

    # 2. 准备测试数据
    person_data = {
        "name": "张三",
        "age": 28,
        "occupation": "软件工程师",
        "skills": ["Python", "JavaScript", "React", "Docker"],
        "experience_years": 5,
        "location": "北京"
    }

    # 3. 测试不同分析类型
    analysis_types = ["basic", "career", "skills", "comprehensive"]

    for analysis_type in analysis_types:
        print(f"\n{'='*50}")
        print(f"分析类型: {analysis_type}")
        print('='*50)

        # 格式化提示
        prompt = template.format(
            person_info=person_data,
            analysis_type=analysis_type
        )

        print(prompt)
        print("\n" + "-"*30 + " 提示结束 " + "-"*30)

def demo_with_llm():
    """与 LLM 结合使用的完整示例"""

    # 初始化组件
    template = PersonInfoPromptTemplate(
        include_skills_analysis=True,
        include_career_advice=True
    )

    llm = Tongyi(temperature=0.3)
    parser = StrOutputParser()

    # 创建链
    chain = template | llm | parser

    # 测试数据
    person_data = PersonInfo(
        name="李四",
        age=32,
        occupation="产品经理",
        skills=["产品设计", "数据分析", "项目管理", "用户研究"],
        experience_years=8,
        location="上海"
    )

    # 执行分析
    result = chain.invoke({
        "person_info": person_data,
        "analysis_type": "comprehensive"
    })

    print("AI 分析结果:")
    print("="*50)
    print(result)

def demo_template_management():
    """演示模板配置管理"""

    # 创建模板
    template = PersonInfoPromptTemplate(
        template_type="senior_analysis",
        include_skills_analysis=True,
        include_career_advice=True,
        output_language="chinese"
    )

    # 保存配置
    template.save_template_config("person_template_config.json")
    print("模板配置已保存")

    # 加载配置
    loaded_template = PersonInfoPromptTemplate.load_template_config(
        "person_template_config.json"
    )
    print("模板配置已加载")

    # 验证配置
    print(f"模板类型: {loaded_template.template_type}")
    print(f"包含技能分析: {loaded_template.include_skills_analysis}")
    print(f"包含职业建议: {loaded_template.include_career_advice}")

if __name__ == "__main__":
    # 运行演示
    print("1. 基础模板演示")
    demo_custom_template()

    print("\n\n2. 与 LLM 结合使用")
    demo_with_llm()  # 需要配置 API Key

    print("\n\n3. 模板配置管理")
    demo_template_management()
