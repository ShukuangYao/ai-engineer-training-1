# 营养餐清单（加入业务逻辑）
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi

# 1. 初始化组件
parser = CommaSeparatedListOutputParser()
llm = Tongyi(temperature=0)

# 2. 创建营养均衡的提示模板
nutrition_prompt = PromptTemplate(
    template="""根据{meal_type}，生成一个包含5个食材的营养餐购物清单，需要满足以下要求：
1. 食材之间不能重复
2. 符合{meal_type}的热量要求：{calorie_range}卡路里
3. 营养均衡，包含蛋白质、碳水化合物、脂肪、维生素和矿物质
4. 食材要常见易购买
5. 避免过于油腻或高热量的食物
6. 请使用中文输出食材名称

{format_instructions}
""",
    input_variables=["meal_type", "calorie_range"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 3. 创建链
nutrition_chain = nutrition_prompt | llm | parser

# 4. 定义各餐次的热量范围
meal_calories = {
    "早餐": "300-400",
    "午餐": "500-600",
    "晚餐": "400-500"
}

# 5. 生成不同餐型的购物清单
shopping_lists = {}
all_items = set()  # 用于跟踪所有已使用的食材，避免重复

for meal, calorie_range in meal_calories.items():
    # 生成购物清单
    items = nutrition_chain.invoke({"meal_type": meal, "calorie_range": calorie_range})

    # 去重处理
    unique_items = []
    for item in items:
        if item not in all_items:
            unique_items.append(item)
            all_items.add(item)

    # 如果去重后不足5个，补充一些常见食材
    if len(unique_items) < 5:
        common_items = ["鸡蛋", "牛奶", "燕麦", "西兰花", "鸡胸肉", "糙米", "苹果", "香蕉"]
        for item in common_items:
            if item not in all_items and len(unique_items) < 5:
                unique_items.append(item)
                all_items.add(item)

    shopping_lists[meal] = unique_items
    print(f"{meal}购物清单: {shopping_lists[meal]}")

# 6. 格式化输出
print("\n=== 详细营养餐清单 ===")
for meal, items in shopping_lists.items():
    print(f"\n🍽️ {meal} (热量范围: {meal_calories[meal]}卡路里):")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")

# 7. 生成总购物清单（去重）
total_items = sorted(all_items)
print("\n=== 总购物清单（去重）===")
for i, item in enumerate(total_items, 1):
    print(f"  {i}. {item}")

if __name__ == "__main__":
    pass
