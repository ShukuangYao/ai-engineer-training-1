"""
vLLM 自定义封装器演示
展示不同参数配置对生成结果的影响
"""
from custom_vllm_wrapper import CustomVLLMWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import json
import time

def demo_basic_usage():
    """基础使用演示"""
    print("=== 基础使用演示 ===\n")

    # 创建 vLLM 实例
    llm = CustomVLLMWrapper(
        model_name="Qwen2-7B-Instruct",
        base_url="http://localhost:8000",
        temperature=0.7,
        max_tokens=100
    )

    # 简单调用
    prompt = "请用一句话介绍人工智能："
    print(f"提示: {prompt}")

    try:
        # 注意：这里会尝试连接实际的 vLLM 服务
        # 如果没有运行 vLLM 服务，会显示模拟结果
        result = llm.invoke(prompt)
        print(f"回答: {result}")
    except Exception as e:
        print(f"连接失败 (这是正常的，因为没有实际的 vLLM 服务): {e}")
        print("模拟回答: 人工智能是一种让计算机模拟人类智能行为的技术。")

    print("\n" + "="*50 + "\n")

def demo_parameter_comparison():
    """参数对比演示"""
    print("=== 参数对比演示 ===\n")

    # 不同温度参数的配置
    configs = [
        {"name": "保守型", "temperature": 0.1, "top_p": 0.8, "top_k": 10},
        {"name": "平衡型", "temperature": 0.7, "top_p": 0.9, "top_k": 50},
        {"name": "创意型", "temperature": 1.2, "top_p": 0.95, "top_k": 100},
    ]

    prompt = "写一个关于春天的短诗："

    for config in configs:
        print(f" {config['name']} 配置:")
        print(f"   Temperature: {config['temperature']}")
        print(f"   Top-P: {config['top_p']}")
        print(f"   Top-K: {config['top_k']}")

        llm = CustomVLLMWrapper(
            model_name="qwen-plus",
            base_url="http://localhost:8000",
            **config
        )

        result = llm.invoke(prompt)

        print(f"   提示: {prompt}")
        print(f"   回答: {result}")

    print("="*50 + "\n")

def demo_advanced_parameters():
    """高级参数演示"""
    print("=== 高级参数演示 ===\n")

    # 展示各种高级参数的作用
    advanced_configs = {
        "重复控制": {
            "repetition_penalty": 1.2,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "description": "减少重复内容，提高文本多样性"
        },
        "精确控制": {
            "temperature": 0.3,
            "top_p": 0.8,
            "min_p": 0.05,
            "description": "更精确的生成控制"
        },
        "束搜索": {
            "use_beam_search": True,
            "best_of": 3,
            "length_penalty": 1.2,
            "description": "使用束搜索获得更好的结果"
        }
    }
    # 束搜索：束搜索是一种启发式搜索算法，用于在大语言模型生成文本时找到更优质的输出序列。
    # 它是贪心搜索和穷举搜索之间的折中方案。

    for config_name, config in advanced_configs.items():
        print(f" {config_name} 配置:")
        description = config.pop("description")
        print(f"   说明: {description}")

        for param, value in config.items():
            print(f"   {param}: {value}")

        llm = CustomVLLMWrapper(
            model_name="Qwen2-7B-Instruct",
            base_url="http://localhost:8000",
            max_tokens=150,
            **config
        )

        # 显示参数摘要
        summary = llm.get_params_summary()
        print(f"   当前配置摘要: {json.dumps(summary['核心参数'], ensure_ascii=False)}")
        print()

    print("="*50 + "\n")

def demo_streaming():
    """流式输出演示"""
    print("=== 流式输出演示 ===\n")

    llm = CustomVLLMWrapper(
        model_name="Qwen2-7B-Instruct",
        base_url="http://localhost:8000",
        temperature=0.8,
        max_tokens=200
    )

    prompt = "请详细解释什么是机器学习："
    print(f"提示: {prompt}")
    print("流式回答: ", end="", flush=True)

    try:
        # 模拟流式输出
        simulated_response = "机器学习是人工智能的一个重要分支，它让计算机能够从数据中自动学习和改进，而无需明确编程。"

        for char in simulated_response:
            print(char, end="", flush=True)
            time.sleep(0.05)  # 模拟流式输出的延迟
        print("\n")

        # 实际的流式调用代码（需要真实服务）
        # for chunk in llm.stream(prompt):
        #     print(chunk, end="", flush=True)

    except Exception as e:
        print(f"\n流式输出失败: {e}")

    print("\n" + "="*50 + "\n")

def demo_langchain_integration():
    """与 LangChain 集成演示"""
    print("=== LangChain 集成演示 ===\n")

    # 创建提示模板
    template = PromptTemplate.from_template(
        "作为一个{role}，请回答以下问题：{question}"
    )

    # 创建 LLM
    llm = CustomVLLMWrapper(
        model_name="Qwen2-7B-Instruct",
        base_url="http://localhost:8000",
        temperature=0.6,
        max_tokens=200,
        top_p=0.9
    )

    # 创建输出解析器
    parser = StrOutputParser()

    # 创建链
    chain = template | llm | parser

    # 测试不同角色
    test_cases = [
        {"role": "Python专家", "question": "如何优化Python代码性能？"},
        {"role": "产品经理", "question": "如何设计用户友好的界面？"},
        {"role": "数据科学家", "question": "如何选择合适的机器学习算法？"}
    ]

    for case in test_cases:
        print(f" 角色: {case['role']}")
        print(f" 问题: {case['question']}")

        try:
            # result = chain.invoke(case)
            # print(f" 回答: {result}")
            print(f" 模拟回答: [作为{case['role']}的专业回答]")
        except Exception as e:
            print(f" 调用失败: {e}")
        print()

    print("="*50 + "\n")

def demo_parameter_validation():
    """参数验证演示"""
    print("=== 参数验证演示 ===\n")

    print(" 测试参数验证功能:")

    # 测试有效参数
    try:
        llm = CustomVLLMWrapper(
            model_name="test-model",
            temperature=0.8,
            top_p=0.9,
            top_k=50
        )
        print(" 有效参数配置成功")
    except Exception as e:
        print(f" 有效参数配置失败: {e}")

    # 测试无效参数
    invalid_configs = [
        {"temperature": 3.0, "error": "temperature 超出范围"},
        {"top_p": 1.5, "error": "top_p 超出范围"},
        {"top_k": 0, "error": "top_k 小于最小值"}
    ]

    for config in invalid_configs:
        try:
            error_desc = config.pop("error")
            llm = CustomVLLMWrapper(
                model_name="test-model",
                **config
            )
            print(f" 应该失败但成功了: {error_desc}")
        except Exception as e:
            print(f" 正确捕获错误: {error_desc} - {str(e)[:50]}...")

    print("\n" + "="*50 + "\n")

def main():
    """主演示函数"""
    print(" vLLM 自定义封装器完整演示\n")

    # 运行所有演示
    demo_basic_usage()
    demo_parameter_comparison()
    demo_advanced_parameters()
    demo_streaming()
    demo_langchain_integration()
    demo_parameter_validation()

    print(" 演示完成！")
    print("\n 注意事项:")
    print("1. 实际使用需要启动 vLLM 服务")
    print("2. 根据模型调整参数范围")
    print("3. 监控生成质量和性能")
    print("4. 合理设置超时和重试参数")

if __name__ == "__main__":
    main()