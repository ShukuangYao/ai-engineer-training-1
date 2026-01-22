import sys
import os
import argparse
from pprint import pprint

# 确保我们可以从 src 导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dsl_parser import DSLParser
from src.graph_builder import GraphBuilder

def main():
    """LangGraph DSL 引擎的主入口函数。"""
    parser = argparse.ArgumentParser(description="LangGraph DSL 引擎")
    parser.add_argument("file", help="DSL YAML 文件的路径")
    args = parser.parse_args()

    # 1. 解析 DSL
    print(f"正在从 {args.file} 加载 DSL...")
    try:
        dsl_parser = DSLParser()
        dsl_data = dsl_parser.load_file(args.file)
    except Exception as e:
        print(f"加载 DSL 出错: {e}")
        return

    # 2. 构建图 (LangGraph)
    print("正在构建 LangGraph...")
    try:
        builder = GraphBuilder(dsl_data)
        graph = builder.build()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"构建图出错: {e}")
        return

    # 3. 运行图
    print("正在运行工作流...")
    initial_state = {
        "context": {},
        "logs": [],
        "__router__": ""
    }
    
    try:
        final_state = graph.invoke(initial_state)
        
        print("\n--- 执行日志 ---")
        for log in final_state["logs"]:
            print(log)
            
        print("\n--- 最终上下文 ---")
        pprint(final_state["context"])
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"运行时错误: {e}")

if __name__ == "__main__":
    main()
