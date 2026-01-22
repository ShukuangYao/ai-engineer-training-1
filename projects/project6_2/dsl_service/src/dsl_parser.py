import yaml
import os

class DSLParser:
    """DSL 解析器类，负责加载和验证 YAML 格式的 DSL 文件。"""
    
    def __init__(self):
        pass

    def load_file(self, file_path):
        """加载 DSL 文件。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到 DSL 文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"YAML 解析错误: {e}")
        
        self.validate(data)
        return data

    def validate(self, data):
        """验证 DSL 数据的结构完整性。"""
        if "graph" not in data:
            raise ValueError("无效的 DSL: 缺少 'graph' 键。")
        
        graph = data["graph"]
        required_keys = ["name", "start_node", "nodes"]
        for key in required_keys:
            if key not in graph:
                raise ValueError(f"无效的 DSL: Graph 缺少 '{key}'。")
        
        nodes = graph["nodes"]
        if not isinstance(nodes, list):
            raise ValueError("无效的 DSL: 'nodes' 必须是一个列表。")
        
        node_ids = set()
        for node in nodes:
            if "id" not in node:
                raise ValueError("无效的 DSL: 节点缺少 'id'。")
            if "type" not in node:
                raise ValueError(f"无效的 DSL: 节点 '{node['id']}' 缺少 'type'。")
            
            node_ids.add(node["id"])
            
            node_type = node["type"]
            if node_type == "task":
                # 任务节点可能有工具调用，这里暂时不做深度校验
                pass 
            elif node_type == "condition":
                if "condition" not in node:
                    raise ValueError(f"无效的 DSL: 条件节点 '{node['id']}' 缺少 'condition' 块。")
                cond = node["condition"]
                if "expression" not in cond or "then" not in cond or "else" not in cond:
                    raise ValueError(f"无效的 DSL: 条件节点 '{node['id']}' 不完整（需要 expression, then, else）。")
            elif node_type == "end":
                pass
            else:
                raise ValueError(f"无效的 DSL: 未知的节点类型 '{node_type}'。")

        # 验证 start_node 是否存在
        if graph["start_node"] not in node_ids:
            raise ValueError(f"无效的 DSL: start_node '{graph['start_node']}' 未在 nodes 中找到。")
        
        # 验证跳转（基本检查）
        for node in nodes:
            if "next" in node:
                if node["next"] not in node_ids:
                    raise ValueError(f"无效的 DSL: 节点 '{node['id']}' 具有无效的 next 跳转 '{node['next']}'。")
            
            if node["type"] == "condition":
                cond = node["condition"]
                if cond["then"] not in node_ids:
                    raise ValueError(f"无效的 DSL: 节点 '{node['id']}' 具有无效的 'then' 跳转 '{cond['then']}'。")
                if cond["else"] not in node_ids:
                    raise ValueError(f"无效的 DSL: 节点 '{node['id']}' 具有无效的 'else' 跳转 '{cond['else']}'。")

        print("DSL 验证成功。")
