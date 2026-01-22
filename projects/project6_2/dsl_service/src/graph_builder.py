from typing import Dict, Any, TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from .tools import get_tool

class GraphState(TypedDict):
    """图状态定义。"""
    context: Dict[str, Any]                  # 上下文数据，用于存储工具结果和中间变量
    logs: Annotated[List[str], operator.add] # 执行日志，追加模式
    __router__: str                          # 内部路由变量，用于条件分支跳转

class GraphBuilder:
    """图构建器：将 DSL 解析结果转换为 LangGraph 的 StateGraph。"""
    
    def __init__(self, dsl_data):
        self.dsl = dsl_data["graph"]
        self.nodes = {n["id"]: n for n in self.dsl["nodes"]}
        self.workflow = StateGraph(GraphState)

    def build(self):
        """构建并编译 LangGraph 工作流。"""
        # 1. 添加所有节点
        for node_def in self.dsl["nodes"]:
            node_id = node_def["id"]
            node_type = node_def["type"]
            
            if node_type == "task":
                self.workflow.add_node(node_id, self._create_task_handler(node_def))
            elif node_type == "condition":
                self.workflow.add_node(node_id, self._create_condition_handler(node_def))
            elif node_type == "end":
                self.workflow.add_node(node_id, self._create_end_handler(node_id))
            else:
                print(f"警告: 未知的节点类型 {node_type}")

        # 2. 添加边 (连接节点)
        for node_def in self.dsl["nodes"]:
            node_id = node_def["id"]
            node_type = node_def["type"]

            if node_type == "task":
                if "next" in node_def:
                    next_id = node_def["next"]
                    if self._is_end_node(next_id):
                        # 如果下一个节点显式定义为 'end' 类型，链接到它
                        # 如果它映射到系统 END，链接到 END
                        if next_id in self.nodes:
                            self.workflow.add_edge(node_id, next_id)
                        else:
                            self.workflow.add_edge(node_id, END)
                    else:
                        self.workflow.add_edge(node_id, next_id)
                else:
                    self.workflow.add_edge(node_id, END)

            elif node_type == "condition":
                cond_def = node_def["condition"]
                then_node = cond_def["then"]
                else_node = cond_def["else"]
                
                mapping = {
                    "then": then_node if then_node in self.nodes else END,
                    "else": else_node if else_node in self.nodes else END
                }
                
                self.workflow.add_conditional_edges(
                    node_id,
                    lambda state: state.get("__router__", "else"),
                    mapping
                )
            
            elif node_type == "end":
                 self.workflow.add_edge(node_id, END)

        self.workflow.set_entry_point(self.dsl["start_node"])
        return self.workflow.compile()

    def _is_end_node(self, node_id):
        """检查节点 ID 是否指向结束节点。"""
        if node_id == "END": return True
        node = self.nodes.get(node_id)
        return node and node["type"] == "end"

    def _create_task_handler(self, node_def):
        """创建任务节点的处理函数。"""
        def handler(state: GraphState):
            tool_name = node_def.get("tool")
            tool_args = node_def.get("args", {})
            
            # 解析参数中的变量引用 (例如 "$var_name")
            final_args = {}
            current_context = state.get("context", {})
            
            for k, v in tool_args.items():
                if isinstance(v, str) and v.startswith("$"):
                    key = v[1:]
                    val = current_context.get(key)
                    # 如果找不到 key，尝试嵌套查找或返回 None (这里简化处理)
                    final_args[k] = val
                else:
                    final_args[k] = v

            tool_func = get_tool(tool_name)
            log_prefix = f"[节点: {node_def['id']}]"
            
            if tool_func:
                try:
                    result = tool_func(final_args)
                    log = f"{log_prefix} 执行工具 '{tool_name}'。结果: {result}"
                    
                    # 更新上下文
                    new_context = current_context.copy()
                    new_context.update(result)
                    
                    return {"context": new_context, "logs": [log]}
                except Exception as e:
                    log = f"{log_prefix} 执行工具 '{tool_name}' 出错: {e}"
                    return {"logs": [log]}
            else:
                log = f"{log_prefix} 未找到工具 '{tool_name}'。"
                return {"logs": [log]}
        return handler

    def _create_condition_handler(self, node_def):
        """创建条件节点的处理函数。"""
        def handler(state: GraphState):
            expression = node_def["condition"]["expression"]
            context = state.get("context", {})
            log_prefix = f"[节点: {node_def['id']}]"
            
            try:
                # 使用上下文评估表达式
                # 允许的全局变量：安全的内置函数
                safe_globals = {"__builtins__": {
                    "str": str, "int": int, "float": float, "bool": bool, 
                    "len": len, "list": list, "dict": dict, "set": set,
                    "min": min, "max": max, "sum": sum, "abs": abs
                }}
                result = eval(expression, safe_globals, context)
                branch = "then" if result else "else"
                log = f"{log_prefix} 条件 '{expression}' -> {result} -> 跳转至 {branch}"
            except Exception as e:
                log = f"{log_prefix} 条件错误: {e}。默认为 'else'。"
                branch = "else"
            
            return {"logs": [log], "__router__": branch}
        return handler

    def _create_end_handler(self, node_id):
        """创建结束节点的处理函数。"""
        def handler(state: GraphState):
            return {"logs": [f"到达结束节点 '{node_id}'"]}
        return handler
