#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于规则的意图识别系统 - LangChain 风格实现 (完整注释版)
========================================================

系统架构说明:
1. 采用多策略融合的方式进行意图识别
2. 支持正则匹配、关键词匹配两种主要识别方式
3. 包含槽位填充功能，提取关键参数信息
4. 使用 LangChain 风格的链式调用设计模式

核心组件:
- RegexIntentParser: 正则表达式意图解析器
- KeywordIntentParser: 关键词权重意图解析器  
- SlotExtractor: 槽位信息提取器
- RuleBasedIntentChain: 主要的意图识别链

作者: AI工程化训练营
版本: 1.0
日期: 2025年
"""

import re
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass

try:
    from langgraph.graph import StateGraph, START, END
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    StateGraph = START = END = None

@dataclass
class IntentResult:
    """
    意图识别结果数据类
    ================
    
    用于封装单个解析器的识别结果，包含以下信息:
    - intent: 识别出的意图类型 (如 'query_order', 'refund' 等)
    - confidence: 置信度分数 (0.0-1.0)
    - matched_rules: 匹配的规则列表 (用于可解释性)
    - extracted_entities: 提取的实体信息 (如订单号、时间等)
    """
    intent: str = "unknown"                    # 默认为未知意图
    confidence: float = 0.0                    # 默认置信度为0
    matched_rules: List[str] = None            # 匹配的规则列表
    extracted_entities: Optional[tuple] = None # 提取的实体元组
    
    def __post_init__(self):
        """数据类初始化后处理，确保 matched_rules 不为 None"""
        if self.matched_rules is None:
            self.matched_rules = []

class RegexIntentParser:
    """
    正则表达式意图解析器
    ==================
    
    功能说明:
    - 使用预定义的正则表达式模式匹配用户输入
    - 支持多种意图类型的精确匹配
    - 能够提取结构化信息(如订单号、数字等)
    - 具有最高的匹配优先级(置信度0.9)
    
    适用场景:
    - 结构化表达的识别 (如"订单号123456")
    - 固定格式的用户输入
    - 需要提取特定信息的场景
    """
    
    def __init__(self):
        """
        初始化正则模式字典
        
        模式设计原则:
        1. 使用 .* 匹配任意字符，增加灵活性
        2. 使用 (\d+) 捕获数字信息
        3. 使用 .*? 进行非贪婪匹配
        4. 按匹配精确度排序，精确的模式放在前面
        """
        self.patterns = {
            # 查询订单相关模式
            'query_order': [
                r'查.*订单.*(\d+)',      # 匹配: "查订单123" -> 提取数字
                r'订单号.*?(\d{6,})',     # 匹配: "订单号123456" -> 提取6位以上数字  
                r'我的订单.*状态'         # 匹配: "我的订单状态" -> 无提取
            ],
            # 退款相关模式
            'refund': [
                r'退.*款',               # 匹配: "退款"、"申请退款"
                r'取消.*订单',           # 匹配: "取消订单"、"取消这个订单"
                r'不要.*了'              # 匹配: "不要了"、"我不要这个了"
            ],
            # 开发票相关模式
            'issue_invoice': [
                r'开.*发票',             # 匹配: "开发票"、"帮我开个发票"
                r'要.*发票',             # 匹配: "要发票"、"我要发票"
                r'发票.*开'              # 匹配: "发票怎么开"
            ]
        }
    
    def parse(self, text: str) -> IntentResult:
        """
        解析文本并返回意图结果
        
        Args:
            text: 用户输入的文本
            
        Returns:
            IntentResult: 包含意图、置信度、匹配规则等信息的结果对象
            
        处理流程:
        1. 遍历所有意图类型
        2. 对每个意图的所有模式进行匹配
        3. 找到第一个匹配的模式就立即返回(优先级机制)
        4. 如果没有匹配，返回默认的未知意图结果
        """
        # 遍历所有意图类型和对应的正则模式
        for intent, patterns in self.patterns.items():
            # 遍历当前意图的所有正则模式
            for i, pattern in enumerate(patterns):
                # 执行正则匹配，忽略大小写
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # 匹配成功，构造并返回结果
                    return IntentResult(
                        intent=intent,                                    # 意图类型
                        confidence=0.9,                                   # 正则匹配的高置信度
                        matched_rules=[f"regex_{intent}_{i}"],           # 匹配规则标识
                        extracted_entities=match.groups() if match.groups() else None  # 提取的实体
                    )
        
        # 没有任何模式匹配，返回默认结果
        return IntentResult()

class KeywordIntentParser:
    """
    关键词权重意图解析器
    ==================
    
    功能说明:
    - 基于关键词权重打分机制进行意图识别
    - 支持主关键词和次关键词的分层权重设计
    - 通过累积得分确定最终意图
    - 提供中等置信度的识别结果
    
    设计思路:
    - 主关键词: 强相关词汇，权重较高(0.8)
    - 次关键词: 弱相关词汇，权重较低(0.4)  
    - 总分计算: 各匹配词汇权重之和，最大值截断为1.0
    
    适用场景:
    - 自然语言表达的意图识别
    - 模糊匹配和语义相关性判断
    - 正则匹配失败时的备选方案
    """
    
    def __init__(self):
        """
        初始化关键词权重配置
        
        配置结构说明:
        - primary: 主关键词列表，直接表达意图的核心词汇
        - secondary: 次关键词列表，间接相关的辅助词汇
        - weights: 权重配置，定义不同级别关键词的得分
        
        权重设计原则:
        - 主关键词权重(0.8): 单个词就能较强表达意图
        - 次关键词权重(0.4): 需要多个词组合才能确定意图
        - 总分上限(1.0): 避免过度累积导致的置信度失真
        """
        self.keywords = {
            # 查询订单意图的关键词配置
            'query_order': {
                'primary': ['查订单', '订单状态', '物流信息'],    # 直接表达查询意图
                'secondary': ['快递', '发货', '到了吗'],        # 间接相关的查询词汇
                'weights': {'primary': 0.8, 'secondary': 0.4}
            },
            # 退款意图的关键词配置
            'refund': {
                'primary': ['退钱', '退款', '退货'],           # 直接表达退款意图
                'secondary': ['不要', '取消', '退回'],         # 间接表达不满意的词汇
                'weights': {'primary': 0.8, 'secondary': 0.4}
            },
            # 开发票意图的关键词配置
            'issue_invoice': {
                'primary': ['开发票', '要发票', '发票'],        # 直接表达开票意图
                'secondary': ['报销', '开票'],                # 相关的财务词汇
                'weights': {'primary': 0.8, 'secondary': 0.4}
            }
        }
    
    def parse(self, text: str) -> IntentResult:
        """
        基于关键词权重解析意图
        
        Args:
            text: 用户输入的文本
            
        Returns:
            IntentResult: 包含意图、置信度、匹配词汇等信息的结果对象
            
        算法流程:
        1. 遍历所有意图类型
        2. 对每个意图计算关键词匹配得分
        3. 累积主关键词和次关键词的权重得分
        4. 选择得分最高的意图作为最终结果
        5. 如果没有任何匹配，返回未知意图
        """
        scores = {}  # 存储每个意图的得分信息
        
        # 遍历所有意图类型及其关键词配置
        for intent, config in self.keywords.items():
            score = 0                # 当前意图的累积得分
            matched_words = []       # 匹配到的关键词列表
            
            # 计算主关键词得分
            for word in config['primary']:
                if word in text:     # 简单的字符串包含匹配
                    score += config['weights']['primary']  # 累加主关键词权重
                    matched_words.append(word)             # 记录匹配的词汇
            
            # 计算次关键词得分
            for word in config['secondary']:
                if word in text:     # 简单的字符串包含匹配
                    score += config['weights']['secondary'] # 累加次关键词权重
                    matched_words.append(word)              # 记录匹配的词汇
            
            # 如果有匹配的关键词，记录该意图的得分信息
            if score > 0:
                scores[intent] = {
                    'score': min(score, 1.0),              # 得分上限截断为1.0
                    'matched_words': matched_words          # 保存匹配的词汇列表
                }
        
        # 如果有得分的意图，选择得分最高的作为结果
        if scores:
            # 找到得分最高的意图
            best_intent = max(scores.keys(), key=lambda x: scores[x]['score'])
            return IntentResult(
                intent=best_intent,                                    # 最佳意图
                confidence=scores[best_intent]['score'],               # 对应的置信度得分
                matched_rules=[f"keyword_{best_intent}"],             # 匹配规则标识
                extracted_entities=tuple(scores[best_intent]['matched_words'])  # 匹配的关键词
            )
        
        # 没有任何关键词匹配，返回默认的未知意图结果
        return IntentResult()

class SlotExtractor:
    """
    槽位信息提取器
    ==============
    
    功能说明:
    - 根据已识别的意图类型，提取执行该意图所需的参数信息
    - 使用正则表达式从用户输入中抽取结构化数据
    - 支持多种数据类型的提取(订单号、时间、金额等)
    
    槽位设计原则:
    - 每个意图类型对应一组特定的槽位
    - 槽位名称语义化，便于后续业务逻辑使用
    - 正则模式兼顾准确性和覆盖面
    
    应用场景:
    - 订单查询: 需要订单号、时间等参数
    - 退款申请: 需要订单号、退款原因、时间等
    - 开具发票: 需要订单号、金额等参数
    """
    
    def __init__(self):
        """
        初始化槽位提取模式配置
        
        配置结构说明:
        - 外层key: 意图类型 (如 'query_order')
        - 内层key: 槽位名称 (如 'order_id')  
        - 内层value: 正则表达式模式 (用于提取对应信息)
        
        正则模式设计要点:
        - 使用捕获组 () 提取目标信息
        - 考虑中文表达的多样性
        - 平衡精确度和召回率
        """
        self.slot_patterns = {
            # 查询订单意图的槽位配置
            'query_order': {
                'order_id': r'(\d{6,})',                    # 提取6位以上数字作为订单号
                'time': r'(昨天|今天|前天|上周|本月)'        # 提取时间表达
            },
            # 退款意图的槽位配置  
            'refund': {
                'order_id': r'订单.*?(\d{6,})',             # 在"订单"关键词后提取数字
                'reason': r'因为(.*?)所以',                  # 提取"因为...所以"中的原因
                'time': r'(昨天|今天|前天).*下.*单'          # 提取下单时间表达
            },
            # 开发票意图的槽位配置
            'issue_invoice': {
                'order_id': r'(\d{6,})',                    # 提取订单号
                'amount': r'(\d+\.?\d*)元'                  # 提取金额数字(支持小数)
            }
        }
    
    def extract_slots(self, text: str, intent: str) -> Dict[str, str]:
        """
        根据意图类型提取槽位信息
        
        Args:
            text: 用户输入的原始文本
            intent: 已识别的意图类型
            
        Returns:
            Dict[str, str]: 槽位名称到提取值的映射字典
            
        提取流程:
        1. 检查意图类型是否在配置中存在
        2. 遍历该意图对应的所有槽位模式
        3. 对每个槽位执行正则匹配
        4. 将匹配成功的结果保存到字典中
        5. 返回包含所有提取信息的槽位字典
        
        注意事项:
        - 如果意图类型不存在，返回空字典
        - 如果某个槽位匹配失败，该槽位不会出现在结果中
        - 只提取正则捕获组中的内容 (match.group(1))
        """
        slots = {}  # 初始化槽位结果字典
        
        # 检查当前意图是否有对应的槽位配置
        if intent in self.slot_patterns:
            patterns = self.slot_patterns[intent]  # 获取该意图的槽位模式
            
            # 遍历所有槽位，尝试提取信息
            for slot_name, pattern in patterns.items():
                # 执行正则匹配
                match = re.search(pattern, text)
                if match:
                    # 匹配成功，提取捕获组的内容
                    slots[slot_name] = match.group(1)
                    # 注意: match.group(1) 获取第一个捕获组的内容
                    # 如果需要多个捕获组，可以使用 match.groups()
        
        return slots  # 返回提取到的槽位信息字典

class RuleBasedIntentChain:
    """
    LangChain 风格的意图识别主链
    ===========================
    
    系统架构说明:
    - 采用 LangChain 的链式调用设计模式
    - 集成多个解析器组件，实现模块化架构
    - 支持并行处理和智能融合决策
    - 提供完整的意图识别和槽位填充功能
    
    核心特性:
    1. 多策略融合: 正则匹配 + 关键词匹配
    2. 智能决策: 基于置信度和规则优先级
    3. 槽位提取: 自动提取业务参数
    4. 可解释性: 提供详细的推理过程
    
    工作流程:
    输入文本 → 并行解析 → 结果融合 → 槽位提取 → 推理解释 → 输出结果
    """
    
    def __init__(self):
        """
        初始化意图识别链的各个组件
        
        组件说明:
        - regex_parser: 正则表达式解析器，处理结构化输入
        - keyword_parser: 关键词解析器，处理自然语言输入  
        - slot_extractor: 槽位提取器，提取业务参数
        
        设计优势:
        - 组件解耦: 各解析器独立工作，便于维护和扩展
        - 职责分离: 每个组件专注于特定的识别策略
        - 易于测试: 可以单独测试每个组件的功能
        """
        self.regex_parser = RegexIntentParser()      # 正则表达式意图解析器
        self.keyword_parser = KeywordIntentParser()  # 关键词权重意图解析器
        self.slot_extractor = SlotExtractor()        # 槽位信息提取器
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整的意图识别流程
        
        Args:
            input_dict: 输入字典，必须包含 'text' 键
            
        Returns:
            Dict[str, Any]: 包含完整识别结果的字典
            
        返回字段说明:
        - intent: 识别的意图类型
        - confidence: 置信度分数 (0.0-1.0)
        - slots: 提取的槽位信息字典
        - matched_rules: 匹配的规则列表
        - extracted_entities: 提取的实体信息
        - reasoning: 推理过程的文字描述
        
        处理流程详解:
        1. 输入验证: 从输入字典中提取文本
        2. 并行解析: 同时运行正则和关键词解析器
        3. 结果融合: 根据策略选择最佳识别结果
        4. 槽位提取: 基于意图类型提取相关参数
        5. 推理生成: 生成可解释的推理过程
        6. 结果封装: 将所有信息整合为输出字典
        """
        # 步骤1: 提取输入文本，提供默认值避免KeyError
        text = input_dict.get("text", "")
        
        # 步骤2: 并行执行多个解析器
        # 注意: 这里是"并行"的概念，实际是顺序执行，但逻辑上独立
        regex_result = self.regex_parser.parse(text)      # 正则匹配解析
        keyword_result = self.keyword_parser.parse(text)  # 关键词匹配解析
        
        # 步骤3: 融合多个解析器的结果
        # 使用智能策略选择最佳结果
        final_result = self._merge_results([regex_result, keyword_result])
        
        # 步骤4: 基于最终意图提取槽位信息
        slots = self.slot_extractor.extract_slots(text, final_result.intent)
        
        # 步骤5: 生成人类可读的推理解释
        reasoning = self._generate_reasoning(final_result)
        
        # 步骤6: 构造并返回完整的结果字典
        return {
            "intent": final_result.intent,                    # 最终识别的意图
            "confidence": final_result.confidence,            # 置信度分数
            "slots": slots,                                   # 提取的槽位参数
            "matched_rules": final_result.matched_rules,      # 匹配的规则标识
            "extracted_entities": final_result.extracted_entities,  # 提取的实体
            "reasoning": reasoning                            # 推理过程说明
        }
    
    def _merge_results(self, results: List[IntentResult]) -> IntentResult:
        """
        融合多个解析器的识别结果
        
        Args:
            results: 各个解析器返回的结果列表
            
        Returns:
            IntentResult: 融合后的最终识别结果
            
        融合策略说明:
        1. 优先级策略: 正则匹配 > 关键词匹配
        2. 置信度阈值: 正则匹配置信度 > 0.8 时直接采用
        3. 最优选择: 其他情况选择置信度最高的结果
        4. 兜底机制: 无有效结果时返回未知意图
        
        设计理念:
        - 正则匹配精确度高，优先级最高
        - 关键词匹配覆盖面广，作为补充
        - 置信度机制确保结果质量
        - 兜底策略保证系统稳定性
        """
        # 步骤1: 过滤掉未知意图的结果
        # 只保留有效的识别结果进行后续处理
        valid_results = [r for r in results if r.intent != "unknown"]
        
        # 步骤2: 如果没有有效结果，返回默认的未知意图
        if not valid_results:
            return IntentResult()
        
        # 步骤3: 正则匹配优先策略
        # 如果正则匹配的置信度足够高(>0.8)，直接采用
        regex_results = [r for r in valid_results 
                        if any("regex" in rule for rule in r.matched_rules)]
        if regex_results and regex_results[0].confidence > 0.8:
            return regex_results[0]
        
        # 步骤4: 置信度最优策略
        # 选择所有有效结果中置信度最高的
        best_result = max(valid_results, key=lambda x: x.confidence)
        return best_result
    
    def _generate_reasoning(self, result: IntentResult) -> str:
        """
        生成人类可读的推理解释
        
        Args:
            result: 最终的识别结果
            
        Returns:
            str: 推理过程的文字描述
            
        功能说明:
        - 提供系统决策的透明度
        - 帮助用户理解识别过程
        - 便于系统调试和优化
        - 增强用户对系统的信任度
        
        解释内容包括:
        - 使用的识别方法(正则/关键词)
        - 识别的意图类型
        - 对应的置信度分数
        """
        # 处理未知意图的情况
        if result.intent == "unknown":
            return "未匹配到任何规则"
        
        # 判断使用的识别方法
        rule_type = ("正则匹配" if any("regex" in rule for rule in result.matched_rules) 
                    else "关键词匹配")
        
        # 生成格式化的推理说明
        return f"通过{rule_type}识别为{result.intent}，置信度{result.confidence:.2f}"

class FSMProcessor:
    """
    有限状态机处理器 - 多轮对话状态管理
    ===================================
    
    功能说明:
    - 管理多轮对话中的状态转换
    - 支持复杂的业务流程建模
    - 提供上下文相关的意图识别
    - 可扩展的状态机架构设计
    
    应用场景:
    - 多步骤的业务流程 (如退款申请的多个确认步骤)
    - 上下文相关的对话管理
    - 复杂业务逻辑的状态跟踪
    - 用户引导和流程控制
    
    设计思路:
    - 状态定义: 每个状态代表对话中的一个阶段
    - 转换规则: 定义状态之间的合法转换路径
    - 上下文管理: 维护对话历史和用户信息
    - 扩展性: 支持动态添加新的状态和转换
    
    注意: 当前为简化实现，实际项目中可扩展为完整的状态机
    """
    
    def __init__(self):
        """
        初始化状态机配置
        
        状态机设计说明:
        - start: 初始状态，用户刚开始对话
        - order_query: 订单查询状态，可进一步询问详情
        - refund_request: 退款申请状态，需要收集退款信息
        - invoice_request: 开票申请状态，需要收集开票信息
        
        转换路径设计:
        - 从start可以转换到任何业务状态
        - 每个业务状态有对应的子状态用于细化流程
        - 支持状态回退和跳转(在实际实现中)
        """
        self.states = {
            # 初始状态: 对话开始，等待用户表达意图
            'start': {
                'transitions': ['order_query', 'refund_request', 'invoice_request']
            },
            # 订单查询状态: 用户想查询订单信息
            'order_query': {
                'transitions': ['order_detail', 'logistics_query']  # 可查询详情或物流
            },
            # 退款申请状态: 用户想申请退款
            'refund_request': {
                'transitions': ['refund_reason', 'refund_confirm']  # 需要原因和确认
            },
            # 开票申请状态: 用户想开具发票
            'invoice_request': {
                'transitions': ['invoice_detail', 'invoice_confirm']  # 需要详情和确认
            }
        }
        self.current_state = 'start'  # 初始状态设为开始状态
        # 集成意图识别链 (复用 RegexIntentParser + KeywordIntentParser + SlotExtractor)
        self.intent_chain = RuleBasedIntentChain()
        # 意图类型到状态机状态的映射
        self._intent_to_state = {
            'query_order': 'order_query',
            'refund': 'refund_request',
            'issue_invoice': 'invoice_request'
        }
        # 子状态触发关键词: 当前状态 -> [ (关键词列表, 下一状态) ]，按步推进直至完成
        self._substate_triggers = {
            'order_query': [
                (['物流', '快递', '发货', '到了吗'], 'logistics_query'),
                (['详情', '订单详情', '具体'], 'order_detail'),
            ],
            'refund_request': [
                (['原因', '因为', '理由'], 'refund_reason'),
                (['确认', '好的', '同意', '可以'], 'refund_confirm'),
            ],
            'refund_reason': [
                (['确认', '好的', '同意', '可以', '提交'], 'refund_confirm'),
            ],
            'invoice_request': [
                (['详情', '抬头', '金额', '元'], 'invoice_detail'),
                (['确认', '好的', '同意', '开吧'], 'invoice_confirm'),
            ],
            'invoice_detail': [
                (['确认', '好的', '同意', '开吧'], 'invoice_confirm'),
            ],
        }
        # 流程终点状态：进入后表示该意图已完成，下一轮将重置为 start
        self._done_states = ('order_done', 'refund_done', 'invoice_done')
        # 每步的引导话术（系统应说的下一句，引导用户完成当前意图）
        self._state_prompts = {
            'start': '',
            'order_query': '请问您要查询订单详情还是物流信息？',
            'order_detail': '已为您查询订单详情。',
            'logistics_query': '已为您查询到物流信息。',
            'order_done': '订单查询已完成，如需其他帮助请继续说。',
            'refund_request': '请提供退款原因或订单号，也可直接说「确认」提交申请。',
            'refund_reason': '请确认是否提交退款申请？',
            'refund_confirm': '已收到您的确认。',
            'refund_done': '您的退款申请已提交，请等待审核。',
            'invoice_request': '请提供开票金额或抬头，也可直接说「开吧」确认开票。',
            'invoice_detail': '请确认是否按上述信息开票？',
            'invoice_confirm': '已收到您的确认。',
            'invoice_done': '发票将按您填写的信息开具并发送。',
        }
    
    def process(self, text: str, context: Dict = None) -> Optional[IntentResult]:
        """
        状态机处理逻辑
        
        Args:
            text: 用户当前输入的文本
            context: 对话上下文信息 (包括历史状态、用户信息等)
            
        Returns:
            Optional[IntentResult]: 基于状态机的识别结果，当前返回None
            
        实现思路 (当前为占位符，可扩展):
        1. 根据当前状态和用户输入判断下一步动作
        2. 检查状态转换的合法性
        3. 更新状态机的当前状态
        4. 返回对应的意图识别结果
        5. 维护对话上下文信息
        
        扩展方向:
        - 实现完整的状态转换逻辑
        - 添加状态转换条件判断
        - 集成上下文信息管理
        - 支持状态回退和异常处理
        """
        context = context if context is not None else {}
        # 累积槽位：跨轮次合并
        if "slots" not in context:
            context["slots"] = {}
        # 上一轮已到达流程终点则重置，以便本轮从 start 开始新意图
        if self.current_state in self._done_states:
            self.current_state = 'start'

        # 使用意图识别链得到基础意图与槽位
        chain_result = self.intent_chain.invoke({"text": text})
        intent = chain_result["intent"]
        confidence = chain_result["confidence"]
        slots = chain_result.get("slots", {})
        context["slots"].update(slots)
        matched_rules = chain_result.get("matched_rules", [])
        extracted_entities = chain_result.get("extracted_entities")
        context["is_completed"] = False
        context["next_prompt"] = ""

        # 1. 当前在 start：根据识别出的意图做状态转换
        if self.current_state == 'start':
            if intent == "unknown":
                context["next_prompt"] = "请问您需要什么帮助？可以查订单、申请退款或开发票。"
                return IntentResult()
            next_state = self._intent_to_state.get(intent)
            if next_state and next_state in self.states:
                self.current_state = next_state
            context["current_state"] = self.current_state
            context["next_prompt"] = self._state_prompts.get(self.current_state, "")
            return IntentResult(
                intent=intent,
                confidence=confidence,
                matched_rules=matched_rules,
                extracted_entities=extracted_entities
            )

        # 2. 当前在业务状态：先检查是否命中子状态触发词，一步步推进
        _substates_of = {
            'order_query': ['order_detail', 'logistics_query'],
            'refund_request': ['refund_reason', 'refund_confirm'],
            'invoice_request': ['invoice_detail', 'invoice_confirm'],
        }
        if self.current_state in self._substate_triggers:
            for keywords, sub_state in self._substate_triggers[self.current_state]:
                if any(kw in text for kw in keywords):
                    self.current_state = sub_state
                    context["current_state"] = self.current_state
                    # 订单：进入详情/物流即视为可返回结果，本步完成
                    if sub_state in ('logistics_query', 'order_detail'):
                        self.current_state = 'order_done'
                        context["current_state"] = 'order_done'
                        context["is_completed"] = True
                        context["next_prompt"] = self._state_prompts.get('order_done', '')
                    # 退款：进入确认态且用户说确认 → 流程完成
                    elif sub_state == 'refund_confirm':
                        if any(k in text for k in ['确认', '好的', '同意', '可以']):
                            self.current_state = 'refund_done'
                            context["current_state"] = 'refund_done'
                            context["is_completed"] = True
                            context["next_prompt"] = self._state_prompts.get('refund_done', '')
                        else:
                            context["next_prompt"] = self._state_prompts.get('refund_confirm', '')
                    # 开票：进入确认态且用户说确认/开吧 → 流程完成
                    elif sub_state == 'invoice_confirm':
                        if any(k in text for k in ['确认', '好的', '同意', '开吧']):
                            self.current_state = 'invoice_done'
                            context["current_state"] = 'invoice_done'
                            context["is_completed"] = True
                            context["next_prompt"] = self._state_prompts.get('invoice_done', '')
                        else:
                            context["next_prompt"] = self._state_prompts.get('invoice_confirm', '')
                    else:
                        context["next_prompt"] = self._state_prompts.get(sub_state, '')
                    break
            else:
                # 未命中子状态触发词，仅引导下一步
                context["next_prompt"] = self._state_prompts.get(self.current_state, '')
        else:
            context["next_prompt"] = self._state_prompts.get(self.current_state, '')

        context["current_state"] = self.current_state
        # 用户表达新主意图时切换到对应状态；若当前已是该意图下的子状态则不覆盖
        if intent != "unknown" and intent in self._intent_to_state:
            new_state = self._intent_to_state[intent]
            if new_state in self.states and new_state != self.current_state:
                if self.current_state not in _substates_of.get(new_state, []):
                    self.current_state = new_state
                    context["current_state"] = self.current_state
                    context["next_prompt"] = self._state_prompts.get(self.current_state, '')

        return IntentResult(
            intent=intent,
            confidence=confidence,
            matched_rules=matched_rules,
            extracted_entities=extracted_entities
        )

    def reset(self) -> None:
        """将状态机重置为初始状态 start，用于新会话或重新开始流程。"""
        self.current_state = 'start'


# =============================================================================
# LangGraph 多节点意图流程（可选，需安装 langgraph）
# =============================================================================
# 将上述状态机改造成 LangGraph 的 StateGraph：多节点 + 条件边，每节点只做一件事，
# 通过图结构表达「意图识别 → 按意图路由 → 各业务流节点 → 结束」。
# =============================================================================

class IntentGraphState(TypedDict, total=False):
    """LangGraph 图状态：在节点间传递的共享状态。"""
    text: str
    intent: str
    confidence: float
    slots: Dict[str, str]
    current_node: str
    next_prompt: str
    is_completed: bool
    matched_rules: List[str]
    extracted_entities: Optional[tuple]


def _build_intent_graph():
    """
    构建 LangGraph 意图识别与流程图。
    节点：intent_recognizer → 条件路由 → order_flow | refund_flow | invoice_flow | start_unknown → END
    """
    if not _LANGGRAPH_AVAILABLE:
        return None
    chain = RuleBasedIntentChain()
    intent_to_node = {"query_order": "order_query", "refund": "refund_request", "issue_invoice": "invoice_request"}
    done_states = {"order_done", "refund_done", "invoice_done"}
    state_prompts = {
        "start": "",
        "order_query": "请问您要查询订单详情还是物流信息？",
        "order_detail": "已为您查询订单详情。",
        "logistics_query": "已为您查询到物流信息。",
        "order_done": "订单查询已完成，如需其他帮助请继续说。",
        "refund_request": "请提供退款原因或订单号，也可直接说「确认」提交申请。",
        "refund_reason": "请确认是否提交退款申请？",
        "refund_confirm": "已收到您的确认。",
        "refund_done": "您的退款申请已提交，请等待审核。",
        "invoice_request": "请提供开票金额或抬头，也可直接说「开吧」确认开票。",
        "invoice_detail": "请确认是否按上述信息开票？",
        "invoice_confirm": "已收到您的确认。",
        "invoice_done": "发票将按您填写的信息开具并发送。",
    }
    substate_triggers = {
        "order_query": [
            (["物流", "快递", "发货", "到了吗"], "logistics_query"),
            (["详情", "订单详情", "具体"], "order_detail"),
        ],
        "refund_request": [
            (["原因", "因为", "理由"], "refund_reason"),
            (["确认", "好的", "同意", "可以"], "refund_confirm"),
        ],
        "refund_reason": [
            (["确认", "好的", "同意", "可以", "提交"], "refund_confirm"),
        ],
        "invoice_request": [
            (["详情", "抬头", "金额", "元"], "invoice_detail"),
            (["确认", "好的", "同意", "开吧"], "invoice_confirm"),
        ],
        "invoice_detail": [
            (["确认", "好的", "同意", "开吧"], "invoice_confirm"),
        ],
    }
    substates_of = {
        "order_query": ["order_detail", "logistics_query"],
        "refund_request": ["refund_reason", "refund_confirm"],
        "invoice_request": ["invoice_detail", "invoice_confirm"],
    }
    confirm_words = ["确认", "好的", "同意", "可以", "开吧", "提交"]

    def intent_recognizer_node(state: IntentGraphState) -> IntentGraphState:
        """节点：意图识别 + 若在完成态则重置，若在 start 则按意图进入业务节点。"""
        text = state.get("text", "")
        current = state.get("current_node", "start")
        slots = dict(state.get("slots") or {})
        if current in done_states:
            current = "start"
        result = chain.invoke({"text": text})
        slots.update(result.get("slots", {}))
        intent = result["intent"]
        confidence = result["confidence"]
        matched_rules = list(result.get("matched_rules") or [])
        extracted_entities = result.get("extracted_entities")
        next_node = current
        if current == "start" and intent != "unknown":
            next_node = intent_to_node.get(intent, "start")
        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
            "current_node": next_node,
            "matched_rules": matched_rules,
            "extracted_entities": extracted_entities,
            "next_prompt": "",
            "is_completed": False,
        }

    def route_after_intent(state: IntentGraphState) -> str:
        """条件路由：根据 current_node / intent 进入对应业务流节点。"""
        node = state.get("current_node", "start")
        intent = state.get("intent", "unknown")
        if node.startswith("order") or intent == "query_order":
            return "order_flow"
        if node.startswith("refund") or intent == "refund":
            return "refund_flow"
        if node.startswith("invoice") or intent == "issue_invoice":
            return "invoice_flow"
        return "start_unknown"

    def run_substate_triggers(text: str, current: str, triggers: dict) -> tuple:
        """根据当前节点和用户输入返回 (下一节点, 是否完成, 完成态名)."""
        if current not in triggers:
            return current, False, None
        for keywords, sub in triggers[current]:
            if any(kw in text for kw in keywords):
                if sub in ("logistics_query", "order_detail"):
                    return "order_done", True, "order_done"
                if sub == "refund_confirm" and any(c in text for c in confirm_words):
                    return "refund_done", True, "refund_done"
                if sub == "invoice_confirm" and any(c in text for c in confirm_words):
                    return "invoice_done", True, "invoice_done"
                return sub, False, None
        return current, False, None

    def order_flow_node(state: IntentGraphState) -> IntentGraphState:
        """节点：订单查询流（order_query → logistics/order_detail → order_done）."""
        text = state.get("text", "")
        current = state.get("current_node", "order_query")
        next_node, completed, _ = run_substate_triggers(text, current, substate_triggers)
        prompt = state_prompts.get(next_node, "")
        return {"current_node": next_node, "next_prompt": prompt, "is_completed": completed}

    def refund_flow_node(state: IntentGraphState) -> IntentGraphState:
        """节点：退款流（refund_request → refund_reason? → refund_confirm → refund_done）."""
        text = state.get("text", "")
        current = state.get("current_node", "refund_request")
        next_node, completed, _ = run_substate_triggers(text, current, substate_triggers)
        prompt = state_prompts.get(next_node, "")
        return {"current_node": next_node, "next_prompt": prompt, "is_completed": completed}

    def invoice_flow_node(state: IntentGraphState) -> IntentGraphState:
        """节点：开票流（invoice_request → invoice_detail? → invoice_confirm → invoice_done）."""
        text = state.get("text", "")
        current = state.get("current_node", "invoice_request")
        next_node, completed, _ = run_substate_triggers(text, current, substate_triggers)
        prompt = state_prompts.get(next_node, "")
        return {"current_node": next_node, "next_prompt": prompt, "is_completed": completed}

    def start_unknown_node(state: IntentGraphState) -> IntentGraphState:
        """节点：未识别意图时的引导。"""
        return {
            "next_prompt": "请问您需要什么帮助？可以查订单、申请退款或开发票。",
            "is_completed": False,
        }

    workflow = StateGraph(IntentGraphState)
    workflow.add_node("intent_recognizer", intent_recognizer_node)
    workflow.add_node("order_flow", order_flow_node)
    workflow.add_node("refund_flow", refund_flow_node)
    workflow.add_node("invoice_flow", invoice_flow_node)
    workflow.add_node("start_unknown", start_unknown_node)
    workflow.add_edge(START, "intent_recognizer")
    workflow.add_conditional_edges("intent_recognizer", route_after_intent, {
        "order_flow": "order_flow",
        "refund_flow": "refund_flow",
        "invoice_flow": "invoice_flow",
        "start_unknown": "start_unknown",
    })
    workflow.add_edge("order_flow", END)
    workflow.add_edge("refund_flow", END)
    workflow.add_edge("invoice_flow", END)
    workflow.add_edge("start_unknown", END)
    return workflow.compile()


# 全局图实例（懒加载）
_intent_graph = None


def get_intent_graph():
    """获取或构建 LangGraph 意图图（需已安装 langgraph）。"""
    global _intent_graph
    if _intent_graph is None:
        _intent_graph = _build_intent_graph()
    return _intent_graph


class LangGraphIntentProcessor:
    """
    基于 LangGraph 多节点的意图处理器。
    与 FSMProcessor 同效：process(text, context) 返回 IntentResult，context 中写入 current_state / next_prompt / is_completed。
    内部通过 StateGraph 多节点（意图识别 → 条件路由 → 各业务流节点）实现。
    """

    def __init__(self):
        self._graph = get_intent_graph()
        self._state: Optional[IntentGraphState] = None

    def process(self, text: str, context: Dict = None) -> Optional[IntentResult]:
        if self._graph is None:
            if context is not None:
                context["next_prompt"] = "未安装 langgraph，请 pip install langgraph 后使用 LangGraph 模式。"
            return None
        context = context if context is not None else {}
        if "slots" not in context:
            context["slots"] = {}
        prev = self._state or {}
        if prev.get("current_node") in ("order_done", "refund_done", "invoice_done"):
            prev = {}
        inp: IntentGraphState = {
            "text": text,
            "current_node": prev.get("current_node", "start"),
            "slots": dict(prev.get("slots") or {}),
        }
        out = self._graph.invoke(inp)
        self._state = out
        context["current_state"] = out.get("current_node", "start")
        context["next_prompt"] = out.get("next_prompt") or ""
        context["is_completed"] = out.get("is_completed", False)
        context["slots"].update(out.get("slots") or {})
        intent = out.get("intent", "unknown")
        return IntentResult(
            intent=intent,
            confidence=float(out.get("confidence", 0)),
            matched_rules=list(out.get("matched_rules") or []),
            extracted_entities=out.get("extracted_entities"),
        )

    def reset(self) -> None:
        self._state = None


def main():
    """
    主函数 - 系统演示和测试
    ======================
    
    功能说明:
    - 演示 LangChain 风格意图识别系统的完整功能
    - 提供多种测试用例验证系统性能
    - 展示单个识别和批量处理两种使用模式
    - 输出详细的识别结果和性能指标
    
    测试覆盖:
    1. 正则匹配测试: 结构化输入的精确识别
    2. 关键词匹配测试: 自然语言的模糊匹配
    3. 槽位提取测试: 参数信息的自动提取
    4. 未知意图测试: 兜底机制的有效性
    5. 批量处理测试: 系统的处理效率
    
    输出信息:
    - 识别的意图类型和置信度
    - 提取的槽位参数
    - 匹配的规则和推理过程
    - 系统性能和准确率统计
    """
    print("=== LangChain 风格的基于规则意图识别系统 (完整注释版) ===\n")
    
    # 创建意图识别链实例
    intent_chain = RuleBasedIntentChain()
    
    # 设计多样化的测试用例
    test_cases = [
        "我要查订单号123456的物流状态",    # 测试正则匹配 + 槽位提取
        "退款退款，我不要这个商品了",      # 测试正则匹配
        "帮我开个发票吧",                 # 测试正则匹配
        "昨天下的订单888888想要退货",     # 测试关键词匹配 + 复杂槽位提取
        "查一下我的快递到了吗",           # 测试关键词匹配
        "不知道说什么",                   # 测试未知意图兜底机制
        "我想开个1000元的发票"            # 测试槽位提取(金额)
    ]
    
    print("LangChain 风格意图识别测试:")
    print("=" * 80)
    
    # 逐个测试用例进行详细分析
    for i, text in enumerate(test_cases, 1):
        # 执行意图识别
        result = intent_chain.invoke({"text": text})
        
        # 输出详细的识别结果
        print(f"测试 {i}: {text}")
        print(f"  意图: {result['intent']}")                    # 识别的意图类型
        print(f"  置信度: {result['confidence']:.2f}")          # 置信度分数
        print(f"  槽位: {result['slots']}")                     # 提取的槽位参数
        print(f"  匹配规则: {result['matched_rules']}")         # 匹配的规则标识
        print(f"  推理过程: {result['reasoning']}")             # 推理过程说明
        
        # 如果有提取的实体，额外显示
        if result['extracted_entities']:
            print(f"  提取实体: {result['extracted_entities']}")
        print("-" * 80)
    
    # 演示批量处理能力
    print("\n批量处理演示:")
    print("=" * 80)
    
    # 批量处理的测试数据
    batch_texts = [
        "查订单123",      # 简短的订单查询
        "退货申请",        # 简短的退货申请
        "开发票"          # 简短的开票申请
    ]
    
    # 批量执行意图识别
    batch_results = [intent_chain.invoke({"text": text}) for text in batch_texts]
    
    # 输出批量处理结果的摘要
    for text, result in zip(batch_texts, batch_results):
        print(f"{text} -> {result['intent']} (置信度: {result['confidence']:.2f})")
    
    # 状态机：一步步推进直至完成用户意图
    print("\n状态机 (FSM) 按步推进直至完成:")
    print("=" * 80)
    fsm = FSMProcessor()
    # 演示 1：退款流程完整走完 (退款 -> 说明原因 -> 确认 -> 完成)
    refund_flow = [
        "我要退款",
        "因为商品有质量问题",
        "确认提交",
    ]
    ctx = {}
    for i, text in enumerate(refund_flow, 1):
        result = fsm.process(text, context=ctx)
        print(f"  [{i}] 用户: {text}")
        print(f"      状态: {ctx.get('current_state', '')} | 意图: {result.intent if result else 'unknown'}")
        prompt = ctx.get("next_prompt") or ""
        if prompt:
            print(f"      系统: {prompt}")
        if ctx.get("is_completed"):
            print("      >>> 本意图已完成，下一轮将重置为 start")
        print("-" * 80)
    # 演示 2：订单查询 (查订单 -> 说查物流 -> 完成)
    print("\n订单查询流程:")
    print("-" * 80)
    ctx = {}
    for text in ["查一下订单123456", "查一下物流"]:
        result = fsm.process(text, context=ctx)
        print(f"  用户: {text}")
        print(f"      状态: {ctx.get('current_state', '')}")
        if ctx.get("next_prompt"):
            print(f"      系统: {ctx['next_prompt']}")
        if ctx.get("is_completed"):
            print("      >>> 订单查询已完成")
        print("-" * 80)
    # 演示 3：开票流程 (开票 -> 填金额 -> 确认 -> 完成)
    print("\n开票流程:")
    print("-" * 80)
    ctx = {}
    for text in ["帮我开发票", "开1000元的", "开吧"]:
        result = fsm.process(text, context=ctx)
        print(f"  用户: {text}")
        print(f"      状态: {ctx.get('current_state', '')}")
        if ctx.get("next_prompt"):
            print(f"      系统: {ctx['next_prompt']}")
        if ctx.get("is_completed"):
            print("      >>> 开票流程已完成")
        print("-" * 80)

    # LangGraph 多节点模式演示（需安装 langgraph）
    if _LANGGRAPH_AVAILABLE:
        print("\nLangGraph 多节点意图流程演示:")
        print("=" * 80)
        lg = LangGraphIntentProcessor()
        ctx = {}
        for i, text in enumerate(["我要退款", "因为不合适", "确认提交"], 1):
            result = lg.process(text, context=ctx)
            print(f"  [{i}] 用户: {text}")
            print(f"      状态: {ctx.get('current_state', '')} | 意图: {result.intent if result else 'unknown'}")
            if ctx.get("next_prompt"):
                print(f"      系统: {ctx['next_prompt']}")
            if ctx.get("is_completed"):
                print("      >>> 本意图已完成")
            print("-" * 80)
    else:
        print("\n(LangGraph 未安装，跳过多节点图演示；pip install langgraph 可启用)")

    print("\n" + "=" * 80)
    print("系统特性总结:")
    print("1. 多策略融合: 正则匹配 + 关键词匹配")
    print("2. 智能决策: 基于置信度和规则优先级")
    print("3. 槽位提取: 自动提取业务参数")
    print("4. 可解释性: 提供详细的推理过程")
    print("5. 兜底机制: 未知输入的优雅处理")
    print("6. 模块化设计: LangChain 风格的组件架构")
    print("7. 状态机: 按步推进直至完成用户意图，支持完成态重置")
    if _LANGGRAPH_AVAILABLE:
        print("8. LangGraph: 多节点图（意图识别→路由→order/refund/invoice 流节点）")

if __name__ == "__main__":
    main()