from langchain_core.output_parsers import BaseOutputParser
from typing import Dict, Any
import xml.etree.ElementTree as ET

class XMLOutputParser(BaseOutputParser[Dict[str, Any]]):
    """将 XML 字符串解析为字典的输出解析器"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """解析 XML 字符串为字典
        
        Args:
            text: 包含 XML 的字符串
            
        Returns:
            解析后的字典
        """
        try:
            # 解析 XML
            root = ET.fromstring(text)
            
            # 转换为字典
            result = self._element_to_dict(root)
            return result
        except Exception as e:
            # 尝试提取 XML 部分
            import re
            xml_match = re.search(r'<[^>]+>.*?</[^>]+>', text, re.DOTALL)
            if xml_match:
                try:
                    root = ET.fromstring(xml_match.group(0))
                    return self._element_to_dict(root)
                except:
                    pass
            # 如果解析失败，返回错误信息
            return {"error": f"XML 解析失败: {str(e)}"}
    
    def _element_to_dict(self, element) -> Dict[str, Any]:
        """将 XML 元素转换为字典
        
        Args:
            element: XML 元素
            
        Returns:
            转换后的字典
        """
        result = {}
        
        # 处理属性
        if element.attrib:
            result.update(element.attrib)
        
        # 处理子元素
        children = list(element)
        if children:
            for child in children:
                child_data = self._element_to_dict(child)
                if child.tag in result:
                    # 如果标签已存在，转换为列表
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = child_data
        else:
            # 如果没有子元素，使用文本内容
            if element.text and element.text.strip():
                result = element.text.strip()
        
        return result
    
    def get_format_instructions(self) -> str:
        """获取格式化指令
        
        Returns:
            格式化指令字符串
        """
        return """请以 XML 格式输出，例如：
<result>
    <name>示例</name>
    <value>123</value>
    <items>
        <item>项目1</item>
        <item>项目2</item>
    </items>
</result>
"""

# 测试示例
if __name__ == "__main__":
    from langchain_community.llms import Tongyi
    from langchain_core.prompts import PromptTemplate
    
    # 初始化组件
    parser = XMLOutputParser()
    llm = Tongyi(temperature=0)
    
    # 创建提示模板
    prompt = PromptTemplate(
        template="请提供关于{topic}的信息，包括定义、特点和应用。\n{format_instructions}",
        input_variables=["topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 创建链
    chain = prompt | llm | parser
    
    # 测试
    result = chain.invoke({"topic": "人工智能"})
    print("解析结果:")
    print(result)
