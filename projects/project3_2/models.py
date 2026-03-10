from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

@dataclass
class Entity:
    """实体类

    表示知识图谱中的实体，用于存储实体的基本信息
    """
    name: str  # 实体名称，实体的唯一标识符
    type: str  # 实体类型，如人物、组织、地点等
    confidence: float = 1.0  # 实体识别置信度，范围0-1，默认为1.0

@dataclass
class Relationship:
    """关系类

    表示知识图谱中实体之间的关系，用于存储实体间的关联信息
    """
    source: str  # 源实体，关系的起点
    target: str  # 目标实体，关系的终点
    type: str  # 关系类型，如所属、任职、位置等
    confidence: float = 1.0  # 关系识别置信度，范围0-1，默认为1.0

@dataclass
class Document:
    """文档类

    表示检索系统中的文档，用于存储文档的基本信息和嵌入向量
    """
    id: str  # 文档ID，文档的唯一标识符
    content: str  # 文档内容，文档的具体文本
    metadata: Dict[str, Any]  # 文档元数据，如来源、创建时间等
    embedding: Optional[np.ndarray] = None  # 文档嵌入向量，用于向量检索，默认为None

@dataclass
class RetrievalResult:
    """检索结果类

    表示检索系统返回的结果，包含文档、分数和来源信息
    """
    document: Document  # 检索到的文档对象
    score: float  # 检索分数，用于排序和筛选
    source: str  # 检索来源: 'vector' (向量检索), 'keyword' (关键词检索), 或 'graph' (图谱检索)

@dataclass
class GraphResult:
    """图谱结果类

    表示知识图谱推理的结果，包含实体、关系、置信度和推理路径
    """
    entities: List[str]  # 识别出的实体列表，从输入文本中提取的实体
    relationships: List[Dict]  # 识别出的关系列表，包含源实体、目标实体和关系类型
    confidence: float  # 结果置信度，整个图谱推理的置信度
    reasoning_path: List[str]  # 推理路径，记录图谱推理的步骤