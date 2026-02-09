"""
OCR 图像文本加载器实现
使用 PaddleOCR 从图像中提取文本，并转换为 LlamaIndex Document 对象
"""

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import List, Union, Optional
import os
from pathlib import Path
import numpy as np
from paddleocr import PaddleOCR

class ImageOCRReader(BaseReader):
    """使用 PP-OCR 从图像中提取文本并返回 Document"""

    def __init__(
        self,
        lang='ch',
        use_gpu=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        ocr_version="PP-OCRv4",
        **kwargs
    ):
        """
        Args:
            lang (str): OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu (bool): 是否使用 GPU 加速（内部转换为 device 参数）
            use_doc_orientation_classify (bool): 是否使用文档方向分类模型
            use_doc_unwarping (bool): 是否使用文本图像矫正模型
            use_textline_orientation (bool): 是否使用文本行方向分类模型（默认True，替代已弃用的use_angle_cls）
            ocr_version (str): OCR 版本 ('PP-OCRv4', 'PP-OCRv5', etc.)
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        super().__init__()
        self.lang = lang
        self.ocr_version = ocr_version

        ocr_kwargs = {
            'lang': lang,
            'device': 'gpu' if use_gpu else 'cpu',  # 使用 device 参数替代 use_gpu
            'use_doc_orientation_classify': use_doc_orientation_classify,
            'use_doc_unwarping': use_doc_unwarping,
            'use_textline_orientation': True,
            **kwargs
        }

        # 如果指定了 OCR 版本，添加版本参数
        if ocr_version:
            ocr_kwargs['ocr_version'] = ocr_version

        # 为了性能，在初始化时加载模型
        print(f"正在初始化 PaddleOCR (语言: {lang}, 版本: {ocr_version}, 设备: {'GPU' if use_gpu else 'CPU'})...")
        self._ocr = PaddleOCR(**ocr_kwargs)
        print("PaddleOCR 初始化完成")

    def load_data(self, file: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表

        Args:
            file: 图像路径字符串、Path 对象或路径列表

        Returns:
            List[Document]: Document 对象列表，每个 Document 包含从图像提取的文本和元数据
        """
        # 统一处理为列表
        if isinstance(file, (str, Path)):
            file_paths = [Path(file)]
        else:
            file_paths = [Path(f) for f in file]

        documents = []

        for file_path in file_paths:
            # 检查文件是否存在
            if not file_path.exists():
                print(f"警告: 文件不存在，跳过: {file_path}")
                continue

            # 转换为绝对路径
            file_path = file_path.resolve()

            print(f"正在处理图像: {file_path.name}")

            # 使用 PaddleOCR 进行 OCR
            try:
                # PaddleOCR 返回的结果格式: [[[坐标], (文本, 置信度)], ...]
                result = self._ocr.ocr(str(file_path), cls=True)

                # 处理 OCR 结果
                if result is None or len(result) == 0 or result[0] is None:
                    print(f"警告: 未从图像中检测到文本: {file_path.name}")
                    # 即使没有文本，也创建一个空的 Document
                    doc = Document(
                        text="",
                        metadata={
                            'image_path': str(file_path),
                            'ocr_model': self.ocr_version,
                            'language': self.lang,
                            'num_text_blocks': 0,
                            'avg_confidence': 0.0
                        }
                    )
                    documents.append(doc)
                    continue

                # 提取文本和置信度
                text_blocks = []
                confidences = []

                for line in result[0]:
                    if line is None:
                        continue
                    # line 格式: [[坐标], (文本, 置信度)]
                    if len(line) >= 2:
                        text_info = line[1]
                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            text_blocks.append(text)
                            confidences.append(confidence)

                # 计算平均置信度
                avg_confidence = np.mean(confidences) if confidences else 0.0

                # 拼接文本（格式：Text Block 1 (conf: 0.98): 文本内容）
                text_parts = []
                for i, (text, conf) in enumerate(zip(text_blocks, confidences), 1):
                    text_parts.append(f"[Text Block {i}] (conf: {conf:.2f}): {text}")

                full_text = "\n".join(text_parts)

                # 创建 Document 对象
                doc = Document(
                    text=full_text,
                    metadata={
                        'image_path': str(file_path),
                        'ocr_model': self.ocr_version,
                        'language': self.lang,
                        'num_text_blocks': len(text_blocks),
                        'avg_confidence': round(avg_confidence, 4)
                    }
                )
                documents.append(doc)

                print(f"  成功提取 {len(text_blocks)} 个文本块，平均置信度: {avg_confidence:.4f}")

            except Exception as e:
                print(f"错误: 处理图像时出错 {file_path.name}: {e}")
                # 即使出错，也创建一个包含错误信息的 Document
                doc = Document(
                    text=f"OCR 处理失败: {str(e)}",
                    metadata={
                        'image_path': str(file_path),
                        'ocr_model': self.ocr_version,
                        'language': self.lang,
                        'num_text_blocks': 0,
                        'avg_confidence': 0.0,
                        'error': str(e)
                    }
                )
                documents.append(doc)

        return documents

    def load_data_from_dir(self, dir_path: Union[str, Path]) -> List[Document]:
        """
        从目录中加载所有图像文件

        Args:
            dir_path: 目录路径

        Returns:
            List[Document]: Document 对象列表
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError(f"路径不是目录: {dir_path}")

        # 支持的图像格式
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

        # 查找所有图像文件
        image_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"警告: 目录中没有找到图像文件: {dir_path}")
            return []

        print(f"在目录中找到 {len(image_files)} 个图像文件")
        return self.load_data(image_files)


def setup_llamaindex_environment():
    """配置 LlamaIndex 所需的环境和模型"""
    import os
    from llama_index.core import Settings
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True
    )

    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,  # DashScope API 限制
        embed_input_length=8192
    )

    print("LlamaIndex 环境配置完成")


def main():
    """
    作业的入口函数
    在根目录可以通过 python -m ocr_research.main 运行
    """
    # 1. 设置 LlamaIndex 环境
    print("=" * 80)
    print("OCR 图像文本加载器测试")
    print("=" * 80)

    try:
        setup_llamaindex_environment()
    except Exception as e:
        print(f"环境配置失败: {e}")
        print("请确保已设置 DASHSCOPE_API_KEY 环境变量")
        return

    # 2. 获取测试图像路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "ocr_images"

    if not data_dir.exists():
        print(f"错误: 测试图像目录不存在: {data_dir}")
        print("请确保 data/ocr_images/ 目录存在并包含测试图像")
        return

    # 3. 创建 ImageOCRReader
    print("\n" + "-" * 80)
    print("初始化 ImageOCRReader")
    print("-" * 80)

    reader = ImageOCRReader(
        lang='ch',  # 中文
        use_gpu=False,  # 根据实际情况设置
        use_textline_orientation=True,  # 根据 paddle 是否安装自动调整
        ocr_version="PP-OCRv4"
    )

    # 4. 加载图像并提取文本
    print("\n" + "-" * 80)
    print("加载图像并提取文本")
    print("-" * 80)

    # 方式1: 加载单个图像
    # image_files = [data_dir / "document.png"]

    # 方式2: 加载多个图像
    image_files = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))

    # 方式3: 从目录加载所有图像
    # documents = reader.load_data_from_dir(data_dir)

    if not image_files:
        print(f"错误: 在 {data_dir} 中没有找到图像文件")
        return

    documents = reader.load_data(image_files)

    # 5. 显示提取结果
    print("\n" + "-" * 80)
    print("OCR 提取结果")
    print("-" * 80)
    for i, doc in enumerate(documents, 1):
        print(f"\n文档 {i}:")
        print(f"  图像路径: {doc.metadata.get('image_path', 'N/A')}")
        print(f"  文本块数量: {doc.metadata.get('num_text_blocks', 0)}")
        print(f"  平均置信度: {doc.metadata.get('avg_confidence', 0.0):.4f}")
        print(f"  提取的文本预览 (前200字符):")
        text_preview = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
        print(f"    {text_preview}")

    # 6. 构建索引并进行查询测试
    print("\n" + "-" * 80)
    print("构建向量索引并进行查询测试")
    print("-" * 80)

    from llama_index.core import VectorStoreIndex

    print("正在构建向量索引...")
    index = VectorStoreIndex.from_documents(documents)
    print("向量索引构建完成")

    # 创建查询引擎
    query_engine = index.as_query_engine(similarity_top_k=3)

    # 测试查询
    test_queries = [
        "图片中提到了什么日期？",
        "图片中有哪些文字内容？",
        "图片中的文本内容是什么？"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        try:
            response = query_engine.query(query)
            print(f"回答: {response}")
        except Exception as e:
            print(f"查询失败: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
