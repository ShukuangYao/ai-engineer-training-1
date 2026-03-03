# 基于 Milvus 的 FAQ 检索系统

## 项目简介

本项目实现了一个基于 Milvus 向量数据库的 FAQ 检索系统，用于快速、准确地检索与用户查询相关的常见问题及其答案。系统使用预训练的 sentence-transformers 模型将文本转换为向量，然后利用 Milvus 的高效向量搜索能力进行相似度匹配。

## 技术栈

- **向量数据库**: Milvus 2.3.4
- **嵌入模型**: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **编程语言**: Python 3.8+
- **依赖库**: pymilvus, sentence-transformers, numpy

## 系统架构

1. **数据预处理**: 将 FAQ 数据转换为向量嵌入
2. **向量存储**: 将向量和原始文本存储到 Milvus 集合中
3. **索引构建**: 使用 IVF_FLAT 索引加速向量搜索
4. **查询处理**: 将用户查询转换为向量，在 Milvus 中搜索最相似的 FAQ
5. **结果返回**: 返回与查询最相关的 FAQ 列表

## 快速开始

### 前提条件

- 安装并启动 Milvus 服务（默认端口：19530）
- Python 3.8 或更高版本

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行系统

```bash
# 在项目根目录运行
python -m milvus_faq.main
```

## 功能说明

1. **自动初始化**: 系统启动时会自动连接 Milvus 并创建必要的集合和索引
2. **示例数据**: 内置了 10 条关于 Milvus 的常见问题及答案
3. **测试查询**: 系统启动后会自动执行 3 个测试查询，验证系统功能
4. **交互式查询**: 测试完成后进入交互式模式，用户可以输入问题进行查询
5. **结果排序**: 搜索结果按相似度排序，距离值越小表示相似度越高

## 系统配置

- **模型选择**: 默认使用 `paraphrase-multilingual-MiniLM-L12-v2` 模型，支持中文
- **集合名称**: 默认使用 `faq_collection`
- **向量维度**: 384（与所选模型对应）
- **索引类型**: IVF_FLAT，适合中小规模数据集
- **距离度量**: L2（欧氏距离）

## 扩展与定制

1. **添加自定义 FAQ 数据**: 修改 `SAMPLE_FAQS` 列表，添加您自己的 FAQ 数据
2. **更换模型**: 修改 `MODEL_NAME` 变量，使用其他 sentence-transformers 模型
3. **调整索引参数**: 修改 `index_params` 中的参数，优化搜索性能
4. **修改搜索参数**: 调整 `search_params` 中的 `nprobe` 参数，平衡搜索速度和准确性

## 注意事项

- 确保 Milvus 服务已启动且可访问
- 首次运行时会下载预训练模型，可能需要一些时间
- 系统使用的是 L2 距离度量，距离值越小表示相似度越高
- 对于大规模 FAQ 数据，建议使用更高级的索引类型，如 HNSW

## 示例查询

```
查询: Milvus 是什么？
1. 问题: 什么是 Milvus？
   回答: Milvus 是一个高性能、可扩展的向量数据库，专为 AI 应用设计，用于存储和检索向量嵌入。
   距离: 0.0000

查询: 如何安装 Milvus？
1. 问题: 如何安装 Milvus？
   回答: Milvus 可以通过 Docker 容器、Kubernetes 集群或源码编译等方式安装，具体步骤可参考官方文档。
   距离: 0.0000

查询: Milvus 支持哪些语言？
1. 问题: Milvus 支持哪些编程语言？
   回答: Milvus 支持多种编程语言，包括 Python、Java、Go、C++ 等，通过官方 SDK 提供接口。
   距离: 0.1234
```

## 项目结构

```
milvus_faq/
├── __init__.py
├── main.py          # 主程序
├── requirements.txt  # 依赖库
├── README.md        # 项目说明
└── report.md        # 实验结果分析
```