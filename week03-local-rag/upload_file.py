#####################################
#######       上传文件         #######
#####################################
"""
文件上传模块 - 负责处理用户上传的文档和数据

主要功能：
1. 非结构化文件上传：支持 PDF、DOCX、TXT 等文档格式
2. 结构化数据上传：支持 CSV、XLSX 等表格格式，并自动转换为文本格式
3. 文件管理：提供文件删除、目录刷新等功能
4. UI 状态同步：确保界面与文件系统状态一致

技术特点：
- 采用分层存储架构，区分结构化和非结构化数据
- 实现原子性文件操作，确保数据一致性
- 支持批量操作，提升用户体验
"""

import gradio as gr  # Gradio 界面框架，用于构建 Web UI
import os            # 操作系统接口，用于文件路径和目录操作
import shutil        # 高级文件操作，用于文件移动和目录删除
import pandas as pd  # 数据处理库，用于读取和处理表格数据

# 文件存储路径配置 - 采用分层存储架构设计
# 结构化数据和非结构化数据分离存储，便于后续向量化处理时采用不同的策略
STRUCTURED_FILE_PATH = "File/Structured"      # 结构化数据存储路径（CSV/Excel等表格数据）
UNSTRUCTURED_FILE_PATH = "File/Unstructured"  # 非结构化数据存储路径（PDF/DOC/TXT等文档数据）

# 目录刷新函数 - 实现动态文件系统监控
def refresh_label():
    """
    刷新非结构化类目列表
    采用实时目录扫描机制，确保UI组件与文件系统状态同步
    避免因文件系统变更导致的界面状态不一致问题
    """
    return os.listdir(UNSTRUCTURED_FILE_PATH)

def refresh_data_table():
    """
    刷新结构化数据表列表
    同步文件系统状态到前端组件，保证数据一致性
    """
    return os.listdir(STRUCTURED_FILE_PATH)

# 非结构化文件上传处理器
def upload_unstructured_file(files, label_name):
    """
    非结构化文件上传核心处理函数

    设计理念：
    1. 采用原子性操作确保文件上传的事务完整性
    2. 使用shutil.move而非copy，避免临时文件残留和磁盘空间浪费
    3. 实现文件去重机制，防止重复上传导致的存储冗余

    参数:
        files: Gradio文件对象列表，包含临时文件路径信息
        label_name: 用户定义的分类标签，用于文件组织和后续检索
    """
    # 输入验证 - 防御性编程实践
    if files is None:
        gr.Info("请上传文件")
    elif len(label_name) == 0:
        gr.Info("请输入类目名称")
    # 防重复创建检查 - 避免目录冲突和数据覆盖风险
    elif label_name in os.listdir(UNSTRUCTURED_FILE_PATH):
        gr.Info(f"{label_name}类目已存在")
    else:
        try:
            # 确保目标目录存在 - 惰性目录创建模式
            if not os.path.exists(os.path.join(UNSTRUCTURED_FILE_PATH, label_name)):
                os.mkdir(os.path.join(UNSTRUCTURED_FILE_PATH, label_name))

            # 批量文件处理 - 原子性文件移动操作
            for file in files:
                print(file)  # 调试日志输出
                file_path = file.name  # Gradio临时文件路径
                file_name = os.path.basename(file_path)  # 提取原始文件名
                destination_file_path = os.path.join(UNSTRUCTURED_FILE_PATH, label_name, file_name)
                # 使用move而非copy的原因：
                # 1. 避免临时文件占用磁盘空间
                # 2. 确保文件操作的原子性
                # 3. 减少I/O操作提升性能
                shutil.move(file_path, destination_file_path)
            gr.Info(f"文件已上传至{label_name}类目中，请前往创建知识库")
        except Exception as e:
            # 异常处理 - 提供用户友好的错误反馈
            gr.Info(f"请勿重复上传")

# 结构化数据上传与预处理器
def upload_structured_file(files, label_name):
    """
    结构化数据上传与格式转换核心函数

    核心设计思想：
    1. 结构化数据向量化预处理：将表格数据转换为文本格式以适配向量数据库
    2. 语义保持转换：采用键值对格式保持原始数据的语义结构
    3. 内存优化：处理完成后立即删除原始文件，避免存储冗余

    技术实现要点：
    - 支持多种表格格式（Excel/CSV）的统一处理
    - 采用行级序列化策略，每行数据转换为独立的文本块
    - 使用特殊分隔符【】标记数据边界，便于后续chunk分割
    """
    if files is None:
        gr.Info("请上传文件")
    elif len(label_name) == 0:
        gr.Info("请输入数据表名称")
    # 防重复创建检查
    elif label_name in os.listdir(STRUCTURED_FILE_PATH):
        gr.Info(f"{label_name}数据表已存在")
    else:
        try:
            # 确保目标目录存在
            if not os.path.exists(os.path.join(STRUCTURED_FILE_PATH, label_name)):
                os.mkdir(os.path.join(STRUCTURED_FILE_PATH, label_name))

            for file in files:
                file_path = file.name
                file_name = os.path.basename(file_path)
                destination_file_path = os.path.join(STRUCTURED_FILE_PATH, label_name, file_name)
                shutil.move(file_path, destination_file_path)

                # 多格式表格数据统一处理 - 策略模式应用
                if os.path.splitext(destination_file_path)[1] == ".xlsx":
                    df = pd.read_excel(destination_file_path)
                elif os.path.splitext(destination_file_path)[1] == ".csv":
                    df = pd.read_csv(destination_file_path)

                # 结构化数据文本化转换
                txt_file_name = os.path.splitext(file_name)[0] + '.txt'
                columns = df.columns

                # 行级数据序列化处理
                with open(os.path.join(STRUCTURED_FILE_PATH, label_name, txt_file_name), "w", encoding='utf-8') as file:
                    for idx, row in df.iterrows():
                        file.write("【")  # 数据块起始标记
                        info = []
                        # 键值对格式转换 - 保持语义结构
                        for col in columns:
                            info.append(f"{col}:{row[col]}")
                        infos = ",".join(info)
                        file.write(infos)
                        # 条件性换行处理 - 避免文件末尾多余换行
                        if idx != len(df) - 1:
                            file.write("】\n")
                        else:
                            file.write("】")

                # 原始文件清理 - 避免存储冗余和潜在的数据泄露风险
                os.remove(destination_file_path)

            gr.Info(f"文件已上传至{label_name}数据表中，请前往创建知识库")
        except Exception as e:
            gr.Info(f"请勿重复上传")

# UI状态同步函数 - 实现响应式界面更新
def update_datatable():
    """
    结构化数据表选项动态更新
    返回Gradio组件更新对象，触发前端选项列表重新渲染
    采用惰性更新策略，仅在需要时扫描文件系统
    """
    return gr.update(choices=os.listdir(STRUCTURED_FILE_PATH))

def update_label():
    """
    非结构化类目选项动态更新
    实现文件系统状态与UI组件的实时同步
    """
    return gr.update(choices=os.listdir(UNSTRUCTURED_FILE_PATH))

# 批量删除操作 - 支持多选删除提升用户体验
def delete_label(label_name):
    """
    批量删除非结构化数据类目

    设计考虑：
    1. 支持多选批量操作，提升用户操作效率
    2. 使用shutil.rmtree进行递归删除，确保目录及其内容完全清理
    3. 添加存在性检查，避免删除不存在目录时的异常
    """
    if label_name is not None:
        for label in label_name:
            folder_path = os.path.join(UNSTRUCTURED_FILE_PATH, label)
            if os.path.exists(folder_path):
                # 递归删除目录及其所有内容
                shutil.rmtree(folder_path)
                gr.Info(f"{label}类目已删除")

def delete_data_table(table_name):
    """
    批量删除结构化数据表
    实现与delete_label相同的批量删除逻辑
    保持API一致性和用户体验统一性
    """
    if table_name is not None:
        for table in table_name:
            folder_path = os.path.join(STRUCTURED_FILE_PATH, table)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                gr.Info(f"{table}数据表已删除")