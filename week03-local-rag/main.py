"""
主应用入口 - FastAPI + Gradio 混合架构的 RAG 系统

系统架构：
- FastAPI：提供主页面路由和 API 服务
- Gradio：提供交互式 Web 界面（聊天、文件上传、知识库管理）
- 多页面设计：主页、上传数据、创建知识库、RAG 问答

页面路由：
- / : 主页，提供系统导航和说明
- /chat : RAG 问答界面
- /upload_data : 文件上传界面
- /create_knowledge_base : 知识库创建界面

技术栈：
- FastAPI: 现代 Python Web 框架，提供 RESTful API
- Gradio: 快速构建 ML 应用界面的框架
- LlamaIndex: RAG 框架，处理向量索引和检索
- DashScope: 阿里云大模型和向量嵌入服务
"""

# 导入依赖 - 构建多模态RAG系统的核心组件
from fastapi import FastAPI              # FastAPI Web 框架
from fastapi.responses import HTMLResponse  # HTML 响应类型
import gradio as gr                      # Gradio 界面框架
import os                                # 操作系统接口
from html_string import main_html, plain_html  # HTML 模板字符串
from upload_file import *                # 文件上传模块的所有函数
from create_kb import *                  # 知识库创建模块的所有函数
from chat import get_model_response      # RAG 对话核心函数

def user(user_message, history):
    """
    用户消息预处理函数

    功能：
    - 提取用户输入的文本和文件
    - 更新对话历史记录
    - 清空输入框，准备接收下一条消息

    参数:
        user_message: Gradio MultimodalTextbox 对象，包含文本和文件
        history: 对话历史列表，格式为 [[用户消息, 助手回复], ...]

    返回:
        tuple: (清空后的输入框, 更新后的对话历史)
    """

    print(user_message)  # 调试日志
    return {'text': '', 'files': user_message['files']}, history + [[user_message['text'], None]]

#####################################
######       Gradio界面架构       #######
#####################################

def get_chat_block():
    """
    RAG对话界面构建器

    构建功能完整的 RAG 问答界面，包括：
    1. 多模态输入：支持文本和文件混合输入
    2. 知识库选择：动态加载已创建的知识库
    3. 参数配置：模型参数、RAG 参数、生成参数
    4. 召回文本显示：展示检索到的相关文档片段
    5. 流式响应：实时显示模型生成的内容

    返回:
        gr.Blocks: Gradio 界面块对象
    """
    with gr.Blocks(theme=gr.themes.Base(), css=".gradio_container { background-color: #f0f0f0; }") as chat:
        gr.HTML(plain_html)

        with gr.Row():
            # 主对话区域 - 占据主要视觉空间
            with gr.Column(scale=10):
                # 聊天机器人组件 - 支持自定义头像增强用户体验
                chatbot = gr.Chatbot(
                    label="Chatbot",
                    height=750,
                    avatar_images=("images/user.jpeg", "images/tongyi.png")
                )
                with gr.Row():
                    # 多模态输入框 - 支持文本+文件混合输入
                    input_message = gr.MultimodalTextbox(
                        label="请输入",
                        file_types=[".xlsx", ".csv", ".docx", ".pdf", ".txt"],  # 支持的文件格式
                        scale=7
                    )
                    clear_btn = gr.ClearButton(chatbot, input_message, scale=1)

            # 参数控制面板 - 提供精细化控制能力
            with gr.Column(scale=5):
                # 知识库选择器 - 动态加载可用知识库
                knowledge_base = gr.Dropdown(
                    choices=os.listdir(DB_PATH),
                    label="加载知识库",
                    interactive=True,
                    scale=2
                )

                # 召回文本段显示 - 增强系统可解释性
                with gr.Accordion(label="召回文本段", open=False):
                    chunk_text = gr.Textbox(
                        label="召回文本段",
                        interactive=False,
                        scale=5,
                        lines=10
                    )

                # 大语言模型参数配置
                with gr.Accordion(label="模型设置", open=True):
                    model = gr.Dropdown(
                        choices=['qwen-max', 'qwen-plus', 'qwen-turbo'],
                        label="选择模型",
                        interactive=True,
                        value="qwen-max",  # 默认使用最强模型
                        scale=2
                    )
                    # 温度参数 - 控制生成文本的随机性和创造性
                    temperature = gr.Slider(
                        maximum=2, minimum=0,
                        interactive=True,
                        label="温度参数",
                        step=0.01,
                        value=0.85,  # 平衡创造性和准确性
                        scale=2
                    )
                    # 最大token数 - 控制响应长度避免过长输出
                    max_tokens = gr.Slider(
                        maximum=2000, minimum=0,
                        interactive=True,
                        label="最大回复长度",
                        step=50,
                        value=1024,
                        scale=2
                    )
                    # 上下文轮数 - 控制对话记忆长度
                    history_round = gr.Slider(
                        maximum=30, minimum=1,
                        interactive=True,
                        label="携带上下文轮数",
                        step=1,
                        value=3,  # 平衡上下文理解和token消耗
                        scale=2
                    )

                # RAG检索参数配置
                with gr.Accordion(label="RAG参数设置", open=True):
                    # 召回片段数 - 控制检索到的文档块数量
                    chunk_cnt = gr.Slider(
                        maximum=20, minimum=1,
                        interactive=True,
                        label="选择召回片段数",
                        step=1,
                        value=5,  # 平衡信息完整性和处理效率
                        scale=2
                    )
                    # 相似度阈值 - 过滤低相关性文档
                    similarity_threshold = gr.Slider(
                        maximum=1, minimum=0,
                        interactive=True,
                        label="相似度阈值",
                        step=0.01,
                        value=0.2,  # 较低阈值确保召回覆盖度
                        scale=2
                    )

        # 事件绑定 - 实现响应式交互
        # 链式调用：用户输入 -> 消息预处理 -> 模型响应生成
        input_message.submit(
            fn=user,
            inputs=[input_message, chatbot],
            outputs=[input_message, chatbot],
            queue=False  # 禁用队列确保实时响应
        ).then(
            fn=get_model_response,
            inputs=[input_message, chatbot, model, temperature, max_tokens,
                   history_round, knowledge_base, similarity_threshold, chunk_cnt],
            outputs=[chatbot, chunk_text]
        )

        # 页面加载时的初始化操作
        chat.load(update_knowledge_base, [], knowledge_base)  # 刷新知识库列表
        chat.load(clear_tmp)  # 清理临时文件

    return chat


def get_upload_block():
    """
    文件上传界面构建器

    提供两个标签页：
    1. 非结构化数据：上传 PDF、DOCX、TXT 等文档
    2. 结构化数据：上传 CSV、XLSX 等表格文件

    功能包括：
    - 新建类目/数据表：上传文件并创建分类
    - 管理类目/数据表：查看和删除已上传的数据

    返回:
        gr.Blocks: Gradio 界面块对象
    """
    with gr.Blocks(theme=gr.themes.Base()) as upload:
        gr.HTML(plain_html)
        with gr.Tab("非结构化数据"):
            with gr.Accordion(label="新建类目",open=True):
                with gr.Column(scale=2):
                    unstructured_file = gr.Files(file_types=[".pdf",".docx",".txt"])
                    with gr.Row():
                        new_label = gr.Textbox(label="类目名称",placeholder="请输入类目名称",scale=5)
                        create_label_btn = gr.Button("新建类目",variant="primary",scale=1)
            with gr.Accordion(label="管理类目",open=False):
                with gr.Row():
                    data_label =gr.Dropdown(choices=os.listdir(UNSTRUCTURED_FILE_PATH),label="管理类目",interactive=True,scale=8,multiselect=True)
                    delete_label_btn = gr.Button("删除类目",variant="stop",scale=1)
        with gr.Tab("结构化数据"):
            with gr.Accordion(label="新建数据表",open=True):
                with gr.Column(scale=2):
                    structured_file = gr.Files(file_types=[".xlsx",".csv"])
                    with gr.Row():
                        new_label_1 = gr.Textbox(label="数据表名称",placeholder="请输入数据表名称",scale=5)
                        create_label_btn_1 = gr.Button("新建数据表",variant="primary",scale=1)
            with gr.Accordion(label="管理数据表",open=False):
                with gr.Row():
                    data_label_1 =gr.Dropdown(choices=os.listdir(STRUCTURED_FILE_PATH),label="管理数据表",interactive=True,scale=8,multiselect=True)
                    delete_data_table_btn = gr.Button("删除数据表",variant="stop",scale=1)
        delete_label_btn.click(delete_label,inputs=[data_label]).then(fn=update_label,outputs=[data_label])
        create_label_btn.click(fn=upload_unstructured_file,inputs=[unstructured_file,new_label]).then(fn=update_label,outputs=[data_label])
        delete_data_table_btn.click(delete_data_table,inputs=[data_label_1]).then(fn=update_datatable,outputs=[data_label_1])
        create_label_btn_1.click(fn=upload_structured_file,inputs=[structured_file,new_label_1]).then(fn=update_datatable,outputs=[data_label_1])
        upload.load(update_label,[],data_label)
        upload.load(update_datatable,[],data_label_1)
    return upload

def get_knowledge_base_block():
    """
    知识库管理界面构建器

    提供知识库的创建和管理功能：
    1. 非结构化数据知识库：基于上传的文档创建向量索引
    2. 结构化数据知识库：基于转换后的表格数据创建向量索引
    3. 知识库管理：查看和删除已创建的知识库

    工作流程：
    用户选择数据源 -> 输入知识库名称 -> 点击创建 -> 系统生成向量索引

    返回:
        gr.Blocks: Gradio 界面块对象
    """
    with gr.Blocks(theme=gr.themes.Base()) as knowledge:
        gr.HTML(plain_html)
        # 非结构化数据知识库
        with gr.Tab("非结构化数据"):
            with gr.Row():
                data_label_2 =gr.Dropdown(choices=os.listdir(UNSTRUCTURED_FILE_PATH),label="选择类目",interactive=True,scale=2,multiselect=True)
                knowledge_base_name = gr.Textbox(label="知识库名称",placeholder="请输入知识库名称",scale=2)
                create_knowledge_base_btn = gr.Button("确认创建知识库",variant="primary",scale=1)
        # 结构化数据知识库
        with gr.Tab("结构化数据"):
            with gr.Row():
                data_label_3 =gr.Dropdown(choices=os.listdir(STRUCTURED_FILE_PATH),label="选择数据表",interactive=True,scale=2,multiselect=True)
                knowledge_base_name_1 = gr.Textbox(label="知识库名称",placeholder="请输入知识库名称",scale=2)
                create_knowledge_base_btn_1 = gr.Button("确认创建知识库",variant="primary",scale=1)
        with gr.Row():
            knowledge_base =gr.Dropdown(choices=os.listdir(DB_PATH),label="管理知识库",interactive=True,scale=4)
            delete_db_btn = gr.Button("删除知识库",variant="stop",scale=1)
        create_knowledge_base_btn.click(fn=create_unstructured_db,inputs=[knowledge_base_name,data_label_2]).then(update_knowledge_base,outputs=[knowledge_base])
        delete_db_btn.click(delete_db,inputs=[knowledge_base]).then(update_knowledge_base,outputs=[knowledge_base])
        create_knowledge_base_btn_1.click(fn=create_structured_db,inputs=[knowledge_base_name_1,data_label_3]).then(update_knowledge_base,outputs=[knowledge_base])
        knowledge.load(update_knowledge_base,[],knowledge_base)
        knowledge.load(update_label,[],data_label_2)
        knowledge.load(update_datatable,[],data_label_3)
    return knowledge

# FastAPI 应用实例化
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_main():
    """
    主页路由处理器

    返回系统主页 HTML，提供导航链接和功能说明

    返回:
        HTMLResponse: 包含主页 HTML 内容的响应对象
    """
    html_content = main_html
    return HTMLResponse(content=html_content)


# 将 Gradio 应用挂载到 FastAPI 路由
# 这样可以在同一个应用中同时使用 FastAPI 和 Gradio
app = gr.mount_gradio_app(app, get_chat_block(), path="/chat")
app = gr.mount_gradio_app(app, get_upload_block(), path="/upload_data")
app = gr.mount_gradio_app(app, get_knowledge_base_block(), path="/create_knowledge_base")


if __name__ == "__main__":
    """
    应用启动入口

    使用 uvicorn 作为 ASGI 服务器运行 FastAPI 应用
    配置：
    - host: 0.0.0.0 表示监听所有网络接口
    - port: 7866 是 Gradio 的默认端口
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7866)