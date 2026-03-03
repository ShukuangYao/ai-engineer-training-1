"""
QAnything 服务端 API 请求处理器模块。

本模块提供知识库问答系统（QAnything）的 HTTP 接口实现，包括：
- 知识库管理：创建、删除、重命名、列举知识库
- 文档管理：上传文件/网页/FAQ、列举/删除文档、获取文档分块
- 问答与检索：基于知识库的对话（流式/非流式）、重排序结果
- Bot 管理：创建、删除、更新、查询 Bot 及 LLM 配置
- 用户与状态：用户校验、状态查询、健康检查、QA 日志与统计

所有接口均通过 Sanic 的 request 接收参数，返回 JSON 或流式响应。
依赖 LocalDocQA、Milvus/ES 等内核组件完成实际业务逻辑。
"""
import shutil

from qanything_kernel.core.local_file import LocalFile
from qanything_kernel.core.local_doc_qa import LocalDocQA
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.configs.model_config import (BOT_DESC, BOT_IMAGE, BOT_PROMPT, BOT_WELCOME,
                                                   DEFAULT_PARENT_CHUNK_SIZE, MAX_CHARS, VECTOR_SEARCH_TOP_K,
                                                   UPLOAD_ROOT_PATH, IMAGES_ROOT_PATH)
from qanything_kernel.utils.general_utils import *
from langchain.schema import Document
from sanic.response import ResponseStream
from sanic.response import json as sanic_json
from sanic.response import text as sanic_text
from sanic import request, response
import uuid
import json
import asyncio
import urllib.parse
import re
from datetime import datetime
from collections import defaultdict
import os
from tqdm import tqdm
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import base64

# ========== 对外接口列表（供路由注册） ==========
__all__ = ["new_knowledge_base", "upload_files", "list_kbs", "list_docs", "delete_knowledge_base", "delete_docs",
           "rename_knowledge_base", "get_total_status", "clean_files_by_status", "upload_weblink", "local_doc_chat",
           "document", "upload_faqs", "get_doc_completed", "get_qa_info", "get_user_id", "get_doc",
           "get_rerank_results", "get_user_status", "health_check", "update_chunks", "get_file_base64",
           "get_random_qa", "get_related_qa", "new_bot", "delete_bot", "update_bot", "get_bot_info"]

# ========== 全局常量 ==========
# 用户 ID 校验失败时的统一错误提示（user_id 须为字母开头，仅含字母、数字、下划线）
INVALID_USER_ID = f"fail, Invalid user_id: . user_id 必须只含有字母，数字和下划线且字母开头"
# 网关 IP：用于将请求中的 0.0.0.0/127.0.0.1/localhost 替换为实际可访问的地址（如容器内访问宿主机）
GATEWAY_IP = os.getenv("GATEWAY_IP", "localhost")
debug_logger.info(f"GATEWAY_IP: {GATEWAY_IP}")

# ========== 工具函数 ==========


async def run_in_background(func, *args):
    """
    在线程池中异步执行同步函数，避免阻塞 Sanic 事件循环。
    用于 Milvus/ES 删除等耗时同步 I/O 操作，保证主线程可继续处理其他请求。
    :param func: 要执行的同步函数
    :param args: 传给 func 的位置参数
    """
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        await loop.run_in_executor(pool, func, *args)


async def fetch(session, url, input_json):
    """
    使用 aiohttp session 向指定 URL 发送 POST JSON 请求，并返回解析后的 JSON。
    :param session: aiohttp ClientSession 实例
    :param url: 目标 URL
    :param input_json: 请求体（会被序列化为 JSON）
    :return: 响应 JSON 解析后的 Python 对象
    """
    headers = {'Content-Type': 'application/json'}
    async with session.post(url, json=input_json, headers=headers) as response:
        return await response.json()


def sync_function_with_args(arg1, arg2):
    """带参数的同步函数示例，用于模拟耗时操作（如测试 run_in_background）。"""
    import time
    time.sleep(5)
    print(f"同步函数执行完毕，参数值：arg1={arg1}, arg2={arg2}")


@get_time_async
async def new_knowledge_base(req: request):
    """
    创建新知识库。
    校验 user_id/user_info，生成或使用传入的 kb_id（须以 KB 开头），
    若 quick=True 则在 kb_id 后追加 _QUICK。在 Milvus 侧创建对应 base 并返回 kb_id、kb_name、timestamp。
    :param req: Sanic 请求，需含 user_id、user_info、kb_name，可选 kb_id、quick
    :return: JSON 含 code/msg/data（kb_id、kb_name、timestamp）或错误码 2001
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    # 从请求中安全获取用户标识并校验格式（字母开头，仅字母数字下划线）
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("new_knowledge_base %s", user_id)
    kb_name = safe_get(req, 'kb_name')
    debug_logger.info("kb_name: %s", kb_name)
    # 未传 kb_id 时自动生成 KB + uuid
    default_kb_id = 'KB' + uuid.uuid4().hex
    kb_id = safe_get(req, 'kb_id', default_kb_id)
    kb_id = correct_kb_id(kb_id)

    # 快速模式：kb_id 追加 _QUICK 标识，用于快速问答场景
    is_quick = safe_get(req, 'quick', False)
    if is_quick:
        kb_id += "_QUICK"

    if kb_id[:2] != 'KB':
        return sanic_json({"code": 2001, "msg": "fail, kb_id must start with 'KB'"})
    # 检查该 kb_id 是否已存在（check_kb_exist 返回不存在的 id 列表，空表示已存在）
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not not_exist_kb_ids:
        return sanic_json({"code": 2001, "msg": "fail, knowledge Base {} already exist".format(kb_id)})

    # 在 Milvus 与元数据表中创建新知识库
    local_doc_qa.milvus_summary.new_milvus_base(kb_id, user_id, kb_name)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    return sanic_json({"code": 200, "msg": "success create knowledge base {}".format(kb_id),
                       "data": {"kb_id": kb_id, "kb_name": kb_name, "timestamp": timestamp}})


@get_time_async
async def upload_weblink(req: request):
    """
    上传网页链接到指定知识库。
    支持单条 url+title 或批量 urls+titles。校验 kb_id 存在、URL 以 http 开头且长度≤2048，
    对标题做全角字符清理和文件名截断。mode：soft=跳过同名，strong=强制覆盖。将 URL 内容拉取为 LocalFile 并登记到 Milvus。
    :param req: Sanic 请求，需含 user_id、user_info、kb_id，以及 url+title 或 urls+titles；可选 mode、chunk_size
    :return: JSON 含 code、msg、data（file_id、file_name、file_url、status、bytes、timestamp）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("upload_weblink %s", user_id)
    debug_logger.info("user_info %s", user_info)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})

    # 支持单 URL（url + title）或批量（urls + titles）两种入参
    url = safe_get(req, 'url')
    if url:
        urls = [url]
        if url.endswith('/'):
            url = url[:-1]
        titles = [safe_get(req, 'title', url.split('/')[-1]) + '.web']
    else:
        urls = safe_get(req, 'urls')
        titles = safe_get(req, 'titles')
        if len(urls) != len(titles):
            return sanic_json({"code": 2003, "msg": "fail, urls and titles length not equal"})

    # 校验每条 URL：必须以 http 开头且长度不超过 2048
    for url in urls:
        if not url.startswith('http'):
            return sanic_json({"code": 2001, "msg": "fail, url must start with 'http'"})
        if len(url) > 2048:
            return sanic_json({"code": 2002, "msg": f"fail, url too long, max length is 2048."})

    # 标题清理：去除全角字符（\uFF01-\uFF5E、\u3000-\u303F），截断长度避免路径过长
    file_names = []
    for title in titles:
        debug_logger.info('ori name: %s', title)
        file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', title)
        debug_logger.info('cleaned name: %s', file_name)
        file_name = truncate_filename(file_name, max_length=110)
        file_names.append(file_name)

    # soft：同名不重复上传；strong：允许覆盖同名
    mode = safe_get(req, 'mode', default='soft')
    debug_logger.info("mode: %s", mode)
    chunk_size = safe_get(req, 'chunk_size', default=DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info("chunk_size: %s", chunk_size)

    # soft 模式下检查同名文件，已存在的文件名加入 exist_file_names 并跳过上传
    exist_file_names = []
    if mode == 'soft':
        exist_files = local_doc_qa.milvus_summary.check_file_exist_by_name(user_id, kb_id, file_names)
        exist_file_names = [f[1] for f in exist_files]
        for exist_file in exist_files:
            file_id, file_name, file_size, status = exist_file
            debug_logger.info(f"{url}, {status}, existed files, skip upload")
            # await post_data(user_id, -1, file_id, status, msg='existed files, skip upload')
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")

    # 遍历 URL 与文件名，对未存在的创建 LocalFile 并登记到 Milvus 摘要表
    data = []
    for url, file_name in zip(urls, file_names):
        if file_name in exist_file_names:
            continue
        # LocalFile 会拉取 URL 内容并写入本地
        local_file = LocalFile(user_id, kb_id, url, file_name)
        file_id = local_file.file_id
        file_size = len(local_file.file_content)
        file_location = local_file.file_location
        msg = local_doc_qa.milvus_summary.add_file(file_id, user_id, kb_id, file_name, file_size, file_location,
                                                   chunk_size, timestamp, url)
        debug_logger.info(f"{url}, {file_name}, {file_id}, {msg}")
        data.append({"file_id": file_id, "file_name": file_name, "file_url": url, "status": "gray", "bytes": 0,
                     "timestamp": timestamp})
        # asyncio.create_task(local_doc_qa.insert_files_to_milvus(user_id, kb_id, [local_file]))
    if exist_file_names:
        msg = f'warning，当前的mode是soft，无法上传同名文件{exist_file_names}，如果想强制上传同名文件，请设置mode：strong'
    else:
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


@get_time_async
async def upload_files(req: request):
    """
    上传本地文件到指定知识库。
    可从请求中取 files 或（use_local_file=true 时）从配置路径读取。校验 kb_id、总文件数≤10000、
    单文件字符数≤MAX_CHARS。对文件名做 URL 解码、全角清理、截断。mode 同 upload_weblink；chunk_size 用于分块。
    返回每个文件的 file_id、file_name、status(gray)、bytes、timestamp 等，超长文件会跳过并列入 failed_files。
    :param req: Sanic 请求，需含 user_id、user_info、kb_id；可选 files、use_local_file、mode、chunk_size
    :return: JSON 含 code、msg、data（每项含 file_id、file_name、status、bytes、timestamp、estimated_chars）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("upload_files %s", user_id)
    debug_logger.info("user_info %s", user_info)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id %s", kb_id)
    mode = safe_get(req, 'mode', default='soft')  # soft：跳过同名；strong：强制上传同名
    debug_logger.info("mode: %s", mode)
    chunk_size = safe_get(req, 'chunk_size', default=DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info("chunk_size: %s", chunk_size)
    # use_local_file=true 时从配置路径读取本地文件列表，否则从请求 multipart 中取 files
    use_local_file = safe_get(req, 'use_local_file', 'false')
    if use_local_file == 'true':
        files = read_files_with_extensions()
    else:
        files = req.files.getlist('files')
    debug_logger.info(f"{user_id} upload files number: {len(files)}")
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})

    # 单知识库下文件总数上限 10000
    exist_files = local_doc_qa.milvus_summary.get_files(user_id, kb_id)
    if len(exist_files) + len(files) > 10000:
        return sanic_json({"code": 2002,
                           "msg": f"fail, exist files is {len(exist_files)}, upload files is {len(files)}, total files is {len(exist_files) + len(files)}, max length is 10000."})

    data = []
    local_files = []
    file_names = []
    # 统一处理文件名：本地路径取 basename，上传文件取 name 并 URL 解码，去全角、截断
    for file in files:
        if isinstance(file, str):
            file_name = os.path.basename(file)
        else:
            debug_logger.info('ori name: %s', file.name)
            file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
            debug_logger.info('decode name: %s', file_name)
        # 删除全角字符，避免存储路径异常
        file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
        debug_logger.info('cleaned name: %s', file_name)
        file_name = truncate_filename(file_name, max_length=110)
        file_names.append(file_name)

    # soft 模式：同名文件不重复上传
    exist_file_names = []
    if mode == 'soft':
        exist_files = local_doc_qa.milvus_summary.check_file_exist_by_name(user_id, kb_id, file_names)
        exist_file_names = [f[1] for f in exist_files]
        for exist_file in exist_files:
            file_id, file_name, file_size, status = exist_file
            debug_logger.info(f"{file_name}, {status}, existed files, skip upload")
            # await post_data(user_id, -1, file_id, status, msg='existed files, skip upload')

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")

    # 超长文件（字符数 > MAX_CHARS）仅记录到 failed_files，不阻断其他文件
    failed_files = []
    for file, file_name in zip(files, file_names):
        if file_name in exist_file_names:
            continue
        local_file = LocalFile(user_id, kb_id, file, file_name)
        chars = fast_estimate_file_char_count(local_file.file_location)
        debug_logger.info(f"{file_name} char_size: {chars}")
        if chars and chars > MAX_CHARS:
            debug_logger.warning(f"fail, file {file_name} chars is {chars}, max length is {MAX_CHARS}.")
            failed_files.append(file_name)
            continue
        file_id = local_file.file_id
        file_size = len(local_file.file_content)
        file_location = local_file.file_location
        local_files.append(local_file)
        msg = local_doc_qa.milvus_summary.add_file(file_id, user_id, kb_id, file_name, file_size, file_location,
                                                   chunk_size, timestamp)
        debug_logger.info(f"{file_name}, {file_id}, {msg}")
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "bytes": len(local_file.file_content),
             "timestamp": timestamp, "estimated_chars": chars})

    # asyncio.create_task(local_doc_qa.insert_files_to_milvus(user_id, kb_id, local_files))
    if exist_file_names:
        msg = f'warning，当前的mode是soft，无法上传同名文件{exist_file_names}，如果想强制上传同名文件，请设置mode：strong'
    elif failed_files:
        msg = f"warning, {failed_files} chars is too much, max characters length is {MAX_CHARS}, skip upload."
    else:
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


@get_time_async
async def upload_faqs(req: request):
    """
    上传 FAQ（问答对）到指定知识库。
    支持 JSON 中的 faqs 列表，或通过 files 上传 Excel 解析为 faqs。单条 question≤512、answer≤2048，
    总条数≤1000。每条 FAQ 转为 LocalFile 并写入 FAQ 表与文件表，供后续检索与问答使用。
    :param req: Sanic 请求，需含 user_id、user_info、kb_id；可选 faqs、files、chunk_size
    :return: JSON 含 code、msg、data（每项含 file_id、file_name、status、length、timestamp）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("upload_faqs %s", user_id)
    debug_logger.info("user_info %s", user_info)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id %s", kb_id)
    faqs = safe_get(req, 'faqs')
    chunk_size = safe_get(req, 'chunk_size', default=DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info("chunk_size: %s", chunk_size)

    # 若无 faqs 则从请求中上传的 Excel 文件解析为 faqs 列表
    file_status = {}
    if faqs is None:
        files = req.files.getlist('files')
        faqs = []
        for file in files:
            debug_logger.info('ori name: %s', file.name)
            file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
            debug_logger.info('decode name: %s', file_name)
            # 删除掉全角字符
            file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
            file_name = file_name.replace("/", "_")
            debug_logger.info('cleaned name: %s', file_name)
            file_name = truncate_filename(file_name)
            file_faqs = check_and_transform_excel(file.body)
            if isinstance(file_faqs, str):
                file_status[file_name] = file_faqs
            else:
                faqs.extend(file_faqs)
                file_status[file_name] = "success"

    # FAQ 总条数上限 1000
    if len(faqs) > 1000:
        return sanic_json({"code": 2002, "msg": f"fail, faqs too many, max length is 1000."})

    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg})

    data = []
    local_files = []
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    debug_logger.info(f"start insert {len(faqs)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")
    # 逐条写入 FAQ 表与文件表：question 最长 512，answer 最长 2048
    for faq in tqdm(faqs):
        ques = faq['question']
        if len(ques) > 512 or len(faq['answer']) > 2048:
            return sanic_json(
                {"code": 2003, "msg": f"fail, faq too long, max length of question is 512, answer is 2048."})
        file_name = f"FAQ_{ques}.faq"
        file_name = file_name.replace("/", "_").replace(":", "_")  # 文件名中的/和：会导致写入时出错
        file_name = simplify_filename(file_name)
        file_size = len(ques) + len(faq['answer'])
        # faq_id = local_doc_qa.milvus_summary.get_faq_by_question(ques, kb_id)
        # if faq_id:
        #     debug_logger.info(f"faq question {ques} already exist, skip")
        #     data.append({
        #         "file_id": faq_id,
        #         "file_name": file_name,
        #         "status": "green",
        #         "length": file_size,
        #         "timestamp": local_doc_qa.milvus_summary.get_file_timestamp(faq_id)
        #     })
        #     continue
        local_file = LocalFile(user_id, kb_id, faq, file_name)
        file_id = local_file.file_id
        file_location = local_file.file_location
        local_files.append(local_file)
        local_doc_qa.milvus_summary.add_faq(file_id, user_id, kb_id, faq['question'], faq['answer'], faq.get('nos_keys', ''))
        local_doc_qa.milvus_summary.add_file(file_id, user_id, kb_id, file_name, file_size, file_location,
                                             chunk_size, timestamp)
        # debug_logger.info(f"{file_name}, {file_id}, {msg}, {faq}")
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "length": file_size,
             "timestamp": timestamp})
    debug_logger.info(f"end insert {len(faqs)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")

    msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


@get_time_async
async def list_kbs(req: request):
    """
    列举当前用户下的所有知识库，返回 kb_id 与 kb_name 列表。
    :param req: Sanic 请求，需含 user_id、user_info
    :return: JSON 含 code、data（[{kb_id, kb_name}, ...]）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("list_kbs %s", user_id)
    kb_infos = local_doc_qa.milvus_summary.get_knowledge_bases(user_id)
    data = []
    for kb in kb_infos:
        data.append({"kb_id": kb[0], "kb_name": kb[1]})
    debug_logger.info("all kb infos: {}".format(data))
    return sanic_json({"code": 200, "data": data})


@get_time_async
async def list_docs(req: request):
    """
    列举指定知识库下的文档列表，支持按 file_id 筛选与分页。
    返回每条的 file_id、file_name、status、bytes、content_length、timestamp、file_location、file_url、chunks_number、msg；
    若为 .faq 文件则附带 question/answer。按 timestamp 倒序，分页返回 total_page、total、status_count、details 等。
    :param req: Sanic 请求，需含 user_id、user_info、kb_id；可选 file_id、page_id、page_limit
    :return: JSON 含 code、msg、data（total_page、total、status_count、details、page_id、page_limit）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("list_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id: {}".format(kb_id))
    file_id = safe_get(req, 'file_id')
    page_id = safe_get(req, 'page_id', 1)
    page_limit = safe_get(req, 'page_limit', 10)
    data = []
    # 未传 file_id 时取该知识库下全部文件；传了则只取该 file_id
    if file_id is None:
        file_infos = local_doc_qa.milvus_summary.get_files(user_id, kb_id)
    else:
        file_infos = local_doc_qa.milvus_summary.get_files(user_id, kb_id, file_id)
    status_count = {}
    # msg_map = {'gray': "已上传到服务器，进入上传等待队列",
    #            'red': "上传出错，请删除后重试或联系工作人员",
    #            'yellow': "已进入上传队列，请耐心等待", 'green': "上传成功"}
    for file_info in file_infos:
        status = file_info[2]
        if status not in status_count:
            status_count[status] = 1
        else:
            status_count[status] += 1
        data.append({"file_id": file_info[0], "file_name": file_info[1], "status": file_info[2], "bytes": file_info[3],
                     "content_length": file_info[4], "timestamp": file_info[5], "file_location": file_info[6],
                     "file_url": file_info[7], "chunks_number": file_info[8], "msg": file_info[9]})
        if file_info[1].endswith('.faq'):
            faq_info = local_doc_qa.milvus_summary.get_faq(file_info[0])
            user_id, kb_id, question, answer, nos_keys = faq_info
            data[-1]['question'] = question
            data[-1]['answer'] = answer

    # 按 timestamp 倒序，时间越新的越靠前
    data = sorted(data, key=lambda x: int(x['timestamp']), reverse=True)

    # 分页：总记录数、总页数、当前页起止索引
    total_count = len(data)
    total_pages = (total_count + page_limit - 1) // page_limit
    if page_id > total_pages and total_count != 0:
        return sanic_json({"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{total_pages}，请检查！'})
    start_index = (page_id - 1) * page_limit
    end_index = start_index + page_limit
    current_page_data = data[start_index:end_index]

    # return sanic_json({"code": 200, "msg": "success", "data": {'total': status_count, 'details': data}})
    return sanic_json({
        "code": 200,
        "msg": "success",
        "data": {
            'total_page': total_pages,  # 总页数
            "total": total_count,  # 总文件数
            "status_count": status_count,  # 各状态的文件数
            "details": current_page_data,  # 当前页码下的文件目录
            "page_id": page_id,  # 当前页码,
            "page_limit": page_limit  # 每页显示的文件数
        }
    })


@get_time_async
async def delete_knowledge_base(req: request):
    """
    批量删除知识库。校验 kb_ids 均存在后，异步删除 Milvus 中对应数据、ES 中文件 chunk、
    摘要表中的文档与 FAQ，并删除该知识库在 UPLOAD_ROOT_PATH 下的文件目录。
    :param req: Sanic 请求，需含 user_id、user_info、kb_ids
    :return: JSON 含 code、msg
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("delete_knowledge_base %s", user_id)
    kb_ids = safe_get(req, 'kb_ids')
    kb_ids = [correct_kb_id(kb_id) for kb_id in kb_ids]
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    # 异步删除 Milvus 中该知识库下的向量数据（按 kb_id 表达式删除）
    for kb_id in kb_ids:
        expr = f"kb_id == \"{kb_id}\""
        asyncio.create_task(run_in_background(local_doc_qa.milvus_kb.delete_expr, expr))
    # 逐知识库：删除 ES 中文件 chunk、摘要表中的文档与 FAQ，并删除本地文件目录
    for kb_id in kb_ids:
        file_infos = local_doc_qa.milvus_summary.get_files(user_id, kb_id)
        file_ids = [file_info[0] for file_info in file_infos]
        file_chunks = [file_info[8] for file_info in file_infos]
        asyncio.create_task(run_in_background(local_doc_qa.es_client.delete_files, file_ids, file_chunks))
        local_doc_qa.milvus_summary.delete_documents(file_ids)
        local_doc_qa.milvus_summary.delete_faqs(file_ids)

        # 删除该知识库在 UPLOAD_ROOT_PATH 下的文件目录
        try:
            upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
            file_dir = os.path.join(upload_path, kb_id)
            debug_logger.info("delete_knowledge_base file dir : %s", file_dir)
            shutil.rmtree(file_dir)
        except Exception as e:
            debug_logger.error("An error occurred while constructing file paths: %s", str(e))


        debug_logger.info(f"""delete knowledge base {kb_id} success""")
    # 从元数据表中删除知识库记录
    local_doc_qa.milvus_summary.delete_knowledge_base(user_id, kb_ids)
    return sanic_json({"code": 200, "msg": "Knowledge Base {} delete success".format(kb_ids)})


@get_time_async
async def rename_knowledge_base(req: request):
    """
    将指定知识库重命名为 new_kb_name，仅更新元数据（kb_name），不涉及向量与文件。
    :param req: Sanic 请求，需含 user_id、user_info、kb_id、new_kb_name
    :return: JSON 含 code、msg
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("rename_knowledge_base %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    new_kb_name = safe_get(req, 'new_kb_name')
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    local_doc_qa.milvus_summary.rename_knowledge_base(user_id, kb_id, new_kb_name)
    return sanic_json({"code": 200, "msg": "Knowledge Base {} rename success".format(kb_id)})


@get_time_async
async def delete_docs(req: request):
    """
    删除指定知识库下的若干文档（file_ids）。校验 kb 与 file 存在后，异步从 Milvus/ES 删除向量与 chunk，
    从摘要表删除文档与 FAQ 记录，并删除 UPLOAD_ROOT_PATH 与 IMAGES_ROOT_PATH 下对应目录。
    :param req: Sanic 请求，需含 user_id、user_info、kb_id、file_ids
    :return: JSON 含 code、msg
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("delete_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    file_ids = safe_get(req, "file_ids")
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    valid_file_infos = local_doc_qa.milvus_summary.check_file_exist(user_id, kb_id, file_ids)
    if len(valid_file_infos) == 0:
        return sanic_json({"code": 2004, "msg": "fail, files {} not found".format(file_ids)})
    valid_file_ids = [file_info[0] for file_info in valid_file_infos]
    debug_logger.info("delete_docs valid_file_ids %s", valid_file_ids)
    # 异步删除 Milvus 中该 kb_id 下指定 file_id 的向量记录
    expr = f"""kb_id == "{kb_id}" and file_id in {valid_file_ids}"""
    asyncio.create_task(run_in_background(local_doc_qa.milvus_kb.delete_expr, expr))
    file_chunks = local_doc_qa.milvus_summary.get_chunk_size(valid_file_ids)
    asyncio.create_task(run_in_background(local_doc_qa.es_client.delete_files, valid_file_ids, file_chunks))

    # 从摘要表删除文件、文档、FAQ 记录
    local_doc_qa.milvus_summary.delete_files(kb_id, valid_file_ids)
    local_doc_qa.milvus_summary.delete_documents(valid_file_ids)
    local_doc_qa.milvus_summary.delete_faqs(valid_file_ids)
    # 删除 UPLOAD_ROOT_PATH 与 IMAGES_ROOT_PATH 下对应 file_id 目录
    for file_id in file_ids:
        try:
            upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
            file_dir = os.path.join(upload_path, kb_id, file_id)
            debug_logger.info("delete_docs file_dir %s", file_dir)
            # delete file dir
            shutil.rmtree(file_dir)
            # delele images dir
            images_dir = os.path.join(IMAGES_ROOT_PATH, file_id)
            debug_logger.info("delete_docs images_dir %s", images_dir)
            shutil.rmtree(images_dir)
        except Exception as e:
            debug_logger.error("An error occurred while constructing file paths: %s", str(e))

    return sanic_json({"code": 200, "msg": "documents {} delete success".format(valid_file_ids)})


@get_time_async
async def get_total_status(req: request):
    """
    获取用户下各知识库的文件状态统计。by_date=True 时按日期汇总；否则按 kb 返回 green/yellow/red/gray 数量。
    未传 user_id 时统计所有用户。
    :param req: Sanic 请求，需含 user_info；可选 user_id、by_date
    :return: JSON 含 code、status（按 user 和 kb/date 聚合的统计）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info('get_total_status %s', user_id)
    by_date = safe_get(req, 'by_date', False)
    # 未传 user_id 时统计所有用户；传了则只统计该用户
    if not user_id:
        users = local_doc_qa.milvus_summary.get_users()
        users = [user[0] for user in users]
    else:
        users = [user_id]
    res = {}
    for user in users:
        res[user] = {}
        if by_date:
            # 按日期汇总该用户各知识库状态
            res[user] = local_doc_qa.milvus_summary.get_total_status_by_date(user)
            continue
        # 按知识库汇总 green/yellow/red/gray 数量
        kbs = local_doc_qa.milvus_summary.get_knowledge_bases(user)
        for kb_id, kb_name in kbs:
            gray_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'gray')
            red_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'red')
            yellow_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'yellow')
            green_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'green')
            res[user][kb_name + kb_id] = {'green': len(green_file_infos), 'yellow': len(yellow_file_infos),
                                          'red': len(red_file_infos),
                                          'gray': len(gray_file_infos)}

    return sanic_json({"code": 200, "status": res})


@get_time_async
async def clean_files_by_status(req: request):
    """
    按状态清理文件：从摘要表中删除指定 kb_ids（或全部 kb）下状态为 gray/red/yellow 的文件记录。
    不删除向量与物理文件，仅移除元数据，用于清理异常或未完成的上传记录。
    :param req: Sanic 请求，需含 user_id、user_info；可选 kb_ids、status（gray/red/yellow）
    :return: JSON 含 code、msg、data（被清理的 file_name 列表）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info('clean_files_by_status %s', user_id)
    # 仅允许清理 gray/red/yellow 状态（不清理 green）
    status = safe_get(req, 'status', default='gray')
    if status not in ['gray', 'red', 'yellow']:
        return sanic_json({"code": 2003, "msg": "fail, status {} must be in ['gray', 'red', 'yellow']".format(status)})
    kb_ids = safe_get(req, 'kb_ids')
    kb_ids = [correct_kb_id(kb_id) for kb_id in kb_ids]
    if not kb_ids:
        # 未传 kb_ids 时对该用户下全部知识库执行清理
        kbs = local_doc_qa.milvus_summary.get_knowledge_bases(user_id)
        kb_ids = [kb[0] for kb in kbs]
    else:
        not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    gray_file_infos = local_doc_qa.milvus_summary.get_file_by_status(kb_ids, status)
    gray_file_ids = [f[0] for f in gray_file_infos]
    gray_file_names = [f[1] for f in gray_file_infos]
    debug_logger.info(f'{status} files number: {len(gray_file_names)}')
    # 仅从摘要表删除文件记录，不删向量与物理文件
    if gray_file_ids:
        # expr = f"file_id in \"{gray_file_ids}\""
        # asyncio.create_task(run_in_background(local_doc_qa.milvus_kb.delete_expr, expr))
        for kb_id in kb_ids:
            local_doc_qa.milvus_summary.delete_files(kb_id, gray_file_ids)
    return sanic_json({"code": 200, "msg": f"delete {status} files success", "data": gray_file_names})


@get_time_async
async def local_doc_chat(req: request):
    """
    知识库问答主接口。支持两种调用方式：
    1) 传 bot_id：从 Bot 配置中读取 kb_ids、custom_prompt、LLM 参数（api_base/api_key/model/top_k 等）。
    2) 传 kb_ids 及各项 LLM 参数。
    校验 kb 存在、top_k≤100、必填 LLM 参数齐全；若 only_need_search_results 与 streaming 同时为 True 则报错。
    无有效文件时退化为纯对话模式（kb_ids=[]）。支持流式（SSE）与非流式；流式结束时写入 [DONE] 并附带
    retrieval_documents、source_documents、time_record 等，并写入 QA 日志。
    :param req: Sanic 请求，需含 user_id、user_info、question；可选 bot_id 或 kb_ids+LLM 参数、streaming、history 等
    :return: 流式返回 SSE 流；非流式返回 JSON（code、response、history、source_documents、time_record 等）
    """
    preprocess_start = time.perf_counter()
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info('local_doc_chat %s', user_id)
    debug_logger.info('user_info %s', user_info)
    bot_id = safe_get(req, 'bot_id')
    if bot_id:
        # Bot 模式：从 Bot 配置中读取 kb_ids、custom_prompt、LLM 参数
        if not local_doc_qa.milvus_summary.check_bot_is_exist(bot_id):
            return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
        bot_info = local_doc_qa.milvus_summary.get_bot(None, bot_id)[0]
        bot_id, bot_name, desc, image, prompt, welcome, kb_ids_str, upload_time, user_id, llm_setting = bot_info
        kb_ids = kb_ids_str.split(',')
        if not kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, Bot {} unbound knowledge base.".format(bot_id)})
        custom_prompt = prompt
        if not llm_setting:
            return sanic_json({"code": 2003, "msg": "fail, Bot {} llm_setting is empty.".format(bot_id)})
        llm_setting = json.loads(llm_setting)
        rerank = llm_setting.get('rerank', True)
        only_need_search_results = llm_setting.get('only_need_search_results', False)
        need_web_search = llm_setting.get('networking', False)
        api_base = llm_setting.get('api_base', '')
        api_key = llm_setting.get('api_key', 'ollama')
        api_context_length = llm_setting.get('api_context_length', 4096)
        top_p = llm_setting.get('top_p', 0.99)
        temperature = llm_setting.get('temperature', 0.5)
        top_k = llm_setting.get('top_k', VECTOR_SEARCH_TOP_K)
        model = llm_setting.get('model', 'gpt-4o-mini')
        max_token = llm_setting.get('max_token')
        hybrid_search = llm_setting.get('hybrid_search', False)
        chunk_size = llm_setting.get('chunk_size', DEFAULT_PARENT_CHUNK_SIZE)
    else:
        # 非 Bot 模式：直接从请求中取 kb_ids 与 LLM 参数
        kb_ids = safe_get(req, 'kb_ids')
        custom_prompt = safe_get(req, 'custom_prompt', None)
        rerank = safe_get(req, 'rerank', default=True)
        only_need_search_results = safe_get(req, 'only_need_search_results', False)
        need_web_search = safe_get(req, 'networking', False)
        api_base = safe_get(req, 'api_base', '')
        # 容器/内网场景：将 0.0.0.0/127.0.0.1/localhost 替换为 GATEWAY_IP 以便客户端能访问
        api_base = api_base.replace('0.0.0.0', GATEWAY_IP).replace('127.0.0.1', GATEWAY_IP).replace('localhost',
                                                                                                    GATEWAY_IP)
        api_key = safe_get(req, 'api_key', 'ollama')
        api_context_length = safe_get(req, 'api_context_length', 4096)
        top_p = safe_get(req, 'top_p', 0.99)
        temperature = safe_get(req, 'temperature', 0.5)
        top_k = safe_get(req, 'top_k', VECTOR_SEARCH_TOP_K)

        model = safe_get(req, 'model', 'gpt-4o-mini')
        max_token = safe_get(req, 'max_token')

        hybrid_search = safe_get(req, 'hybrid_search', False)
        chunk_size = safe_get(req, 'chunk_size', DEFAULT_PARENT_CHUNK_SIZE)

    debug_logger.info('rerank %s', rerank)

    if len(kb_ids) > 20:
        return sanic_json({"code": 2005, "msg": "fail, kb_ids length should less than or equal to 20"})
    kb_ids = [correct_kb_id(kb_id) for kb_id in kb_ids]
    question = safe_get(req, 'question')
    streaming = safe_get(req, 'streaming', False)
    history = safe_get(req, 'history', [])

    if top_k > 100:
        return sanic_json({"code": 2003, "msg": "fail, top_k should less than or equal to 100"})

    missing_params = []
    if not api_base:
        missing_params.append('api_base')
    if not api_key:
        missing_params.append('api_key')
    if not api_context_length:
        missing_params.append('api_context_length')
    if not top_p:
        missing_params.append('top_p')
    if not top_k:
        missing_params.append('top_k')
    if top_p == 1.0:
        top_p = 0.99
    if not temperature:
        missing_params.append('temperature')

    if missing_params:
        missing_params_str = " and ".join(missing_params) if len(missing_params) > 1 else missing_params[0]
        return sanic_json({"code": 2003, "msg": f"fail, {missing_params_str} is required"})

    # 仅要检索结果时不能同时开启流式（流式会持续写 SSE，无法在结束时统一返回检索结果）
    if only_need_search_results and streaming:
        return sanic_json(
            {"code": 2006, "msg": "fail, only_need_search_results and streaming can't be True at the same time"})
    request_source = safe_get(req, 'source', 'unknown')

    debug_logger.info("history: %s ", history)
    debug_logger.info("question: %s", question)
    debug_logger.info("kb_ids: %s", kb_ids)
    debug_logger.info("user_id: %s", user_id)
    debug_logger.info("custom_prompt: %s", custom_prompt)
    debug_logger.info("model: %s", model)
    debug_logger.info("max_token: %s", max_token)
    debug_logger.info("request_source: %s", request_source)
    debug_logger.info("only_need_search_results: %s", only_need_search_results)
    debug_logger.info("bot_id: %s", bot_id)
    debug_logger.info("need_web_search: %s", need_web_search)
    debug_logger.info("api_base: %s", api_base)
    debug_logger.info("api_key: %s", api_key)
    debug_logger.info("api_context_length: %s", api_context_length)
    debug_logger.info("top_p: %s", top_p)
    debug_logger.info("top_k: %s", top_k)
    debug_logger.info("temperature: %s", temperature)
    debug_logger.info("hybrid_search: %s", hybrid_search)
    debug_logger.info("chunk_size: %s", chunk_size)

    time_record = {}
    if kb_ids:
        not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})
        # 将 FAQ 知识库（kb_id_FAQ）一并加入检索范围
        faq_kb_ids = [kb + '_FAQ' for kb in kb_ids]
        not_exist_faq_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, faq_kb_ids)
        exist_faq_kb_ids = [kb for kb in faq_kb_ids if kb not in not_exist_faq_kb_ids]
        debug_logger.info("exist_faq_kb_ids: %s", exist_faq_kb_ids)
        kb_ids += exist_faq_kb_ids

    file_infos = []
    for kb_id in kb_ids:
        file_infos.extend(local_doc_qa.milvus_summary.get_files(user_id, kb_id))
    valid_files = [fi for fi in file_infos if fi[2] == 'green']
    if len(valid_files) == 0:
        # 无有效（green）文件时退化为纯对话模式，不检索知识库
        debug_logger.info("valid_files is empty, use only chat mode.")
        kb_ids = []
    preprocess_end = time.perf_counter()
    time_record['preprocess'] = round(preprocess_end - preprocess_start, 2)
    # 更新知识库最近问答时间，用于统计与展示
    qa_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    for kb_id in kb_ids:
        local_doc_qa.milvus_summary.update_knowledge_base_latest_qa_time(kb_id, qa_timestamp)
    debug_logger.info("streaming: %s", streaming)
    if streaming:
        debug_logger.info("start generate answer")

        async def generate_answer(response):
            """SSE 流式生成：每段 delta 写一条 data 行；遇到 [DONE] 时写完整结果、来源文档与 time_record 并 eof。"""
            debug_logger.info("start generate...")
            async for resp, next_history in local_doc_qa.get_knowledge_based_answer(model=model,
                                                                                    max_token=max_token,
                                                                                    kb_ids=kb_ids,
                                                                                    query=question,
                                                                                    retriever=local_doc_qa.retriever,
                                                                                    chat_history=history,
                                                                                    streaming=True,
                                                                                    rerank=rerank,
                                                                                    custom_prompt=custom_prompt,
                                                                                    time_record=time_record,
                                                                                    need_web_search=need_web_search,
                                                                                    hybrid_search=hybrid_search,
                                                                                    web_chunk_size=chunk_size,
                                                                                    temperature=temperature,
                                                                                    api_base=api_base,
                                                                                    api_key=api_key,
                                                                                    api_context_length=api_context_length,
                                                                                    top_p=top_p,
                                                                                    top_k=top_k
                                                                                    ):
                chunk_data = resp["result"]
                if not chunk_data:
                    continue
                chunk_str = chunk_data[6:]
                # [DONE] 表示流式结束，此时写入完整结果、来源文档与 time_record，并写 QA 日志
                if chunk_str.startswith("[DONE]"):
                    retrieval_documents = format_source_documents(resp["retrieval_documents"])
                    source_documents = format_source_documents(resp["source_documents"])
                    result = next_history[-1][1]
                    time_record['chat_completed'] = round(time.perf_counter() - preprocess_start, 2)
                    if time_record.get('llm_completed', 0) > 0:
                        time_record['tokens_per_second'] = round(
                            len(result) / time_record['llm_completed'], 2)
                    formatted_time_record = format_time_record(time_record)
                    chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, "model": model,
                                 "product_source": request_source, 'time_record': formatted_time_record,
                                 'history': history,
                                 'condense_question': resp['condense_question'], 'prompt': resp['prompt'],
                                 'result': result, 'retrieval_documents': retrieval_documents,
                                 'source_documents': source_documents, 'bot_id': bot_id}
                    local_doc_qa.milvus_summary.add_qalog(**chat_data)
                    qa_logger.info("chat_data: %s", chat_data)
                    debug_logger.info("response: %s", chat_data['result'])
                    stream_res = {
                        "code": 200,
                        "msg": "success stream chat",
                        "question": question,
                        "response": result,
                        "model": model,
                        "history": next_history,
                        "condense_question": resp['condense_question'],
                        "source_documents": source_documents,
                        "retrieval_documents": retrieval_documents,
                        "time_record": formatted_time_record,
                        "show_images": resp.get('show_images', [])
                    }
                else:
                    # 流式中间 chunk：记录首包时间与回滚长度，返回 delta_answer
                    time_record['rollback_length'] = resp.get('rollback_length', 0)
                    if 'first_return' not in time_record:
                        time_record['first_return'] = round(time.perf_counter() - preprocess_start, 2)
                    chunk_js = json.loads(chunk_str)
                    delta_answer = chunk_js["answer"]
                    stream_res = {
                        "code": 200,
                        "msg": "success",
                        "question": "",
                        "response": delta_answer,
                        "history": [],
                        "source_documents": [],
                        "retrieval_documents": [],
                        "time_record": format_time_record(time_record),
                    }
                await response.write(f"data: {json.dumps(stream_res, ensure_ascii=False)}\n\n")
                if chunk_str.startswith("[DONE]"):
                    await response.eof()
                await asyncio.sleep(0.001)

        response_stream = ResponseStream(generate_answer, content_type='text/event-stream')
        return response_stream

    else:
        # 非流式：一次性获取完整回答，再写 QA 日志并返回 JSON
        async for resp, history in local_doc_qa.get_knowledge_based_answer(model=model,
                                                                           max_token=max_token,
                                                                           kb_ids=kb_ids,
                                                                           query=question,
                                                                           retriever=local_doc_qa.retriever,
                                                                           chat_history=history, streaming=False,
                                                                           rerank=rerank,
                                                                           custom_prompt=custom_prompt,
                                                                           time_record=time_record,
                                                                           only_need_search_results=only_need_search_results,
                                                                           need_web_search=need_web_search,
                                                                           hybrid_search=hybrid_search,
                                                                           web_chunk_size=chunk_size,
                                                                           temperature=temperature,
                                                                           api_base=api_base,
                                                                           api_key=api_key,
                                                                           api_context_length=api_context_length,
                                                                           top_p=top_p,
                                                                           top_k=top_k
                                                                           ):
            pass
        if only_need_search_results:
            return sanic_json(
                {"code": 200, "question": question, "source_documents": format_source_documents(resp)})
        retrieval_documents = format_source_documents(resp["retrieval_documents"])
        source_documents = format_source_documents(resp["source_documents"])
        formatted_time_record = format_time_record(time_record)
        chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, 'time_record': formatted_time_record,
                     'history': history, "condense_question": resp['condense_question'], "model": model,
                     "product_source": request_source,
                     'retrieval_documents': retrieval_documents, 'prompt': resp['prompt'], 'result': resp['result'],
                     'source_documents': source_documents, 'bot_id': bot_id}
        local_doc_qa.milvus_summary.add_qalog(**chat_data)
        qa_logger.info("chat_data: %s", chat_data)
        debug_logger.info("response: %s", chat_data['result'])
        return sanic_json({"code": 200, "msg": "success no stream chat", "question": question,
                           "response": resp["result"], "model": model,
                           "history": history, "condense_question": resp['condense_question'],
                           "source_documents": source_documents, "retrieval_documents": retrieval_documents,
                           "time_record": formatted_time_record})


@get_time_async
async def document(req: request):
    """
    返回 QAnything 产品介绍与 API 调用指南的 Markdown 文档（纯文本），供前端或文档页展示。
    包含 Base URL、鉴权方式、各接口简要说明等。
    """
    description = """
# QAnything 介绍
[戳我看视频>>>>>【有道QAnything介绍视频.mp4】](https://docs.popo.netease.com/docs/7e512e48fcb645adadddcf3107c97e7c)

**QAnything** (**Q**uestion and **A**nswer based on **Anything**) 是支持任意格式的本地知识库问答系统。

您的任何格式的本地文件都可以往里扔，即可获得准确、快速、靠谱的问答体验。

**目前已支持格式:**
* PDF
* Word(doc/docx)
* PPT
* TXT
* 图片
* 网页链接
* ...更多格式，敬请期待

# API 调用指南

## API Base URL

https://qanything.youdao.com

## 鉴权
目前使用微信鉴权,步骤如下:
1. 客户端通过扫码微信二维码(首次登录需要关注公众号)
2. 获取token
3. 调用下面所有API都需要通过authorization参数传入这个token

注意：authorization参数使用Bearer auth认证方式

生成微信二维码以及获取token的示例代码下载地址：[微信鉴权示例代码](https://docs.popo.netease.com/docs/66652d1a967e4f779594aef3306f6097)

## API 接口说明
    {
        "api": "/api/local_doc_qa/upload_files"
        "name": "上传文件",
        "description": "上传文件接口，支持多个文件同时上传，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/upload_weblink"
        "name": "上传网页链接",
        "description": "上传网页链接，自动爬取网页内容，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/local_doc_chat" 
        "name": "问答接口",
        "description": "知识库问答接口，指定知识库名称，上传用户问题，通过传入history支持多轮对话",
    },
    {
        "api": "/api/local_doc_qa/list_files" 
        "name": "文件列表",
        "description": "列出指定知识库下的所有文件名，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/delete_files" 
        "name": "删除文件",
        "description": "删除指定知识库下的指定文件，需要指定知识库名称",
    },

"""
    return sanic_text(description)


@get_time_async
async def get_doc_completed(req: request):
    """
    获取指定文件（file_id）的解析后的文档分块列表，支持分页。
    返回当前页的 chunks（含 page_content、metadata）、file_path、page_id、page_limit、total_count。
    page_content 中的图片引用会替换为可访问的 URL（replace_image_references）。
    :param req: Sanic 请求，需含 user_id、user_info、kb_id、file_id；可选 page_id、page_limit
    :return: JSON 含 code、msg、chunks、file_path、page_id、page_limit、total_count
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("get_doc_chunks %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id: {}".format(kb_id))
    file_id = safe_get(req, 'file_id')
    if not file_id:
        return sanic_json({"code": 2005, "msg": "fail, file_id is None"})
    debug_logger.info("file_id: {}".format(file_id))
    page_id = safe_get(req, 'page_id', 1)  # 默认为第一页
    page_limit = safe_get(req, 'page_limit', 10)  # 默认每页显示10条记录

    sorted_json_datas = local_doc_qa.milvus_summary.get_document_by_file_id(file_id)
    chunks = [json_data['kwargs'] for json_data in sorted_json_datas]

    # 分页：总记录数、总页数、当前页起止索引
    total_count = len(chunks)
    total_pages = (total_count + page_limit - 1) // page_limit
    if page_id > total_pages and total_count != 0:
        return sanic_json({"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{total_pages}，请检查！'})
    # 计算当前页的起始和结束索引
    start_index = (page_id - 1) * page_limit
    end_index = start_index + page_limit
    current_page_chunks = chunks[start_index:end_index]
    # 将 page_content 中的图片引用替换为可访问的 URL
    for chunk in current_page_chunks:
        chunk['page_content'] = replace_image_references(chunk['page_content'], file_id)

    file_location = local_doc_qa.milvus_summary.get_file_location(file_id)
    file_path = os.path.dirname(file_location)
    return sanic_json({"code": 200, "msg": "success", "chunks": current_page_chunks, "file_path": file_path,
                       "page_id": page_id, "page_limit": page_limit, "total_count": total_count})


@get_time_async
async def get_qa_info(req: request):
    """
    查询 QA 日志。支持按 user_id、bot_id、query、time_start/time_end、qa_ids 筛选。
    only_need_count=True 时仅返回按天统计的问答数量（qa_infos_by_day）；否则分页返回 need_info 指定字段。
    save_to_excel=True 时导出为 Excel 文件并返回文件下载响应。
    :param req: Sanic 请求，需含 user_id 或 any_kb_id；可选 query、bot_id、qa_ids、time_start、time_end、only_need_count、need_info、save_to_excel、page_id、page_limit
    :return: JSON 含 code、msg、qa_infos/qa_infos_by_day 或 Excel 文件下载
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    any_kb_id = safe_get(req, 'any_kb_id')
    user_id = safe_get(req, 'user_id')
    if user_id is None and not any_kb_id:
        return sanic_json({"code": 2005, "msg": "fail, user_id and any_kb_id is None"})
    if any_kb_id:
        any_kb_id = correct_kb_id(any_kb_id)
        debug_logger.info("get_qa_info %s", any_kb_id)
    if user_id:
        user_info = safe_get(req, 'user_info', "1234")
        passed, msg = check_user_id_and_user_info(user_id, user_info)
        if not passed:
            return sanic_json({"code": 2001, "msg": msg})
        user_id = user_id + '__' + user_info
        debug_logger.info("get_qa_info %s", user_id)
    query = safe_get(req, 'query')
    bot_id = safe_get(req, 'bot_id')
    qa_ids = safe_get(req, "qa_ids")
    time_start = safe_get(req, 'time_start')
    time_end = safe_get(req, 'time_end')
    time_range = get_time_range(time_start, time_end)
    if not time_range:
        return {"code": 2002, "msg": f'输入非法！time_start格式错误，time_start: {time_start}，示例：2024-10-05，请检查！'}
    only_need_count = safe_get(req, 'only_need_count', False)
    debug_logger.info(f"only_need_count: {only_need_count}")
    if only_need_count:
        # 仅返回按天统计的问答数量（qa_infos_by_day）
        need_info = ["timestamp"]
        qa_infos = local_doc_qa.milvus_summary.get_qalog_by_filter(need_info=need_info, user_id=user_id, time_range=time_range)
        qa_infos = sorted(qa_infos, key=lambda x: x['timestamp'])
        qa_infos = [qa_info['timestamp'] for qa_info in qa_infos]
        qa_infos = [qa_info[:10] for qa_info in qa_infos]
        qa_infos_by_day = dict(Counter(qa_infos))
        return sanic_json({"code": 200, "msg": "success", "qa_infos_by_day": qa_infos_by_day})

    page_id = safe_get(req, 'page_id', 1)
    page_limit = safe_get(req, 'page_limit', 10)
    default_need_info = ["qa_id", "user_id", "bot_id", "kb_ids", "query", "model", "product_source", "time_record",
                         "history", "condense_question", "prompt", "result", "retrieval_documents", "source_documents",
                         "timestamp"]
    need_info = safe_get(req, 'need_info', default_need_info)
    save_to_excel = safe_get(req, 'save_to_excel', False)
    qa_infos = local_doc_qa.milvus_summary.get_qalog_by_filter(need_info=need_info, user_id=user_id, query=query,
                                                               bot_id=bot_id, time_range=time_range,
                                                               any_kb_id=any_kb_id, qa_ids=qa_ids)
    if save_to_excel:
        # 导出为 Excel 并返回文件下载响应
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        file_name = f"QAnything_QA_{timestamp}.xlsx"
        file_path = export_qalogs_to_excel(qa_infos, need_info, file_name)
        return await response.file(file_path, filename=file_name,
                                   mime_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   headers={'Content-Disposition': f'attachment; filename="{file_name}"'})

    total_count = len(qa_infos)
    total_pages = (total_count + page_limit - 1) // page_limit
    if page_id > total_pages and total_count != 0:
        return sanic_json(
            {"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{total_pages}，请检查！'})
    # 计算当前页的起始和结束索引
    start_index = (page_id - 1) * page_limit
    end_index = start_index + page_limit
    # 截取当前页的数据
    current_qa_infos = qa_infos[start_index:end_index]
    msg = f"检测到的Log总数为{total_count}, 本次返回page_id为{page_id}的数据，每页显示{page_limit}条"

    # if len(qa_infos) > 100:
    #     pages = math.ceil(len(qa_infos) // 100)
    #     if page_id is None:
    #         msg = f"检索到的Log数超过100，需要分页返回，总数为{len(qa_infos)}, 请使用page_id参数获取某一页数据，参数范围：[0, {pages - 1}], 本次返回page_id为0的数据"
    #         qa_infos = qa_infos[:100]
    #         page_id = 0
    #     elif page_id >= pages:
    #         return sanic_json(
    #             {"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{pages - 1}，请检查！'})
    #     else:
    #         msg = f"检索到的Log数超过100，需要分页返回，总数为{len(qa_infos)}, page范围：[0, {pages - 1}], 本次返回page_id为{page_id}的数据"
    #         qa_infos = qa_infos[page_id * 100:(page_id + 1) * 100]
    # else:
    #     msg = f"检索到的Log数为{len(qa_infos)}，一次返回所有数据"
    #     page_id = 0
    return sanic_json({"code": 200, "msg": msg, "page_id": page_id, "page_limit": page_limit, "qa_infos": current_qa_infos, "total_count": total_count})


@get_time_async
async def get_random_qa(req: request):
    """
    随机获取若干条 QA 记录，用于抽样或展示。支持 time_start/time_end 时间范围与 need_info 字段筛选。
    同时返回统计信息：total_users、total_queries。
    :param req: Sanic 请求，可选 limit、time_start、time_end、need_info
    :return: JSON 含 code、msg、total_users、total_queries、qa_infos
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    limit = safe_get(req, 'limit', 10)
    time_start = safe_get(req, 'time_start')
    time_end = safe_get(req, 'time_end')
    need_info = safe_get(req, 'need_info')
    time_range = get_time_range(time_start, time_end)
    if not time_range:
        return {"code": 2002, "msg": f'输入非法！time_start格式错误，time_start: {time_start}，示例：2024-10-05，请检查！'}

    debug_logger.info(f"get_random_qa limit: {limit}, time_range: {time_range}")
    qa_infos = local_doc_qa.milvus_summary.get_random_qa_infos(limit=limit, time_range=time_range, need_info=need_info)

    counts = local_doc_qa.milvus_summary.get_statistic(time_range=time_range)
    return sanic_json({"code": 200, "msg": "success", "total_users": counts["total_users"],
                       "total_queries": counts["total_queries"], "qa_infos": qa_infos})


@get_time_async
async def get_related_qa(req: request):
    """
    根据 qa_id 获取该条 QA 详情及与之相关的近期/较早 QA，按知识库（kb_ids）分组为 recent_sections 与 older_sections。
    每条记录会附带 kb_names（知识库名称列表），便于前端展示。
    :param req: Sanic 请求，需含 qa_id；可选 need_info、need_more
    :return: JSON 含 code、msg、qa_info、recent_sections、older_sections
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    qa_id = safe_get(req, 'qa_id')
    if not qa_id:
        return sanic_json({"code": 2005, "msg": "fail, qa_id is None"})
    need_info = safe_get(req, 'need_info')
    need_more = safe_get(req, 'need_more', False)
    debug_logger.info("get_related_qa %s", qa_id)
    qa_log, recent_logs, older_logs = local_doc_qa.milvus_summary.get_related_qa_infos(qa_id, need_info, need_more)
    # 按 kb_ids 分组为 recent_sections，key 改为自增整数，每条 log 附带 kb_names
    recent_sections = defaultdict(list)
    for log in recent_logs:
        recent_sections[log['kb_ids']].append(log)
    for i, kb_ids in enumerate(list(recent_sections.keys())):
        kb_names = local_doc_qa.milvus_summary.get_knowledge_base_name(json.loads(kb_ids))
        kb_names = [kb_name for user_id, kb_id, kb_name in kb_names]
        kb_names = ','.join(kb_names)
        recent_sections[i] = recent_sections.pop(kb_ids)
        for log in recent_sections[i]:
            log['kb_names'] = kb_names

    older_sections = defaultdict(list)
    for log in older_logs:
        older_sections[log['kb_ids']].append(log)
    for i, kb_ids in enumerate(list(older_sections.keys())):
        kb_names = local_doc_qa.milvus_summary.get_knowledge_base_name(json.loads(kb_ids))
        kb_names = [kb_name for user_id, kb_id, kb_name in kb_names]
        kb_names = ','.join(kb_names)
        older_sections[i] = older_sections.pop(kb_ids)
        for log in older_sections[i]:
            log['kb_names'] = kb_names

    return sanic_json({"code": 200, "msg": "success", "qa_info": qa_log, "recent_sections": recent_sections,
                       "older_sections": older_sections})


@get_time_async
async def get_user_id(req: request):
    """
    根据知识库 kb_id 反查所属 user_id，用于权限或展示。
    :param req: Sanic 请求，需含 kb_id
    :return: JSON 含 code、msg、user_id
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    kb_id = safe_get(req, 'kb_id')
    kb_id = correct_kb_id(kb_id)
    debug_logger.info("kb_id: {}".format(kb_id))
    user_id = local_doc_qa.milvus_summary.get_user_by_kb_id(kb_id)
    if not user_id:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(kb_id)})
    else:
        return sanic_json({"code": 200, "msg": "success", "user_id": user_id})


@get_time_async
async def get_doc(req: request):
    """
    根据 doc_id（单条文档/分块 ID）查询文档内容，返回 doc_text（含 page_content 与 metadata）。
    :param req: Sanic 请求，需含 doc_id
    :return: JSON 含 code、msg、doc_text
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    doc_id = safe_get(req, 'doc_id')
    debug_logger.info("get_doc %s", doc_id)
    if not doc_id:
        return sanic_json({"code": 2005, "msg": "fail, doc_id is None"})
    doc_json_data = local_doc_qa.milvus_summary.get_document_by_doc_id(doc_id)
    return sanic_json({"code": 200, "msg": "success", "doc_text": doc_json_data['kwargs']})


@get_time_async
async def get_rerank_results(req: request):
    """
    对给定 query 与文档列表（doc_ids 或 doc_strs）做重排序，返回按相关性排序后的结果。
    用于单独调用重排模型或调试检索效果。
    :param req: Sanic 请求，需含 query，以及 doc_ids 或 doc_strs 之一
    :return: JSON 含 code、rerank_results（格式化后的来源文档列表）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    query = safe_get(req, 'query')
    if not query:
        return sanic_json({"code": 2005, "msg": "fail, query is None"})
    doc_ids = safe_get(req, 'doc_ids')
    doc_strs = safe_get(req, 'doc_strs')
    if not doc_ids and not doc_strs:
        return sanic_json({"code": 2005, "msg": "fail, doc_ids is None and doc_strs is None"})
    if doc_ids:
        rerank_results = await local_doc_qa.get_rerank_results(query, doc_ids=doc_ids)
    else:
        rerank_results = await local_doc_qa.get_rerank_results(query, doc_strs=doc_strs)

    return sanic_json({"code": 200, "msg": "success", "rerank_results": format_source_documents(rerank_results)})


@get_time_async
async def get_user_status(req: request):
    """
    查询用户状态：0 表示 green（正常），非 0 表示 red（异常或受限）。
    :param req: Sanic 请求，需含 user_id、user_info
    :return: JSON 含 code、status（green/red）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("get_user_status %s", user_id)
    user_status = local_doc_qa.milvus_summary.get_user_status(user_id)
    if user_status is None:
        return sanic_json({"code": 2003, "msg": "fail, user {} not found".format(user_id)})
    if user_status == 0:
        status = 'green'
    else:
        status = 'red'
    return sanic_json({"code": 200, "msg": "success", "status": status})


@get_time_async
async def health_check(req: request):
    """
    服务健康检查：正常返回 code=200，用于负载均衡或监控探活。
    异常时可改为返回 500 表示服务不可用。
    :param req: Sanic 请求（无必填参数）
    :return: JSON 含 code、msg
    """
    return sanic_json({"code": 200, "msg": "success"})


@get_time_async
async def get_bot_info(req: request):
    """
    获取 Bot 信息。不传 bot_id 时返回该用户下所有 Bot；传 bot_id 时返回该 Bot 详情。
    返回 bot_id、bot_name、description、head_image、prompt_setting、welcome_message、kb_ids、kb_names、update_time、llm_setting 等。
    :param req: Sanic 请求，需含 user_id、user_info；可选 bot_id
    :return: JSON 含 code、msg、data（Bot 列表或单条详情）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    bot_id = safe_get(req, 'bot_id')
    if bot_id:
        if not local_doc_qa.milvus_summary.check_bot_is_exist(bot_id):
            return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    debug_logger.info("get_bot_info %s", user_id)
    bot_infos = local_doc_qa.milvus_summary.get_bot(user_id, bot_id)
    data = []
    for bot_info in bot_infos:
        # bot_info[6] 为 kb_ids 字符串，解析后查 kb_names
        if bot_info[6] != "":
            kb_ids = bot_info[6].split(',')
            kb_infos = local_doc_qa.milvus_summary.get_knowledge_base_name(kb_ids)
            kb_names = []
            for kb_id in kb_ids:
                for kb_info in kb_infos:
                    if kb_id == kb_info[1]:
                        kb_names.append(kb_info[2])
                        break
        else:
            kb_ids = []
            kb_names = []
        info = {"bot_id": bot_info[0], "user_id": user_id, "bot_name": bot_info[1], "description": bot_info[2],
                "head_image": bot_info[3], "prompt_setting": bot_info[4], "welcome_message": bot_info[5],
                "kb_ids": kb_ids, "kb_names": kb_names,
                "update_time": bot_info[7].strftime("%Y-%m-%d %H:%M:%S"), "llm_setting": bot_info[9]}
        data.append(info)
    return sanic_json({"code": 200, "msg": "success", "data": data})


@get_time_async
async def new_bot(req: request):
    """
    创建新 Bot。校验 kb_ids 均存在后，生成 bot_id（BOT+uuid），写入名称、描述、头像、提示词、欢迎语、绑定知识库等。
    返回 bot_id、bot_name、create_time。
    :param req: Sanic 请求，需含 user_id、user_info、bot_name；可选 description、head_image、prompt_setting、welcome_message、kb_ids
    :return: JSON 含 code、msg、data（bot_id、bot_name、create_time）
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    bot_name = safe_get(req, "bot_name")
    desc = safe_get(req, "description", BOT_DESC)
    head_image = safe_get(req, "head_image", BOT_IMAGE)
    prompt_setting = safe_get(req, "prompt_setting", BOT_PROMPT)
    welcome_message = safe_get(req, "welcome_message", BOT_WELCOME)
    kb_ids = safe_get(req, "kb_ids", [])
    kb_ids_str = ",".join(kb_ids)

    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    debug_logger.info("new_bot %s", user_id)
    bot_id = 'BOT' + uuid.uuid4().hex
    # 写入 Bot 元数据（名称、描述、头像、提示词、欢迎语、绑定知识库）
    local_doc_qa.milvus_summary.new_qanything_bot(bot_id, user_id, bot_name, desc, head_image, prompt_setting,
                                                  welcome_message, kb_ids_str)
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return sanic_json({"code": 200, "msg": "success create qanything bot {}".format(bot_id),
                       "data": {"bot_id": bot_id, "bot_name": bot_name, "create_time": create_time}})


@get_time_async
async def delete_bot(req: request):
    """
    删除指定 bot_id 的 Bot，仅删除元数据，不删除绑定的知识库。
    :param req: Sanic 请求，需含 user_id、user_info、bot_id
    :return: JSON 含 code、msg
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("delete_bot %s", user_id)
    bot_id = safe_get(req, 'bot_id')
    if not local_doc_qa.milvus_summary.check_bot_is_exist(bot_id):
        return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    local_doc_qa.milvus_summary.delete_bot(user_id, bot_id)
    return sanic_json({"code": 200, "msg": "Bot {} delete success".format(bot_id)})


@get_time_async
async def update_bot(req: request):
    """
    更新 Bot 配置。可更新 bot_name、description、head_image、prompt_setting、welcome_message、kb_ids 以及
    llm_setting 中的 api_base、api_key、model、top_k、temperature、rerank、hybrid_search、networking 等。
    未传的字段保持原值。
    :param req: Sanic 请求，需含 user_id、user_info、bot_id；可选各 Bot/LLM 字段
    :return: JSON 含 code、msg
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("update_bot %s", user_id)
    bot_id = safe_get(req, 'bot_id')
    if not local_doc_qa.milvus_summary.check_bot_is_exist(bot_id):
        return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    bot_info = local_doc_qa.milvus_summary.get_bot(user_id, bot_id)[0]
    bot_name = safe_get(req, "bot_name", bot_info[1])
    description = safe_get(req, "description", bot_info[2])
    head_image = safe_get(req, "head_image", bot_info[3])
    prompt_setting = safe_get(req, "prompt_setting", bot_info[4])
    welcome_message = safe_get(req, "welcome_message", bot_info[5])
    kb_ids = safe_get(req, "kb_ids")
    if kb_ids is not None:
        not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
            return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
        kb_ids_str = ",".join(kb_ids)
    else:
        kb_ids_str = bot_info[6]

    # 从现有 Bot 的 llm_setting 中取默认值，请求中传入的字段覆盖
    llm_setting = json.loads(bot_info[9])
    if api_base := safe_get(req, "api_base"):
        llm_setting["api_base"] = api_base
    if api_key := safe_get(req, "api_key"):
        llm_setting["api_key"] = api_key
    if api_context_length := safe_get(req, "api_context_length"):
        llm_setting["api_context_length"] = api_context_length
    if top_p := safe_get(req, "top_p"):
        llm_setting["top_p"] = top_p
    if top_k := safe_get(req, "top_k"):
        llm_setting["top_k"] = top_k
    if chunk_size := safe_get(req, "chunk_size"):
        llm_setting["chunk_size"] = chunk_size
    if temperature := safe_get(req, "temperature"):
        llm_setting["temperature"] = temperature
    if model := safe_get(req, "model"):
        llm_setting["model"] = model
    if max_token := safe_get(req, "max_token"):
        llm_setting["max_token"] = max_token
    # rerank 为 None 时不改；显式传 False 也会更新
    rerank = safe_get(req, "rerank")
    if rerank is not None:
        llm_setting["rerank"] = rerank
    hybrid_search = safe_get(req, "hybrid_search")
    if hybrid_search is not None:
        llm_setting["hybrid_search"] = hybrid_search
    networking = safe_get(req, "networking")
    if networking is not None:
        llm_setting["networking"] = networking
    only_need_search_results = safe_get(req, "only_need_search_results")
    if only_need_search_results is not None:
        llm_setting["only_need_search_results"] = only_need_search_results

    debug_logger.info(f"update llm_setting: {llm_setting}")

    # 记录哪些字段发生了变更（便于排查与审计）
    if bot_name != bot_info[1]:
        debug_logger.info(f"update bot name from {bot_info[1]} to {bot_name}")
    if description != bot_info[2]:
        debug_logger.info(f"update bot description from {bot_info[2]} to {description}")
    if head_image != bot_info[3]:
        debug_logger.info(f"update bot head_image from {bot_info[3]} to {head_image}")
    if prompt_setting != bot_info[4]:
        debug_logger.info(f"update bot prompt_setting from {bot_info[4]} to {prompt_setting}")
    if welcome_message != bot_info[5]:
        debug_logger.info(f"update bot welcome_message from {bot_info[5]} to {welcome_message}")
    if kb_ids_str != bot_info[6]:
        debug_logger.info(f"update bot kb_ids from {bot_info[6]} to {kb_ids_str}")
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    debug_logger.info(f"update_time: {update_time}")
    local_doc_qa.milvus_summary.update_bot(user_id, bot_id, bot_name, description, head_image, prompt_setting,
                                           welcome_message, kb_ids_str, update_time, llm_setting)
    return sanic_json({"code": 200, "msg": "Bot {} update success".format(bot_id)})


@get_time_async
async def update_chunks(req: request):
    """
    更新单条文档分块内容。要求当前无 yellow 状态文件（无正在解析任务）。
    update_content 的 token 数不得超过 chunk_size；先更新摘要表，再从 Milvus 删除旧向量并重新插入新 Document。
    :param req: Sanic 请求，需含 user_id、user_info、doc_id、update_content；可选 chunk_size
    :return: JSON 含 code、msg
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    user_info = safe_get(req, 'user_info', "1234")
    passed, msg = check_user_id_and_user_info(user_id, user_info)
    if not passed:
        return sanic_json({"code": 2001, "msg": msg})
    user_id = user_id + '__' + user_info
    debug_logger.info("update_chunks %s", user_id)
    doc_id = safe_get(req, 'doc_id')
    debug_logger.info(f"doc_id: {doc_id}")
    # 若有文件处于解析中，不允许更新分块以免数据不一致
    yellow_files = local_doc_qa.milvus_summary.get_files_by_status("yellow")
    if len(yellow_files) > 0:
        return sanic_json({"code": 2002, "msg": f"fail, currently, there are {len(yellow_files)} files being parsed, please wait for all files to finish parsing before updating the chunk."})
    update_content = safe_get(req, 'update_content')
    debug_logger.info(f"update_content: {update_content}")
    chunk_size = safe_get(req, 'chunk_size', DEFAULT_PARENT_CHUNK_SIZE)
    debug_logger.info(f"chunk_size: {chunk_size}")
    update_content_tokens = num_tokens_embed(update_content)
    if update_content_tokens > chunk_size:
        return sanic_json({"code": 2003, "msg": f"fail, update_content too long, please reduce the length, "
                                                f"your update_content tokens is {update_content_tokens}, "
                                                f"the max tokens is {chunk_size}"})
    doc_json = local_doc_qa.milvus_summary.get_document_by_doc_id(doc_id)
    if not doc_json:
        return sanic_json({"code": 2004, "msg": "fail, DocId {} not found".format(doc_id)})
    doc = Document(page_content=update_content, metadata=doc_json['kwargs']['metadata'])
    doc.metadata['doc_id'] = doc_id
    # 先更新摘要表中的文档内容，再从 Milvus 删除旧向量并重新插入新 Document
    local_doc_qa.milvus_summary.update_document(doc_id, update_content)
    expr = f'doc_id == "{doc_id}"'
    local_doc_qa.milvus_kb.delete_expr(expr)
    await local_doc_qa.retriever.insert_documents([doc], chunk_size, True)
    return sanic_json({"code": 200, "msg": "success update doc_id {}".format(doc_id)})


@get_time_async
async def get_file_base64(req: request):
    """
    根据 file_id 获取文件在服务器上的存储路径，读取文件内容并做 Base64 编码返回。
    用于前端预览或下载原始文件（如 PDF、图片）。
    :param req: Sanic 请求，需含 file_id
    :return: JSON 含 code、msg、file_base64
    """
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    file_id = safe_get(req, 'file_id')
    debug_logger.info("get_file_base64 %s", file_id)
    file_location = local_doc_qa.milvus_summary.get_file_location(file_id)
    debug_logger.info("file_location %s", file_location)
    if not file_location:
        return sanic_json({"code": 2005, "msg": "fail, file_id is Invalid"})
    # 读取文件二进制并做 Base64 编码，便于前端预览或下载
    with open(file_location, "rb") as f:
        file_base64 = base64.b64encode(f.read()).decode()
    return sanic_json({"code": 200, "msg": "success", "file_base64": file_base64})
