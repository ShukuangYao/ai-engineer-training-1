import hashlib
import os
import time
from typing import Dict, Any, Optional
from collections import OrderedDict
from threading import Lock

from .dsl_parser import DSLParser
from .graph_builder import GraphBuilder

class WorkflowManager:
    """工作流管理器：负责加载、缓存和执行 DSL。"""
    
    def __init__(self, cache_size: int = 100):
        self.cache_size = cache_size
        self._cache: OrderedDict[str, Any] = OrderedDict() # Hash -> CompiledGraph
        self._file_hashes: Dict[str, str] = {} # FilePath -> Hash
        self._lock = Lock()
        self.parser = DSLParser()

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的 MD5 哈希值。"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_workflow(self, file_path: str):
        """获取编译好的工作流对象，支持热更新缓存。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        current_hash = self._calculate_file_hash(file_path)
        
        with self._lock:
            # 检查是否已有缓存且哈希匹配
            if file_path in self._file_hashes:
                cached_hash = self._file_hashes[file_path]
                if cached_hash == current_hash and cached_hash in self._cache:
                    # 缓存命中，将该项移到末尾（表示最近使用）
                    self._cache.move_to_end(cached_hash)
                    return self._cache[cached_hash]
            
            # 缓存未命中或文件已更新，重新加载
            print(f"[WorkflowManager] 加载/更新工作流: {file_path}")
            dsl_data = self.parser.load_file(file_path)
            builder = GraphBuilder(dsl_data)
            graph = builder.build()
            
            # 更新缓存
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False) # 移除最久未使用的
            
            self._cache[current_hash] = graph
            self._file_hashes[file_path] = current_hash
            
            return graph

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self._file_hashes.clear()
