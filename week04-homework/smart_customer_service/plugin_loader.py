"""
插件热重载：从 plugins 目录加载 LangChain 工具，支持重新扫描以热更新。
"""

import importlib.util
from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool


def _get_plugins_dir() -> Path:
    return Path(__file__).resolve().parent / "plugins"


def load_plugins(plugins_dir: str | Path | None = None) -> List[BaseTool]:
    """
    扫描 plugins_dir 下所有 .py（不含 __ 开头），动态加载并收集 LangChain 工具（BaseTool）。
    返回工具列表，供 create_agent(tools=...) 使用。热更新时重新调用即可。
    """
    if plugins_dir is None:
        plugins_dir = _get_plugins_dir()
    plugins_dir = Path(plugins_dir)
    tools: List[BaseTool] = []
    if not plugins_dir.exists():
        return tools
    for f in sorted(plugins_dir.iterdir()):
        if f.suffix != ".py" or f.name.startswith("__"):
            continue
        name = f.stem
        try:
            spec = importlib.util.spec_from_file_location(name, f)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if isinstance(obj, BaseTool):
                    tools.append(obj)
        except Exception:
            continue
    return tools
