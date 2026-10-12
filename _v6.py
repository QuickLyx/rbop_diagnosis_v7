# -*- coding: utf-8 -*-
"""隔离加载只读 v6 模块，不改变全局模块搜索路径。"""
import importlib.util
from pathlib import Path
import sys

V6_ROOT = Path(__file__).resolve().parent.parent / "rbop_diagnosis_v6"


def load_v6(name):
    """按独立模块名复用 v6 实现，并在执行前登记 dataclass 所需模块。"""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"无效的 v6 模块名：{name}")
    module_name = f"rbop_v6_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = V6_ROOT / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"缺少只读 v6 模块：{path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module
