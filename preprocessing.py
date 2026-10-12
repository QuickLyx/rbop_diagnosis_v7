# -*- coding: utf-8 -*-
"""直接复用 v6 的冻结 logspec512 口径，不重写特征计算。"""
from _v6 import load_v6

_v6prep = load_v6("preprocessing")
extract_logspec = _v6prep.extract_logspec
build_repr_config = _v6prep.build_repr_config
config_from_saved = _v6prep.config_from_saved
TARGET_FS = _v6prep.TARGET_FS

__all__ = ["extract_logspec", "build_repr_config", "config_from_saved", "TARGET_FS"]
