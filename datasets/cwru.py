# -*- coding: utf-8 -*-
"""CWRU 四来源排 028 后枚举，复用 v6 通道读取与四折顺序。"""
from pathlib import Path

import numpy as np

from _v6 import load_v6
from .recording import Recording, _as_signal

_v6ds = load_v6("dataset")
SOURCES = ("Normal", "12k_DE", "12k_FE", "48k_DE")
ALL_CHANNELS = _v6ds.ALL_CHANNELS
LABEL_NAMES = _v6ds.LABEL_NAMES
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "rbop_diagnosis_v6" / "Data"


def enumerate_recordings(data_root=None):
    """153 个文件各产生 DE、FE 两条记录；文件是划分单元。"""
    files = _v6ds.enumerate_files(data_root or DEFAULT_DATA_ROOT, SOURCES, exclude_028=True)
    return [Recording(
        record_key=f"{r['file_key']}::{channel}", unit_key=r["file_key"],
        path=Path(r["path"]), dataset="cwru", channel=channel,
        label=r["label"], src_fs=r["raw_fs"], source=r["dataset"],
    ) for r in files for channel in ALL_CHANNELS]


def read_signal(record):
    """返回原采样率向量；特殊键前缀与唯一性检查直接沿用 v6。"""
    if record.dataset != "cwru" or record.source not in SOURCES:
        raise ValueError(f"不是有效 CWRU 记录：{record.record_key}")
    signal = _v6ds.read_channel_signal({
        "path": str(record.path), "dataset": record.source,
        "filename": record.path.name, "file_key": record.unit_key,
    }, record.channel)
    return _as_signal(signal)


def split_source_units(records, split_seed=0):
    """按类别单元级四折；随机数消耗、排序与 v6 保持一致。"""
    by_unit, seen = {}, set()
    for record in records:
        if record.dataset != "cwru" or record.label not in range(4):
            raise ValueError(f"源四折只能使用有四类真值的 CWRU 记录：{record.record_key}")
        if record.record_key in seen:
            raise ValueError(f"记录键重复：{record.record_key}")
        seen.add(record.record_key)
        if record.unit_key in by_unit and by_unit[record.unit_key] != record.label:
            raise ValueError(f"同单元标签不一致：{record.unit_key}")
        by_unit[record.unit_key] = record.label
    if not by_unit:
        raise ValueError("源四折记录为空")
    rng = np.random.default_rng(split_seed)
    parts = {}
    for label in range(4):
        keys = np.asarray(sorted(key for key, value in by_unit.items() if value == label))
        parts[label] = np.array_split(keys[rng.permutation(len(keys))], 4)
    folds = []
    for fold in range(4):
        val = sorted(str(key) for label in parts for key in parts[label][fold])
        train = sorted(set(by_unit) - set(val))
        if set(train) & set(val) or set(train) | set(val) != set(by_unit):
            raise ValueError("源四折划分未完整覆盖单元或产生重叠")
        folds.append({"fold": fold, "train": train, "val": val})
    return folds
