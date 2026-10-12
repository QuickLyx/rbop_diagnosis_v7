# -*- coding: utf-8 -*-
"""按记录一次生成冻结 logspec512 缓存，读取时拒绝过期或不完整产物。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from bearing_facts import BY_ID, CONDITIONS
from datasets import cwru, paderborn
from datasets.recording import Recording, to_windows
from preprocessing import build_repr_config, config_from_saved, extract_logspec

ROOT = Path(__file__).resolve().parent
CACHE_VERSION = "v7_logspec512_12k_drop_ka08_v1"
EXPECTED_COUNTS = {
    "cwru": {"records": 306, "units": 153, "windows": 68516},
    "paderborn": {"records": 2559, "units": 128, "windows": 238752,
                  "primary_records": 2319, "primary_units": 116, "primary_windows": 216350},
}
LOADERS = {"cwru": cwru, "paderborn": paderborn}


@dataclass
class FeatureCache:
    X: np.ndarray
    records: list[dict]
    version: str
    repr_config: dict


def _cache_paths(dataset, cache_dir):
    if dataset not in LOADERS:
        raise ValueError(f"未知数据集：{dataset}")
    folder = Path(cache_dir) if cache_dir is not None else ROOT / "outputs" / "features"
    return folder / f"{dataset}.npy", folder / f"{dataset}.json"


def _counts(dataset, records):
    result = {"records": len(records), "units": len({r["unit_key"] for r in records}),
              "windows": sum(r["stop"] - r["start"] for r in records)}
    if dataset == "paderborn":
        primary = [r for r in records if r["label"] is not None]
        result.update(primary_records=len(primary),
                      primary_units=len({r["unit_key"] for r in primary}),
                      primary_windows=sum(r["stop"] - r["start"] for r in primary))
    return result


def _validate_cache(dataset, X, payload):
    if payload.get("version") != CACHE_VERSION:
        raise ValueError("缓存口径版本过期或缺失，拒绝使用；请在新目录重新生成")
    if payload.get("dataset") != dataset or payload.get("script") != "features.py":
        raise ValueError("缓存数据集或生成脚本不匹配")
    config = config_from_saved(payload.get("repr_config"), require_complete=True)
    if config != build_repr_config():
        raise ValueError("缓存表征配置与冻结口径不一致")
    if X.ndim != 3 or X.shape[1:] != (1, 256) or X.dtype != np.float32:
        raise ValueError(f"缓存必须是 (N,1,256) float32，实际为 {X.shape} {X.dtype}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("缓存缺少记录元数据")
    required = {f.name for f in fields(Recording)} | {"start", "stop"}
    seen, unit_values, unit_channels = set(), {}, {}
    offset = 0
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("缓存记录元数据字段不完整")
        start, stop = record["start"], record["stop"]
        if (type(start) is not int or type(stop) is not int or start != offset
                or stop <= start or stop > len(X)):
            raise ValueError("缓存记录区间不连续、为空或超出特征行数")
        offset = stop
        key, unit = record["record_key"], record["unit_key"]
        if not isinstance(key, str) or not isinstance(unit, str) or key in seen:
            raise ValueError("缓存记录键无效或重复")
        seen.add(key)
        if record["dataset"] != dataset or not isinstance(record["path"], str):
            raise ValueError(f"缓存记录数据集或路径无效：{key}")
        label = record["label"]
        if label is not None and (type(label) is not int or label not in range(4)):
            raise ValueError(f"缓存记录标签无效：{key}")
        identity = tuple(record[name] for name in
                         ("label", "source", "bearing_id", "condition", "origin", "severity", "src_fs"))
        if unit in unit_values and identity != unit_values[unit]:
            raise ValueError(f"同单元的记录元数据不一致：{unit}")
        unit_values[unit] = identity
        unit_channels.setdefault(unit, []).append(record["channel"])
        if dataset == "cwru":
            source, channel = record["source"], record["channel"]
            if (source not in cwru.SOURCES or channel not in cwru.ALL_CHANNELS
                    or label is None or record["src_fs"] != cwru._v6ds.RAW_FS[source]
                    or unit != f"{source}/{Path(record['path']).name}"
                    or key != f"{unit}::{channel}"):
                raise ValueError(f"CWRU 缓存记录身份无效：{key}")
        else:
            bearing_id, condition = record["bearing_id"], record["condition"]
            bearing = BY_ID.get(bearing_id)
            if (bearing is None or condition not in CONDITIONS or record["src_fs"] != 64000
                    or record["source"] != "paderborn" or record["channel"] != "vibration_1"
                    or unit != f"{bearing_id}/{condition}"
                    or key != f"{bearing_id}/{Path(record['path']).name}::vibration_1"
                    or label != bearing.label or record["origin"] != bearing.origin
                    or record["severity"] != bearing.severity):
                raise ValueError(f"PU 缓存记录身份或真值无效：{key}")
    if offset != len(X):
        raise ValueError("缓存记录区间未覆盖全部特征")
    for unit, channels in unit_channels.items():
        if dataset == "cwru":
            expected = Counter(cwru.ALL_CHANNELS)
        else:
            expected = Counter({"vibration_1": 19 if unit == "KA08/N15_M01_F10" else 20})
        if Counter(channels) != expected:
            raise ValueError(f"缓存单元的记录数量或通道不完整：{unit}")
    counts = _counts(dataset, records)
    if counts != EXPECTED_COUNTS[dataset] or payload.get("counts") != counts:
        raise ValueError(f"缓存计数与冻结事实不一致：{counts}")
    if not np.isfinite(X).all():
        raise ValueError("缓存包含非有限特征")
    return FeatureCache(X, records, CACHE_VERSION, config)


def load_cache(dataset, cache_dir=None):
    """只读已有缓存，不读取原始信号，不静默重建或接受部分产物。"""
    array_path, metadata_path = _cache_paths(dataset, cache_dir)
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"缓存缺失或未完成：{array_path}、{metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("缓存元数据必须是字典")
    X = np.load(array_path, mmap_mode="r", allow_pickle=False)
    try:
        return _validate_cache(dataset, X, payload)
    except BaseException:
        # Windows 下异常对象可能继续持有数组，必须释放失败缓存的文件映射。
        if isinstance(X, np.memmap):
            X._mmap.close()
        raise


def build_cache(dataset, cache_dir=None, data_root=None):
    """逐记录提取一次；临时数组、JSON 均验证成功后再发布，不覆盖已有缓存。"""
    array_path, metadata_path = _cache_paths(dataset, cache_dir)
    if array_path.exists() or metadata_path.exists():
        return load_cache(dataset, cache_dir)
    loader = LOADERS[dataset]
    recordings = (loader.enumerate_recordings(data_root, include_kb=True)
                  if dataset == "paderborn" else loader.enumerate_recordings(data_root))
    config = build_repr_config()
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    X = None
    try:
        for suffix in (".npy", ".json"):
            with tempfile.NamedTemporaryFile(prefix=f".{dataset}_", suffix=suffix,
                                             dir=array_path.parent, delete=False) as handle:
                temporary.append(Path(handle.name))
        X = np.lib.format.open_memmap(temporary[0], mode="w+", dtype=np.float32,
                                     shape=(EXPECTED_COUNTS[dataset]["windows"], 1, 256))
        metadata, offset = [], 0
        for recording in recordings:
            windows = to_windows(loader.read_signal(recording), recording.src_fs)
            features = extract_logspec(windows, config)
            stop = offset + len(features)
            if stop > len(X):
                raise ValueError("实际窗口数超出冻结事实，拒绝发布缓存")
            X[offset:stop] = features
            record = asdict(recording)
            record.update(path=str(recording.path), start=offset, stop=stop)
            metadata.append(record)
            offset = stop
        if offset != len(X):
            raise ValueError(f"实际窗口数与冻结事实不一致：{offset}，预期 {len(X)}")
        payload = {"dataset": dataset, "version": CACHE_VERSION, "script": "features.py",
                   "repr_config": config, "counts": _counts(dataset, metadata), "records": metadata}
        _validate_cache(dataset, X, payload)
        X.flush()
        X._mmap.close()
        del X
        X = None
        temporary[1].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if array_path.exists() or metadata_path.exists():
            raise FileExistsError("缓存生成期间出现已有产物，拒绝覆盖")
        temporary[0].rename(array_path)
        temporary[1].rename(metadata_path)
    finally:
        if X is not None:
            # Windows 下异常栈仍可能引用数组，须先关闭映射才能删除临时文件。
            X._mmap.close()
            del X
        for path in temporary:
            path.unlink(missing_ok=True)
    return load_cache(dataset, cache_dir)


def feature_subset(cache, unit_keys=None, channels=None, labeled=True):
    """按单元与通道显式选择；有标签模式遇 KB 报错，无标签模式完全不用真值。"""
    units = None if unit_keys is None else set(unit_keys)
    selected_channels = None if channels is None else set(channels)
    if units is not None and units - {r["unit_key"] for r in cache.records}:
        raise KeyError(f"缓存中不存在的单元：{sorted(units - {r['unit_key'] for r in cache.records})[:3]}")
    if selected_channels is not None and selected_channels - {r["channel"] for r in cache.records}:
        raise KeyError(f"缓存中不存在的通道：{sorted(selected_channels)}")
    parts, labels, record_keys, row_channels, unit_rows, records = [], [], [], [], [], []
    offset = 0
    for record in cache.records:
        if units is not None and record["unit_key"] not in units:
            continue
        if selected_channels is not None and record["channel"] not in selected_channels:
            continue
        if labeled and record["label"] is None:
            raise ValueError(f"标注子集含无真值 KB 记录，请显式排除：{record['record_key']}")
        part = cache.X[record["start"]:record["stop"]]
        n = len(part)
        parts.append(part)
        labels.extend([record["label"] if labeled else -1] * n)
        record_keys.extend([record["record_key"]] * n)
        row_channels.extend([record["channel"]] * n)
        unit_rows.extend([record["unit_key"]] * n)
        records.append({**record, "start": offset, "stop": offset + n})
        offset += n
    X = np.concatenate(parts) if parts else np.empty((0, 1, 256), dtype=np.float32)
    return SimpleNamespace(X=torch.from_numpy(X), y=torch.tensor(labels, dtype=torch.long),
                           record_keys=np.asarray(record_keys, dtype=object),
                           channels=np.asarray(row_channels, dtype=object),
                           unit_keys=np.asarray(unit_rows, dtype=object), records=records)


def _self_check():
    """仅枚举身份和抽样前处理，不跑全量缓存、模型或 BN。"""
    source = cwru.enumerate_recordings()
    assert len(source) == 306 and len({r.unit_key for r in source}) == 153
    labels = {r.unit_key: r.label for r in source}
    assert Counter(labels.values()) == {0: 4, 1: 36, 2: 36, 3: 77}
    folds = cwru.split_source_units(source)
    assert sum(len(f["val"]) for f in folds) == 153
    primary = paderborn.enumerate_recordings()
    target = paderborn.enumerate_recordings(include_kb=True)
    assert len(primary) == 2319 and len(target) == 2559
    assert len({r.unit_key for r in primary}) == 116
    for loader, recording in ((cwru, source[0]), (paderborn, primary[0])):
        X = extract_logspec(to_windows(loader.read_signal(recording), recording.src_fs))
        assert X.shape[1:] == (1, 256) and X.dtype == np.float32
        assert np.max(np.abs(X.mean(axis=2))) < 1e-5
    print("数据与特征抽样自检通过（未生成全量缓存，未加载模型）")


def main():
    parser = argparse.ArgumentParser(description="生成一次性 logspec512 特征缓存（无训练）")
    parser.add_argument("--dataset", choices=("cwru", "paderborn", "all"), default="all")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--self-check", action="store_true", help="仅做枚举与抽样自检")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    datasets = tuple(LOADERS) if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        cache = build_cache(dataset, args.cache_dir)
        print(f"{dataset} 缓存就绪：{len(cache.records)} 条记录，{len(cache.X)} 窗，{cache.version}")


if __name__ == "__main__":
    main()
