# -*- coding: utf-8 -*-
"""数据层小样本检查：不训练、不重估 BN、不生成全量缓存。"""
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
from scipy.signal import resample_poly
import torch

from _v6 import load_v6
from bearing_facts import BY_ID, CONDITIONS
from datasets import cwru, paderborn
from datasets.recording import Recording, to_windows
import features
from preprocessing import build_repr_config, config_from_saved, extract_logspec


class DataTests(unittest.TestCase):
    def test_frozen_record_and_exact_resampling(self):
        record = Recording("r", "u", Path("test.mat"), "cwru", "DE", 0, 12000)
        with self.assertRaises(FrozenInstanceError):
            record.label = 1
        signal = np.random.default_rng(0).normal(size=8193)
        for rate, up, down in ((64000, 3, 16), (48000, 1, 4), (12000, 1, 1)):
            result = to_windows(signal, rate)
            expected = signal if rate == 12000 else resample_poly(signal, up, down)
            np.testing.assert_array_equal(result, expected[:len(result) * 512].reshape(-1, 512))
        self.assertEqual(to_windows(np.arange(1025), 12000).shape, (2, 512))
        for signal in ([], [np.nan] * 512, [np.inf] * 512, np.zeros((2, 512)),
                       np.zeros(511), np.ones(512, dtype=complex)):
            with self.subTest(shape=np.asarray(signal).shape):
                with self.assertRaises(ValueError):
                    to_windows(signal, 12000)
        for rate in (0, -12000, 12000.5, True):
            with self.assertRaises(ValueError):
                to_windows(np.zeros(512), rate)
        with self.assertRaises(ValueError):
            to_windows(np.zeros(512), 64000)

    def test_v6_isolation_and_bitwise_features(self):
        original = list(sys.path)
        module = load_v6("preprocessing")
        self.assertIs(module, sys.modules["rbop_v6_preprocessing"])
        self.assertIs(extract_logspec, module.extract_logspec)
        self.assertEqual(sys.path, original)
        X = extract_logspec(np.random.default_rng(0).normal(size=(3, 512)))
        self.assertEqual(X.shape, (3, 1, 256))
        self.assertEqual(X.dtype, np.float32)
        self.assertLess(np.max(np.abs(X.mean(axis=2))), 1e-5)
        self.assertEqual(config_from_saved(build_repr_config(), True), build_repr_config())
        with self.assertRaises(ValueError):
            load_v6("../dataset")

    def test_cwru_counts_folds_and_channel_keys(self):
        records = cwru.enumerate_recordings()
        self.assertEqual(len(records), 306)
        files = cwru._v6ds.enumerate_files(cwru.DEFAULT_DATA_ROOT, cwru.SOURCES, exclude_028=True)
        self.assertEqual(cwru.split_source_units(records), cwru._v6ds.split_source_files(files))
        folds = cwru.split_source_units(records)
        for fold in folds:
            self.assertFalse(set(fold["train"]) & set(fold["val"]))
            self.assertEqual(len(fold["train"]) + len(fold["val"]), 153)
        self.assertEqual(len({k for fold in folds for k in fold["val"]}), 153)
        with self.assertRaises(ValueError):
            cwru.split_source_units([records[0], records[0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "B007_0.mat"
            record = Recording("12k_DE/B007_0.mat::DE", "12k_DE/B007_0.mat", path,
                               "cwru", "DE", 1, 12000, source="12k_DE")
            savemat(path, {"X001_DE_time": np.arange(1024), "X002_DE_time": np.arange(1024)})
            with self.assertRaises(ValueError):
                cwru.read_signal(record)
            savemat(path, {"X001_FE_time": np.arange(1024)})
            with self.assertRaises(ValueError):
                cwru.read_signal(record)
            savemat(path, {"X001_DE_time": np.arange(1024)})
            np.testing.assert_array_equal(cwru.read_signal(record), np.arange(1024))
            special = replace(record, path=Path(directory) / "Normal_2.mat", source="Normal",
                              unit_key="Normal/Normal_2.mat", record_key="Normal/Normal_2.mat::DE")
            savemat(special.path, {"X001_DE_time": np.arange(1024), "X099_DE_time": np.arange(1024) + 1})
            np.testing.assert_array_equal(cwru.read_signal(special), np.arange(1024) + 1)

    def test_pu_channels_unique_and_vibration_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "N15_M07_F10_K001_1.mat"
            def save(names):
                channels = np.empty(len(names), dtype=[("Name", object), ("Data", object)])
                for index, name in enumerate(names):
                    channels[index] = (name, np.arange(1024))
                savemat(path, {path.stem: {"Y": channels}})
            save(["vibration_1", "speed"])
            self.assertEqual(set(paderborn.read_channels(path)), {"vibration_1", "speed"})
            record = Recording("K001/test.mat::vibration_1", "K001/N15_M07_F10", path,
                               "paderborn", "vibration_1", 0, 64000)
            np.testing.assert_array_equal(paderborn.read_signal(record), np.arange(1024))
            for names in (["vibration_1", "vibration_1"], ["speed"]):
                save(names)
                with self.assertRaises(ValueError):
                    paderborn.read_channels(path)
            savemat(path, {"wrong": {"Y": []}})
            with self.assertRaises(ValueError):
                paderborn.read_channels(path)

    def test_pu_counts_and_non_overwriting_drop_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "extracted"
            # 这里只枚举文件名，临时空文件即可覆盖完整口径与 KA08 丢弃凭据。
            for bearing_id in BY_ID:
                folder = root / bearing_id
                folder.mkdir(parents=True)
                for condition in CONDITIONS:
                    for rep in range(1, 21):
                        (folder / f"{condition}_{bearing_id}_{rep}.mat").touch()
            dropped_path = Path(directory) / "dropped.json"
            primary = paderborn.enumerate_recordings(root, dropped_path=dropped_path)
            all_records = paderborn.enumerate_recordings(root, include_kb=True, dropped_path=dropped_path)
            self.assertEqual(len(primary), 2319)
            self.assertEqual(len(all_records), 2559)
            self.assertEqual(len({r.unit_key for r in primary}), 116)
            self.assertEqual(sum(r.label is None for r in all_records), 240)
            dropped = json.loads(dropped_path.read_text(encoding="utf-8"))
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0]["filename"], "N15_M01_F10_KA08_2.mat")
            self.assertFalse(any("N15_M01_F10_KA08_2.mat" in r.record_key for r in all_records))
            self.assertEqual(sum(r.unit_key == "KA08/N15_M01_F10" for r in all_records), 19)
            dropped_path.write_text("[]", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                paderborn.enumerate_recordings(root, dropped_path=dropped_path)
            self.assertEqual(dropped_path.read_text(encoding="utf-8"), "[]")

    def test_cache_identity_accepts_windows_and_posix_paths_without_weakening_checks(self):
        condition = next(iter(CONDITIONS))
        bearing = BY_ID["K001"]
        target = []
        for index in range(20):
            filename = f"{condition}_K001_{index + 1}.mat"
            target.append(Recording(f"K001/{filename}::vibration_1", f"K001/{condition}",
                                    Path(filename), "paderborn", "vibration_1", bearing.label,
                                    64000, "paderborn", "K001", condition, bearing.origin, bearing.severity))
        for dataset, recordings, counts in (
                ("cwru", cwru.enumerate_recordings()[:2], {"records": 2, "units": 1, "windows": 2}),
                ("paderborn", target, {"records": 20, "units": 1, "windows": 20,
                                      "primary_records": 20, "primary_units": 1, "primary_windows": 20})):
            X = np.zeros((len(recordings), 1, 256), dtype=np.float32)
            for prefix in ("E:\\冻结缓存\\", "cache/"):
                records = [{**asdict(record), "path": prefix + record.path.name,
                            "start": index, "stop": index + 1}
                           for index, record in enumerate(recordings)]
                payload = {"version": features.CACHE_VERSION, "dataset": dataset,
                           "script": "features.py", "repr_config": build_repr_config(),
                           "records": records, "counts": counts}
                with self.subTest(dataset=dataset, prefix=prefix), \
                        patch.dict(features.EXPECTED_COUNTS, {dataset: counts}), \
                        patch.object(features, "Path", PurePosixPath):
                    self.assertEqual(len(features._validate_cache(dataset, X, payload).records), len(records))
                    records[0]["path"] = prefix + "wrong.mat"
                    with self.assertRaisesRegex(ValueError, "身份"):
                        features._validate_cache(dataset, X, payload)

    def test_small_cache_roundtrip_rejects_stale_or_corrupt_metadata(self):
        records = cwru.enumerate_recordings()[:2]
        counts = {"records": 2, "units": 1, "windows": 2}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=records), \
                patch.object(cwru, "read_signal", return_value=np.arange(2048)) as read:
            cache = features.build_cache("cwru", directory)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(cache.X.shape, (2, 1, 256))
            self.assertFalse(cache.X.flags.writeable)
            self.assertIsInstance(cache.records[0]["path"], str)
            features.build_cache("cwru", directory)
            self.assertEqual(read.call_count, 2)
            subset = features.feature_subset(cache, channels=["FE"])
            self.assertEqual(subset.X.shape, (1, 1, 256))
            self.assertEqual(subset.X.dtype, torch.float32)
            self.assertEqual(subset.y.dtype, torch.long)
            self.assertEqual((subset.records[0]["start"], subset.records[0]["stop"]), (0, 1))
            self.assertEqual(cache.records[1]["start"], 1)
            empty = features.feature_subset(cache, unit_keys=[])
            self.assertEqual(empty.X.shape, (0, 1, 256))
            with self.assertRaises(KeyError):
                features.feature_subset(cache, unit_keys=["unknown"])
            metadata_path = Path(directory) / "cwru.json"
            original = metadata_path.read_text(encoding="utf-8")
            for mutation in (lambda p: p.update(version="stale"),
                             lambda p: p["repr_config"].update(fs=64000),
                             lambda p: p["counts"].update(windows=3),
                             lambda p: p["records"][1].update(start=0),
                             lambda p: p["records"][0].pop("origin"),
                             lambda p: p["records"][1].update(record_key=p["records"][0]["record_key"])):
                payload = json.loads(original)
                mutation(payload)
                metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    features.load_cache("cwru", directory)
            metadata_path.write_text(original, encoding="utf-8")
            del cache
            with patch("features.np.load", return_value=np.zeros((2, 256), np.float32)):
                with self.assertRaises(ValueError):
                    features.load_cache("cwru", directory)
            with patch("features.np.load", return_value=np.full((2, 1, 256), np.nan, np.float32)):
                with self.assertRaises(ValueError):
                    features.load_cache("cwru", directory)

    def test_failed_read_releases_windows_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            array_path = Path(directory) / "cwru.npy"
            metadata_path = Path(directory) / "cwru.json"
            np.save(array_path, np.zeros((2, 1, 256), np.float32))
            metadata_path.write_text(json.dumps({"version": "stale"}), encoding="utf-8")
            saved_error = None
            try:
                features.load_cache("cwru", directory)
            except ValueError as error:
                saved_error = error
                # 保留异常及其栈时仍能删除文件，才证明映射已经释放。
                array_path.unlink()
                metadata_path.unlink()
            self.assertIsNotNone(saved_error)
            self.assertIn("过期", str(saved_error))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unlabeled_kb_not_silently_filtered(self):
        condition = next(iter(CONDITIONS))
        records = []
        for bearing_id in ("K001", "KB23"):
            bearing = BY_ID[bearing_id]
            for rep in range(1, 21):
                filename = f"{condition}_{bearing_id}_{rep}.mat"
                records.append(Recording(f"{bearing_id}/{filename}::vibration_1",
                                         f"{bearing_id}/{condition}", Path(bearing_id) / filename,
                                         "paderborn", "vibration_1", bearing.label, 64000,
                                         "paderborn", bearing_id, condition, bearing.origin, bearing.severity))
        counts = {"records": 40, "units": 2, "windows": 40,
                  "primary_records": 20, "primary_units": 1, "primary_windows": 20}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, paderborn=counts), \
                patch.object(paderborn, "enumerate_recordings", return_value=records) as enum, \
                patch.object(paderborn, "read_signal", return_value=np.arange(2731)):
            cache = features.build_cache("paderborn", directory)
            enum.assert_called_once_with(None, include_kb=True)
            with self.assertRaises(ValueError):
                features.feature_subset(cache)
            subset = features.feature_subset(cache, labeled=False)
            self.assertEqual(len(subset.X), 40)
            self.assertTrue(torch.all(subset.y == -1))
            primary = features.feature_subset(cache, unit_keys=[f"K001/{condition}"])
            self.assertEqual(len(primary.X), 20)
            self.assertTrue(torch.all(primary.y == 0))
            # 无标签入口不访问真值字段，避免纯迁移路径意外依赖标签。
            stripped = SimpleNamespace(X=cache.X, records=[{k: v for k, v in r.items() if k != "label"}
                                                           for r in cache.records])
            self.assertTrue(torch.all(features.feature_subset(stripped, labeled=False).y == -1))
            del cache, stripped

    def test_failed_build_and_partial_cache_not_success(self):
        records = cwru.enumerate_recordings()[:2]
        counts = {"records": 2, "units": 1, "windows": 2}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=records), \
                patch.object(cwru, "read_signal", side_effect=[np.arange(2048), ValueError("模拟坏信号")]):
            with self.assertRaises(ValueError):
                features.build_cache("cwru", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            np.save(Path(directory) / "cwru.npy", np.zeros((2, 1, 256), np.float32))
            with self.assertRaises(FileNotFoundError):
                features.load_cache("cwru", directory)
            with self.assertRaises(FileNotFoundError):
                features.build_cache("cwru", directory)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=[records[0], records[0]]), \
                patch.object(cwru, "read_signal", return_value=np.arange(2048)):
            with self.assertRaisesRegex(ValueError, "重复"):
                features.build_cache("cwru", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
