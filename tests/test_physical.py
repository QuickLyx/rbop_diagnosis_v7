# -*- coding: utf-8 -*-
"""零训练物理规则与沿用的包络谐波比测试。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from physical_baseline import _envelope_functions, classify_ratios, recording_prediction


class PhysicalTests(unittest.TestCase):
    def test_frozen_boundaries(self):
        self.assertEqual(classify_ratios(3, 2), 0)
        self.assertEqual(classify_ratios(3.0001, 2), 3)
        self.assertEqual(classify_ratios(2, 3.0001), 2)
        self.assertEqual(classify_ratios(0, 0), 0)
        with self.assertRaises(ValueError):
            classify_ratios(float("nan"), 1)
        with self.assertRaises(ValueError):
            classify_ratios(-1, 1)

    def test_synthetic_harmonics_and_gain(self):
        spectrum, ratio = _envelope_functions()
        fs = 12000
        time = np.arange(fs * 2) / fs
        rng = np.random.default_rng(0)
        signal = (1 + 0.8 * np.cos(2 * np.pi * 75 * time)) * np.cos(2 * np.pi * 3000 * time)
        signal += 0.03 * rng.normal(size=len(time))
        freq, amplitude = spectrum(signal)
        outer = ratio(freq, amplitude, 75)
        inner = ratio(freq, amplitude, 125)
        self.assertGreater(outer, 1.5 * inner)
        freq2, amplitude2 = spectrum(signal * 10)
        self.assertAlmostEqual(outer, ratio(freq2, amplitude2, 75), places=5)

    def test_wrong_channel(self):
        with self.assertRaises(ValueError):
            recording_prediction(SimpleNamespace(dataset="paderborn", channel="phase_current_1"))

    def test_sys_path_restored(self):
        """脚本导入链临时插入的 v6 搜索路径在返回前已恢复。"""
        before = sys.path[:]
        _envelope_functions()
        self.assertEqual(sys.path, before)

    def test_preprocessing_stays_v7_shim(self):
        """导入链结束后 sys.modules 里的 preprocessing 仍是 v7 的 load_v6 shim。"""
        _envelope_functions()
        import preprocessing
        self.assertTrue(hasattr(preprocessing, "extract_logspec"))
        self.assertEqual(preprocessing.__name__, "preprocessing")

    def test_label_none_end_to_end(self):
        """假记录端到端：label=None 正常出四类 one-hot，不读真值。"""
        from datasets.recording import Recording
        from scipy.io import savemat
        fs = 12000
        time = np.arange(fs * 2) / fs
        rng = np.random.default_rng(1)
        signal = (1 + 0.9 * np.cos(2 * np.pi * 125 * time)) * np.cos(2 * np.pi * 3000 * time)
        signal += 0.02 * rng.normal(size=len(time))
        stem = "N15_M01_F10_KA04_1"
        path = Path("tests") / (stem + ".mat")
        savemat(path, {stem: {"Y": [{"Name": "vibration_1", "Data": signal}]}})
        record = Recording(
            record_key=f"KA04/{stem}.mat::vibration_1",
            unit_key="KA04/N15_M01_F10", path=path,
            dataset="paderborn", channel="vibration_1", label=None,
            src_fs=12000, bearing_id="KA04", condition="N15_M01_F10",
            origin="artificial", severity=1)
        try:
            row = recording_prediction(record)
        finally:
            path.unlink()
        self.assertIsNone(row["label"])
        self.assertEqual(len(row["probabilities"]), 4)
        self.assertEqual(sum(row["probabilities"]), 1.0)
        self.assertIn(row["physical_prediction"], (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
