# -*- coding: utf-8 -*-
"""选标只依赖信号、绝对预算嵌套、冻结划分完整性。"""
import unittest

import numpy as np

from selection import budget_splits, select_units


class SelectionTests(unittest.TestCase):
    def test_deterministic_medoids_and_nested_budgets(self):
        rng = np.random.default_rng(7)
        centers = rng.normal(size=(8, 256)) * 4
        matrix = np.concatenate([center + rng.normal(size=(14, 256)) * 0.2 for center in centers])
        keys = [f"单元{i:03d}" for i in range(len(matrix))]
        first = select_units(keys, matrix)
        second = select_units(keys, matrix)
        self.assertEqual(first["ranked_units"], second["ranked_units"])
        self.assertEqual(len(first["medoids"]), first["k_selected"])
        self.assertEqual(len(set(first["cluster_labels"][k] for k in first["medoids"])),
                         first["k_selected"])
        k = first["k_selected"]
        splits = budget_splits(first, [k, 2 * k, len(keys)])
        self.assertTrue(set(splits[f"K{k}"]["pool"]) <= set(splits[f"K{2*k}"]["pool"]))
        for split in splits.values():
            self.assertEqual(set(split["pool"]) | set(split["test"]), set(keys))
            self.assertFalse(set(split["pool"]) & set(split["test"]))
        self.assertFalse(splits[f"K{len(keys)}"]["test"])
        with self.assertRaises(ValueError):
            budget_splits(first, [k - 1])

    def test_invalid_features(self):
        matrix = np.zeros((116, 256))
        matrix[0, 0] = np.nan
        with self.assertRaises(ValueError):
            select_units([str(i) for i in range(116)], matrix)
        with self.assertRaises(ValueError):
            select_units(["重复"] * 116, np.zeros((116, 256)))


if __name__ == "__main__":
    unittest.main()
