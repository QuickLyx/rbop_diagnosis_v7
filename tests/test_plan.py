# -*- coding: utf-8 -*-
"""冻结矩阵与实际数据的无训练集成检查。"""
import json
from pathlib import Path
import unittest

from datasets.cwru import enumerate_recordings, split_source_units

ROOT = Path(__file__).resolve().parents[1]


class FrozenPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/stage2.json").read_text(encoding="utf-8"))
        cls.selection = json.loads((ROOT / "outputs/selection/selection.json").read_text(encoding="utf-8"))

    def test_matrix_and_transductive_scope(self):
        self.assertEqual(self.config["status"], "frozen")
        self.assertEqual(self.selection["budgets_status"], "frozen")
        self.assertEqual(self.config["budgets"], [19, 38, 116])
        self.assertEqual(self.config["unlabeled_target_scope"], "all_primary")
        self.assertEqual(self.config["unlabeled_target_label_usage"], 0)
        count = len(self.config["arms"]) * len(self.config["folds"]) * len(self.config["seeds"])
        self.assertEqual(count, self.config["planned_model_runs"])
        self.assertEqual(count, 96)
        self.assertEqual(self.config["criterion"]["scope"], "all_primary")
        self.assertEqual(self.config["criterion"]["unit_count"], 116)
        for left, right in self.config["paired_chain"]:
            self.assertIn(left, self.config["arms"])
            self.assertIn(right, self.config["arms"])

    def test_nested_budget_and_medoids(self):
        all_units = set(self.selection["all"])
        self.assertEqual(len(all_units), 116)
        self.assertEqual(set(self.selection["medoids"]), set(self.selection["budgets"]["K19"]["pool"]))
        previous = set()
        for k in self.config["budgets"]:
            split = self.selection["budgets"][f"K{k}"]
            pool, test = set(split["pool"]), set(split["test"])
            self.assertEqual(len(pool), k)
            self.assertFalse(pool & test)
            self.assertEqual(pool | test, all_units)
            self.assertTrue(previous <= pool)
            previous = pool
        self.assertFalse(self.selection["budgets"]["K116"]["test"])

    def test_source_folds_cover_all_units_once(self):
        records = enumerate_recordings()
        units = {record.unit_key for record in records}
        folds = split_source_units(records, split_seed=self.config["split_seed"])
        self.assertEqual(len(records), 306)
        self.assertEqual(len(units), 153)
        self.assertEqual([len(fold["val"]) for fold in folds], [39, 38, 38, 38])
        self.assertEqual(set(key for fold in folds for key in fold["val"]), units)
        for fold in folds:
            self.assertFalse(set(fold["train"]) & set(fold["val"]))
            self.assertEqual(set(fold["train"]) | set(fold["val"]), units)


if __name__ == "__main__":
    unittest.main()
