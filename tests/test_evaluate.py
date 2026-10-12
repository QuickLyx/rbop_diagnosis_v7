# -*- coding: utf-8 -*-
"""小样本评估测试：全部模型前向 eval/no_grad，不训练或重估 BN。"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from evaluate import (_evaluation_info, aggregate_records, aggregate_units, infer_records,
                      main, save_reports, score_predictions)
from features import FeatureCache, feature_subset


def pu_records(bearing="KI01", condition="N15_M01_F10", label=2, count=20):
    result = []
    for index in range(count):
        filename = f"{condition}_{bearing}_{index + 1}.mat"
        result.append({"record_key": f"{bearing}/{filename}::vibration_1",
                       "unit_key": f"{bearing}/{condition}", "path": filename,
                       "dataset": "paderborn", "channel": "vibration_1", "label": label,
                       "src_fs": 64000, "source": "paderborn", "bearing_id": bearing,
                       "condition": condition, "origin": "artificial", "severity": 1,
                       "start": index, "stop": index + 1})
    return result


def pu_predictions(**kwargs):
    return [{**row, "probabilities": [0.1, 0.1, 0.7, 0.1], "window_count": 1}
            for row in pu_records(**kwargs)]


def cw_predictions(label=1, source="12k_DE", filename="B007_0.mat"):
    unit = f"{source}/{filename}"
    return [{"record_key": f"{unit}::{channel}", "unit_key": unit, "path": filename,
             "dataset": "cwru", "channel": channel, "label": label, "source": source,
             "bearing_id": None, "condition": None, "origin": None, "severity": None,
             "src_fs": 12000, "start": index, "stop": index + 1,
             "probabilities": np.eye(4)[label].tolist(), "window_count": 1}
            for index, channel in enumerate(("DE", "FE"))]


def selection_payload():
    units = [f"B{index:03}/C" for index in range(116)]
    return {"protocol": "v7_stage2", "k_selected": 19, "all": units,
            "budgets": {f"K{count}": {"pool": units[:count], "test": units[count:]}
                        for count in (19, 38, 116)}}


def checkpoint_payload(arm="joint"):
    return {"protocol": "v7_stage2", "arm": arm, "fold": 0, "seed": 0,
            "checkpoint_kind": "last", "epoch": 30, "checkpoint_epoch": 30, "last_epoch": 30,
            "repr_config": {},
            "model": {"n_channels": 1, "n_classes": 4, "frontend": "spectral_v1"}}


class TinyModel(nn.Module):
    def __init__(self, tuple_output=False, corrupt_bn=False, invalid=False):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.head = nn.Linear(256, 4)
        self.tuple_output = tuple_output
        self.corrupt_bn = corrupt_bn
        self.invalid = invalid
        self.grad_modes = []
        self.eval()

    def forward(self, inputs):
        assert not self.training and not self.bn.training
        assert not torch.is_grad_enabled()
        self.grad_modes.append(torch.is_grad_enabled())
        logits = self.head(self.bn(inputs).flatten(1))
        if self.corrupt_bn:
            self.bn.running_mean.add_(1)
        if self.invalid:
            logits = logits * float("nan")
        return (logits, None) if self.tuple_output else logits


class EvaluateTests(unittest.TestCase):
    def test_record_then_unit_equal_weight_not_window_weight(self):
        records = pu_records()
        probabilities = []
        offset = 0
        for index, row in enumerate(records):
            count = 100 if index == 0 else 1
            row.update(start=offset, stop=offset + count)
            offset += count
            probabilities.extend([[1, 0, 0, 0] if index == 0 else [0, 0, 1, 0]] * count)
        rows = aggregate_records(records, np.asarray(probabilities))
        unit = aggregate_units(rows)[0]
        self.assertEqual(unit["record_count"], 20)
        self.assertEqual(unit["window_count"], 119)
        np.testing.assert_allclose(unit["probabilities"], [0.05, 0, 0.95, 0])
        self.assertEqual(np.asarray(probabilities).mean(axis=0).argmax(), 0)
        self.assertEqual(np.asarray(unit["probabilities"]).argmax(), 2)

    def test_record_mean_and_subset_relocated_offsets(self):
        records = pu_records(count=2)
        records[0].update(start=0, stop=2)
        records[1].update(start=2, stop=5)
        rows = aggregate_records(records, np.asarray([[1, 0, 0, 0], [0, 0, 1, 0]] + [[0, 0, 0, 1]] * 3))
        np.testing.assert_allclose(rows[0]["probabilities"], [0.5, 0, 0.5, 0])
        self.assertEqual(rows[1]["window_count"], 3)
        other = pu_records("KA01", label=3, count=1)[0]
        other.update(start=5, stop=7)
        cache = FeatureCache(np.zeros((7, 1, 256), dtype=np.float32), records + [other], "test", {})
        subset = feature_subset(cache, [other["unit_key"]], labeled=False)
        self.assertEqual((subset.records[0]["start"], subset.records[0]["stop"]), (0, 2))
        self.assertTrue(torch.equal(subset.y, torch.full((2,), -1)))
        predicted = infer_records(TinyModel(), subset)
        self.assertEqual(predicted[0]["window_count"], 2)

    def test_ka08_exact_19_exception(self):
        rows = pu_predictions(bearing="KA08", count=19, label=3)
        self.assertEqual(aggregate_units(rows)[0]["record_count"], 19)
        for kwargs in ({"bearing": "KA08", "count": 20},
                       {"bearing": "KA08", "condition": "N15_M07_F10", "count": 19},
                       {"count": 19}, {"count": 21}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                aggregate_units(pu_predictions(**kwargs))

    def test_unit_rejects_missing_duplicate_conflicting_label_channel_and_path(self):
        valid = pu_predictions()
        mutations = [lambda rows: rows.pop(), lambda rows: rows.append(deepcopy(rows[0])),
                     lambda rows: rows[1].update(label=3),
                     lambda rows: rows[1].update(channel="phase_current_1"),
                     lambda rows: rows[1].update(path=rows[0]["path"]),
                     lambda rows: rows[1].update(condition="wrong"),
                     lambda rows: rows[1].update(window_count=0),
                     lambda rows: rows[1].update(probabilities=[float("nan")] * 4),
                     lambda rows: rows[1].update(probabilities=[1, 1, 1, 1])]
        for mutation in mutations:
            rows = deepcopy(valid)
            mutation(rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate_units(rows)

    def test_records_reject_bad_intervals_duplicate_and_probability_shape(self):
        records = pu_records(count=2)
        for altered in ([{**records[0], "stop": 0}, records[1]],
                        [records[0], {**records[1], "start": 0}],
                        [records[0], {**records[1], "record_key": records[0]["record_key"]}],
                        [records[0]]):
            with self.assertRaises(ValueError):
                aggregate_records(altered, np.tile([1, 0, 0, 0], (2, 1)))
        for probabilities in (np.ones((2, 3)), np.full((2, 4), np.nan),
                              np.tile([-0.1, 0.1, 1, 0], (2, 1))):
            with self.assertRaises(ValueError):
                aggregate_records(records, probabilities)

    def test_e2_masks_probability_not_negative_logits_and_keeps_denominator(self):
        with torch.no_grad():
            logits = torch.tensor([[-8.0, -1.0, -3.0, -4.0]])
            probabilities = torch.softmax(logits, dim=1).numpy()[0].tolist()
            wrong = logits.clone()
            wrong[:, 1] = 0
            self.assertEqual(int(wrong.argmax()), 1)
        row = {**pu_predictions(count=1)[0], "probabilities": probabilities}
        scores = score_predictions([row], pool_units=[])
        self.assertEqual(scores["E1"]["n"], scores["E2"]["n"])
        self.assertEqual(scores["E1"]["accuracy"], 0)
        self.assertEqual(scores["E2"]["accuracy"], 1)
        self.assertEqual(scores["P_ball"], 1)
        self.assertEqual(scores["E1"]["confusion"][2][1], 1)
        self.assertEqual(scores["E2"]["confusion"][2][2], 1)

    def test_pu_macro_supported_classes_and_kb_exclusion(self):
        rows = []
        for truth, guesses in ((0, [0, 0, 0]), (2, [1]), (3, [3])):
            for guess in guesses:
                rows.append({**pu_predictions(count=1)[0], "label": truth,
                             "probabilities": np.eye(4)[guess].tolist()})
        kb = [{**pu_predictions(bearing=bearing, label=None, count=1)[0],
               "probabilities": np.eye(4)[guess].tolist()}
              for bearing, guess in (("KB23", 2), ("KB24", 2), ("KB27", 3))]
        scores = score_predictions(rows + kb, pool_units=["KI01/another_condition"])
        self.assertEqual(scores["n"], 5)
        self.assertEqual(scores["E1"]["averaged_classes"], ["Normal", "IR", "OR"])
        self.assertAlmostEqual(scores["E1"]["macro_recall"], 2 / 3)
        self.assertEqual(scores["E1"]["accuracy"], 0.8)
        self.assertEqual(scores["E1"]["macro_f1"], 2 / 3)
        self.assertEqual(scores["P_ball"], 0.2)
        self.assertEqual(scores["KB"]["n"], 3)
        self.assertEqual(scores["KB"]["prediction_counts"], [0, 0, 2, 1])
        self.assertEqual(scores["KB"]["by_bearing"]["KB27"]["official_dominant_reference"], 3)
        self.assertEqual(scores["groups"]["seen_unseen_bearing"]["seen"]["n"], 5)
        self.assertEqual(scores["groups"]["seen_unseen_bearing"]["unseen"]["n"], 0)
        self.assertEqual(scores["groups"]["condition"]["N15_M01_F10"]["n"], 5)
        self.assertEqual(scores["groups"]["origin"]["artificial"]["n"], 5)

    def test_cw_de_fe_and_four_class_source_metrics(self):
        rows = []
        for label in range(4):
            rows.extend(cw_predictions(label=label, filename=f"record_{label}.mat"))
        units = aggregate_units(rows)
        self.assertEqual(len(units), 4)
        self.assertEqual(units[0]["channels"], ["DE", "FE"])
        self.assertEqual(units[0]["record_count"], 2)
        scores = score_predictions(rows)
        self.assertEqual(scores["source"]["E1"]["support"], [2, 2, 2, 2])
        self.assertEqual(scores["source"]["E1"]["accuracy"], 1)
        self.assertEqual(scores["source"]["E1"]["macro_f1"], 1)
        self.assertEqual(scores["source_by_channel"]["DE"]["n"], 4)
        self.assertEqual(scores["source_by_source"]["12k_DE"]["n"], 8)
        self.assertEqual(scores["E2"]["n"], 8)
        for mutation in (lambda r: r.pop(), lambda r: r[1].update(channel="DE"),
                         lambda r: r[1].update(label=2), lambda r: r[1].update(source="12k_FE")):
            invalid = cw_predictions()
            mutation(invalid)
            with self.assertRaises(ValueError):
                aggregate_units(invalid)

    def test_inference_eval_no_grad_state_unchanged_and_tuple_logits(self):
        feature = SimpleNamespace(X=torch.randn(3, 1, 256, requires_grad=True), records=pu_records(count=3))
        for tuple_output in (False, True):
            model = TinyModel(tuple_output=tuple_output)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            rows = infer_records(model, feature, batch_size=2, device="cpu")
            self.assertEqual(len(rows), 3)
            self.assertEqual(model.grad_modes, [False, False])
            self.assertIsNone(feature.X.grad)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[key], value), key)
            for row in rows:
                self.assertAlmostEqual(sum(row["probabilities"]), 1, places=6)
        with self.assertRaisesRegex(RuntimeError, "BN"):
            infer_records(TinyModel(corrupt_bn=True), feature)
        with self.assertRaises(ValueError):
            infer_records(TinyModel(invalid=True), feature)
        with self.assertRaises(ValueError):
            infer_records(TinyModel(), feature, batch_size=0)

    def test_empty_and_kb_only_have_no_fake_scores(self):
        self.assertEqual(aggregate_records([], np.empty((0, 4))), [])
        self.assertEqual(aggregate_units([]), [])
        self.assertEqual(infer_records(TinyModel(), SimpleNamespace(
            X=torch.empty((0, 1, 256)), records=[])), [])
        for rows in ([], pu_predictions(bearing="KB23", label=None, count=1)):
            scores = score_predictions(rows, pool_units=[])
            self.assertEqual(scores["n"], 0)
            self.assertIsNone(scores["P_ball"])
            for name in ("E1", "E2"):
                self.assertEqual(scores[name]["n"], 0)
                self.assertIsNone(scores[name]["accuracy"])
                self.assertIsNone(scores[name]["macro_f1"])
                self.assertIsNone(scores[name]["macro_recall"])
                self.assertEqual(scores[name]["recall"], [None] * 4)
                self.assertEqual(scores[name]["support"], [0] * 4)

    def test_report_files_allow_rule_but_never_overwrite(self):
        rows = pu_predictions()
        units = aggregate_units(rows)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "rule.json").write_text("{}", encoding="utf-8")
            metrics = save_reports(rows, units, output, pool_units=[],
                                   eval_info={"arm": "phys_envelope", "evaluation_scope": "zero_shot_all_primary"})
            self.assertEqual(metrics["unit_level"]["n"], 1)
            self.assertIsNone(metrics["D3"]["passed"])
            self.assertEqual(json.loads((output / "eval_info.json").read_text(encoding="utf-8"))["supervised_pool_units"], [])
            self.assertEqual(json.loads((output / "metrics.json").read_text(encoding="utf-8")), metrics)
            with (output / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 20)
            self.assertIn("prob_Ball", csv_rows[0])
            original = (output / "metrics.json").read_bytes()
            with self.assertRaises(FileExistsError):
                save_reports([], [], output)
            self.assertEqual((output / "metrics.json").read_bytes(), original)
        for name in ("metrics.json", "predictions.csv", "unit_predictions.csv", "eval_info.json"):
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                (output / name).write_text("existing", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    save_reports([], [], output)
                self.assertEqual(list(output.iterdir()), [output / name])

    def test_regimes_budget_denominators_and_zero_actual_pool(self):
        selection = selection_payload()
        for arm, budget, pool_count, heldout_count, transductive in (
                ("joint", "K19", 0, 116, False), ("source_continue", "K38", 0, 116, False),
                ("adabn", "K19", 0, 116, True), ("mmd_marginal", "K19", 0, 116, True),
                ("ce_K19", "K19", 19, 97, False), ("ce_K38", "K38", 38, 78, False),
                ("ce_mmd", "K19", 19, 97, False), ("ce_K116", "K116", 116, 0, False)):
            with self.subTest(arm=arm):
                info, pool = _evaluation_info(checkpoint_payload(arm), selection, budget)
                self.assertEqual(len(pool), pool_count)
                self.assertEqual(len(info["evaluation_splits"]["natural_heldout"]), heldout_count)
                self.assertEqual(len(info["evaluation_splits"]["common_K19_test"]), 97)
                self.assertEqual(len(info["evaluation_splits"]["common_K38_test"]), 78)
                self.assertEqual(info["transductive"], transductive)
                self.assertEqual(info["unlabeled_target_label_usage"], 0)
                if arm == "ce_K116":
                    self.assertEqual(info["evaluation_scope"], "in_sample_only_no_heldout")
        with self.assertRaises(ValueError):
            _evaluation_info(checkpoint_payload("ce_K19"), selection, "K38")
        invalid = {**checkpoint_payload("adabn"), "target_label_usage": 19}
        with self.assertRaises(ValueError):
            _evaluation_info(invalid, selection, "K19")
        invalid = {**checkpoint_payload("adabn"), "target_train_units": selection["all"][:19]}
        with self.assertRaises(ValueError):
            _evaluation_info(invalid, selection, "K19")
        invalid = deepcopy(selection)
        invalid["budgets"]["K19"]["test"][0] = invalid["budgets"]["K19"]["pool"][0]
        with self.assertRaises(ValueError):
            _evaluation_info(checkpoint_payload(), invalid, "K19")

    def test_full_116_d3_and_no_heldout_reports(self):
        selection = selection_payload()
        info, pool = _evaluation_info(checkpoint_payload("ce_K116"), selection, "K116")
        rows = []
        for index, unit in enumerate(selection["all"]):
            row = pu_predictions(bearing=unit.split("/")[0], condition="C", label=(0, 2, 3)[index % 3], count=1)[0]
            row["probabilities"] = np.eye(4)[row["label"]].tolist()
            row.update(record_count=20, channels=["vibration_1"])
            rows.append(row)
        with tempfile.TemporaryDirectory() as folder:
            metrics = save_reports(rows, rows, folder, pool_units=pool, eval_info=info)
            self.assertTrue(metrics["D3"]["passed"])
            self.assertEqual(metrics["D3"]["interpretation"], "in_sample_only_no_heldout")
            self.assertEqual(metrics["splits"]["natural_heldout"]["unit_level"]["n"], 0)
            self.assertIsNone(metrics["splits"]["natural_heldout"]["unit_level"]["E1"]["accuracy"])
            self.assertEqual(metrics["splits"]["common_K19_test"]["unit_level"]["n"], 97)
            self.assertEqual(metrics["splits"]["common_K38_test"]["unit_level"]["n"], 78)
            self.assertEqual(metrics["splits"]["common_K19_test"]["supervised_pool_overlap_unit_count"], 97)
            self.assertEqual(metrics["splits"]["common_K19_test"]["interpretation"], "descriptive_contains_training_units")

    def test_cli_pu_unlabeled_forward_including_kb_with_mock_cache(self):
        from preprocessing import build_repr_config
        selection = selection_payload()
        records = []
        for unit in selection["all"]:
            records.extend(pu_records(bearing=unit.split("/")[0], condition="C"))
        records.extend(pu_records(bearing="KB23", condition="C", label=None))
        for index, row in enumerate(records):
            row.update(start=index, stop=index + 1)
        config = build_repr_config()
        cache = FeatureCache(np.zeros((len(records), 1, 256), np.float32), records, "test", config)
        model = TinyModel(tuple_output=True)
        checkpoint = checkpoint_payload("adabn")
        checkpoint.update(state_dict=model.state_dict(), repr_config=config, target_budget="K19",
                          target_train_units=selection["all"], target_label_usage=0)
        with tempfile.TemporaryDirectory() as folder:
            selection_path = Path(folder) / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            output = Path(folder) / "eval"
            argv = ["evaluate.py", "--checkpoint", str(Path(folder) / "trusted.pt"),
                    "--dataset", "paderborn", "--selection", str(selection_path), "--output", str(output)]
            with patch.object(sys, "argv", argv), patch("torch.load", return_value=checkpoint), \
                    patch("features.load_cache", return_value=cache), \
                    patch("model.MSRSABiLSTM", return_value=model), \
                    patch("features.feature_subset", wraps=feature_subset) as subset, \
                    patch("torch.cuda.is_available", return_value=False):
                main()
            subset.assert_called_once_with(cache, labeled=False)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            info = json.loads((output / "eval_info.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["unit_level"]["n"], 116)
            self.assertEqual(metrics["unit_level"]["KB"]["n"], 1)
            self.assertEqual(metrics["unit_level"]["total_count"], 117)
            self.assertTrue(info["transductive"])
            self.assertEqual(metrics["splits"]["natural_heldout"]["interpretation"],
                             "transductive_labels_heldout_signals_seen")
            self.assertEqual(info["supervised_pool_units"], [])

    def test_real_model_shim_tuple_forward_is_read_only(self):
        from model import MSRSABiLSTM
        model = MSRSABiLSTM().eval()
        before = {key: value.clone() for key, value in model.named_buffers()}
        feature = SimpleNamespace(X=torch.zeros((2, 1, 256)), records=pu_records(count=2))
        rows = infer_records(model, feature, batch_size=2, device="cpu")
        self.assertEqual(len(rows), 2)
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(before[key], value), key)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_cli_cw_cache_dicts_use_frozen_four_fold_and_source_metrics(self):
        from preprocessing import build_repr_config
        records = []
        for label in range(4):
            for repeat in range(2):
                records.extend(cw_predictions(label=label, filename=f"class{label}_{repeat}.mat"))
        for index, row in enumerate(records):
            row.update(start=index, stop=index + 1)
        config = build_repr_config()
        cache = FeatureCache(np.zeros((len(records), 1, 256), np.float32), records, "test", config)
        selection = selection_payload()
        for arm in ("joint", "adabn", "mmd_marginal"):
            transductive = arm in ("adabn", "mmd_marginal")
            model = TinyModel(tuple_output=True)
            checkpoint = checkpoint_payload(arm)
            checkpoint.update(state_dict=model.state_dict(), repr_config=config, target_budget="K19",
                              target_train_units=selection["all"] if transductive else [], target_label_usage=0)
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as folder:
                selection_path = Path(folder) / "selection.json"
                selection_path.write_text(json.dumps(selection), encoding="utf-8")
                output = Path(folder) / "eval"
                argv = ["evaluate.py", "--checkpoint", "trusted.pt", "--dataset", "cwru",
                        "--selection", str(selection_path), "--output", str(output)]
                with patch.object(sys, "argv", argv), patch("torch.load", return_value=checkpoint), \
                        patch("features.load_cache", return_value=cache), \
                        patch("model.MSRSABiLSTM", return_value=model), \
                        patch("torch.cuda.is_available", return_value=False):
                    main()
                metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                info = json.loads((output / "eval_info.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["unit_level"]["n"], 8)
                self.assertEqual(metrics["unit_level"]["E1"]["support"], [2, 2, 2, 2])
                self.assertEqual(metrics["splits"]["source_validation"]["unit_level"]["n"], 4)
                self.assertEqual(metrics["splits"]["source_validation"]["record_level"]["n"], 8)
                self.assertEqual(metrics["splits"]["source_all"]["interpretation"],
                                 "source_all_descriptive_contains_training_units")
                self.assertEqual(metrics["splits"]["source_validation"]["interpretation"], "source_validation")
                self.assertEqual(info["transductive"], transductive)
                self.assertEqual(info["adaptation_signal_unit_count"], 116 if transductive else 0)

    def test_cli_rejects_incomplete_or_missing_epochs_before_loading_model_or_cache(self):
        complete = {**checkpoint_payload(), "state_dict": {}}
        fields = ("epoch", "checkpoint_epoch", "last_epoch")
        invalid = [{**complete, **dict.fromkeys(fields, 12)}]
        for field in fields:
            invalid.extend({**complete, field: value} for value in (12, 29, 31, None, "30"))
            invalid.append({key: value for key, value in complete.items() if key != field})
        for checkpoint in invalid:
            with self.subTest(epochs={field: checkpoint.get(field) for field in fields}), \
                    tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "eval"
                with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", "trusted.pt", "--dataset", "paderborn",
                                                "--output", str(output)]), patch("torch.load", return_value=checkpoint), \
                        patch("features.load_cache") as load, patch("model.MSRSABiLSTM") as model:
                    with self.assertRaisesRegex(ValueError, "30"):
                        main()
                    load.assert_not_called()
                    model.assert_not_called()
                self.assertFalse(output.exists())

    def test_cli_rejects_v6_and_best_before_loading_model_or_cache(self):
        for checkpoint in ({"state_dict": {}}, {**checkpoint_payload(), "checkpoint_kind": "best", "state_dict": {}}):
            with tempfile.TemporaryDirectory() as folder:
                with patch.object(sys, "argv", ["evaluate.py", "--checkpoint", "trusted.pt", "--dataset", "paderborn",
                                                "--output", folder]), patch("torch.load", return_value=checkpoint), \
                        patch("features.load_cache") as load, patch("model.MSRSABiLSTM") as model:
                    with self.assertRaises(ValueError):
                        main()
                    load.assert_not_called()
                    model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
