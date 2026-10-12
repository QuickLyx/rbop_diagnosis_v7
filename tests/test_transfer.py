# -*- coding: utf-8 -*-
"""只检查采样、损失、eval 前向与 dry-run；不训练、不反传、不重估 BN。"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import ast
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _v6 import load_v6
from bearing_facts import PRIMARY_BEARINGS
from datasets import cwru
from features import CACHE_VERSION
import model
from preprocessing import build_repr_config
import train
import transfer

ROOT = Path(__file__).resolve().parents[1]


class TransferTests(unittest.TestCase):
    def test_v6_model_reexport_and_eval_forward(self):
        self.assertIs(model.MSRSABiLSTM, load_v6("model").MSRSABiLSTM)
        self.assertIs(model.classwise_mmd_loss, load_v6("model").classwise_mmd_loss)
        network = model.MSRSABiLSTM().eval()
        before = {key: value.clone() for key, value in network.state_dict().items()}
        with torch.no_grad():
            logits, attention, features = network(torch.zeros(2, 1, 256), return_feat=True)
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(features.shape, (2, 256))
        self.assertTrue(torch.allclose(attention.sum(1), torch.ones(2)))
        self.assertTrue(all(torch.equal(value, network.state_dict()[key]) for key, value in before.items()))

    def test_four_step_weights_and_sampling_switch(self):
        labels, channels, keys = [], [], []
        for channel in ("DE", "FE"):
            for label in range(4):
                for record in range(label + 1):
                    n = record + 1
                    labels.extend([label] * n)
                    channels.extend([channel] * n)
                    keys.extend([f"{channel}/{label}/{record}"] * n)
        feature = SimpleNamespace(X=torch.zeros(len(labels), 1, 256), y=torch.tensor(labels),
                                  channels=np.asarray(channels), record_keys=np.asarray(keys))
        weights = transfer.balanced_window_weights(feature)
        for channel in ("DE", "FE"):
            for label in range(4):
                mask = (feature.channels == channel) & (feature.y.numpy() == label)
                self.assertAlmostEqual(float(weights[mask].sum()), 1.0)
                for key in set(feature.record_keys[mask]):
                    self.assertAlmostEqual(float(weights[feature.record_keys == key].sum()), 1 / (label + 1))
        balanced, actual = transfer.make_weighted_loader(feature, 4, 12, 0)
        np.testing.assert_array_equal(actual.numpy(), weights)
        uniform, uniform_weights = transfer.make_weighted_loader(feature, 4, 12, 0, class_balance=False)
        repeat, _ = transfer.make_weighted_loader(feature, 4, 12, 0, class_balance=False)
        self.assertTrue(torch.equal(uniform_weights, torch.ones(len(labels), dtype=torch.double)))
        self.assertEqual(list(uniform.sampler), list(repeat.sampler))
        self.assertTrue(balanced.sampler.replacement and uniform.sampler.replacement)
        self.assertEqual(balanced.sampler.num_samples, 12)
        self.assertEqual(len(balanced), 3)
        self.assertEqual(len(uniform), 3)

    def test_target_supervision_balance_is_independent_of_source_switch(self):
        # 不执行训练，仅检查真实调用点，防止源开关再次传进目标采样器。
        tree = ast.parse(inspect.getsource(transfer.train_model))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "make_weighted_loader"]
        target_calls = [node for node in calls if any(k.arg == "labeled" for k in node.keywords)]
        self.assertEqual(len(target_calls), 1)
        balance = next(k.value for k in target_calls[0].keywords if k.arg == "class_balance")
        self.assertIsInstance(balance, ast.Constant)
        self.assertIs(balance.value, True)

    def test_unlabeled_loader_never_reads_target_labels_or_weights(self):
        class UnlabeledFeature:
            X = torch.zeros(3, 1, 256)

            @property
            def y(self):
                raise AssertionError("禁止读取目标真值")

        with patch.object(transfer, "balanced_window_weights", side_effect=AssertionError("禁止目标均衡")):
            loader, weights = transfer.make_weighted_loader(UnlabeledFeature(), 4, 12, 0, labeled=False)
            self.assertTrue(torch.equal(weights, torch.ones(3, dtype=torch.double)))
            self.assertEqual(len(loader), 3)
            sampled = list(loader.sampler)
            self.assertEqual(len(sampled), 12)
            self.assertLess(len(set(sampled)), 12)
            for _, labels in loader:
                self.assertTrue(torch.all(labels == -1))

    def test_fixed_step_budget(self):
        self.assertEqual(transfer.steps_per_epoch(9983, 64), 156)
        self.assertEqual(transfer.steps_per_epoch(128, 64), 2)
        self.assertEqual(transfer.steps_per_epoch(1, 64), 1)
        for s_ref, batch_size in ((0, 64), (10, 0), (-1, 64)):
            with self.assertRaises(ValueError):
                transfer.steps_per_epoch(s_ref, batch_size)

    def test_marginal_is_source_ce_plus_mmd_without_target_label_access(self):
        logits, labels = torch.zeros(4, 4), torch.tensor([0, 1, 2, 3])
        source, target = torch.zeros(4, 3), torch.ones(4, 3)
        losses = transfer.compute_losses("mmd_marginal", logits, labels, source,
                                         logits_target=object(), labels_target=object(), features_target=target)
        expected = F.cross_entropy(logits, labels) + 0.3 * model.mmd_loss(source, target)
        self.assertTrue(torch.allclose(losses["loss"], expected))
        self.assertEqual(float(losses["ce_target"]), 0)
        self.assertGreater(float(losses["align"]), 0)
        for arm in ("joint", "source_continue"):
            actual = transfer.compute_losses(arm, logits, labels)
            self.assertTrue(torch.equal(actual["loss"], F.cross_entropy(logits, labels)))

    def test_shared_normal_participates_and_ball_is_excluded(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        source, target = torch.zeros(8, 3), torch.zeros(8, 3)
        target[:2] = 1
        target[2:4] = 5
        logits = torch.zeros(8, 4)
        losses = transfer.compute_losses("ce_mmd", logits, labels, source, logits, labels, target)
        expected = model.mmd_loss(source[:2], target[:2]) / 3
        self.assertTrue(torch.allclose(losses["align"], expected))
        self.assertGreater(float(losses["align"]), 0)
        target[:2] = 0
        ball_only = transfer.compute_losses("ce_mmd", logits, labels, source, logits, labels, target)
        self.assertEqual(float(ball_only["align"]), 0)
        ce_only = transfer.compute_losses("ce_K19", logits, labels, source, logits, labels, target)
        self.assertEqual(float(ce_only["align"]), 0)
        self.assertTrue(torch.allclose(ce_only["loss"], 2 * F.cross_entropy(logits, labels)))
        skipped = model.classwise_mmd_loss(torch.zeros(1, 3), torch.tensor([0]),
                                           torch.ones(1, 3), torch.tensor([0]),
                                           shared_classes=transfer.SHARED_CLASSES)
        self.assertEqual(float(skipped), 0)

    def test_source_record_then_file_fusion_without_extra_forward_or_bn_changes(self):
        class EvalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(4)
                self.dropout = nn.Dropout(0.5)
                self.calls = 0

            def forward(self, x):
                self.assert_eval()
                self.calls += 1
                return self.dropout(self.bn(x)), None

            def assert_eval(self):
                if self.training or self.bn.training or self.dropout.training or torch.is_grad_enabled():
                    raise AssertionError("验证必须 eval/no_grad")

        # DE 有三窗、FE 有一窗；等记录融合判 Normal，按窗口融合则会误判 Ball。
        de = [0.35, 0.64, 0.005, 0.005]
        fe = [0.70, 0.29, 0.005, 0.005]
        probabilities = torch.tensor([de, de, de, fe])
        records = []
        for channel, start, stop in (("DE", 0, 3), ("FE", 3, 4)):
            records.append({"record_key": f"Normal/test.mat::{channel}", "unit_key": "Normal/test.mat",
                            "path": "test.mat", "dataset": "cwru", "channel": channel,
                            "source": "Normal", "label": 0, "start": start, "stop": stop})
        feature = SimpleNamespace(X=probabilities.log(), records=records)
        network = EvalModel()
        before = {key: value.clone() for key, value in network.state_dict().items()}
        metrics = transfer.source_validation(network, feature, batch_size=4)
        self.assertEqual(network.calls, 1)
        self.assertEqual(metrics["DE"]["accuracy"], 0)
        self.assertEqual(metrics["FE"]["accuracy"], 1)
        self.assertEqual(metrics["file_fused"]["accuracy"], 1)
        self.assertEqual(metrics["by_source"]["Normal"]["file_fused"]["support"], [1, 0, 0, 0])
        self.assertEqual(transfer.select_validation_score(metrics), (0.5, 0.5))
        self.assertTrue(all(torch.equal(value, network.state_dict()[key]) for key, value in before.items()))

    def test_plain_checkpoint_safe_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            transfer.save_checkpoint(path, {"weight": torch.zeros(1)},
                                     {"protocol": "v7_stage2", "epoch": 30, "values": [0, "last", None]})
            with patch.object(transfer.torch, "load", wraps=torch.load) as load:
                payload = transfer.load_checkpoint(path)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertEqual(payload["protocol"], "v7_stage2")
            self.assertTrue(torch.equal(payload["state_dict"]["weight"], torch.zeros(1)))


class DryRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recordings = cwru.enumerate_recordings()
        cls.selection = json.loads((ROOT / "outputs/selection/selection.json").read_text(encoding="utf-8"))
        cls.config = json.loads((ROOT / "configs/stage2.json").read_text(encoding="utf-8"))
        records, offset = [], 0
        for recording in cls.recordings:
            n = 2 if recording.channel == "DE" else 3
            records.append({**asdict(recording), "path": str(recording.path), "start": offset, "stop": offset + n})
            offset += n
        cls.source = SimpleNamespace(records=records, repr_config=build_repr_config(), version=CACHE_VERSION)
        # dry-run 的目标元数据刻意不含真值；只能按 bearing_id 和冻结清单取范围。
        target_records = [{"unit_key": key, "bearing_id": key.split("/")[0],
                           "start": index * 3, "stop": (index + 1) * 3}
                          for index, key in enumerate(cls.selection["all"])]
        cls.target = SimpleNamespace(records=target_records, repr_config=build_repr_config(), version=CACHE_VERSION)

    def _cache(self, dataset, cache_dir):
        return self.source if dataset == "cwru" else self.target

    def test_default_dry_run_does_not_construct_model_optimizer_or_training_features(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(train, "load_cache", side_effect=self._cache), \
                patch.object(train, "MSRSABiLSTM", side_effect=AssertionError("禁止构造模型")), \
                patch.object(train, "feature_subset", side_effect=AssertionError("禁止生成训练副本")), \
                patch.object(train, "seed_everything", side_effect=AssertionError("禁止训练准备")), \
                patch.object(train, "train_model", side_effect=AssertionError("禁止训练")), \
                patch.object(train, "load_checkpoint", side_effect=AssertionError("dry-run 不加载权重")), \
                patch.object(torch.optim, "Adam", side_effect=AssertionError("禁止优化器")), \
                redirect_stdout(io.StringIO()):
            out = Path(directory) / "joint"
            plan = train.main(["--arm", "joint", "--output", str(out), "--no-class-balance"])
            self.assertEqual(plan["mode"], "dry_run")
            self.assertFalse(plan["class_balance"])
            self.assertEqual(plan["target_sampling"], "none")
            self.assertEqual(plan["source_file_count"], 153)
            self.assertEqual(plan["s_ref"], 2 * plan["source_train_unit_count"])
            self.assertEqual(plan["planned_updates"], 30 * plan["steps_per_epoch"])
            self.assertEqual(list(out.iterdir()), [out / "dry_run.json"])
            self.assertEqual(json.loads((out / "dry_run.json").read_text(encoding="utf-8")), plan)
            with self.assertRaises(FileExistsError):
                train.main(["--arm", "joint", "--output", str(out)])

    def test_target_scopes_and_pending_joint(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(train, "load_cache", side_effect=self._cache), redirect_stdout(io.StringIO()):
            for arm, count, labeled in (("adabn", 116, False), ("mmd_marginal", 116, False),
                                        ("ce_K19", 19, True), ("ce_K38", 38, True),
                                        ("ce_K116", 116, True), ("ce_mmd", 19, True),
                                        ("source_continue", 0, False)):
                with self.subTest(arm=arm):
                    plan = train.main(["--arm", arm, "--output", str(Path(directory) / arm),
                                       "--source-checkpoint", str(Path(directory) / "missing.pt"),
                                       "--no-class-balance", "--dry-run"])
                    self.assertFalse(plan["class_balance"])
                    self.assertEqual(plan["status"], "dependency_pending")
                    self.assertEqual(plan["target_train_unit_count"], count)
                    self.assertEqual(plan["target_label_usage"], count if labeled else 0)
                    self.assertEqual(plan["split"]["target"]["labeled"], labeled)
                    if arm in ("adabn", "mmd_marginal"):
                        self.assertEqual(plan["evaluation_regime"], "transductive")
                        self.assertEqual(plan["target_sampling"], "uniform")
                        self.assertTrue(all(k.split("/")[0] in PRIMARY_BEARINGS
                                            for k in plan["split"]["target"]["train"]))
                    elif labeled:
                        self.assertEqual(plan["target_sampling"], "balanced")
                        self.assertEqual(set(plan["split"]["target"]["train"]),
                                         set(self.selection["budgets"][plan["split"]["target"]["budget"]]["pool"]))
                    self.assertEqual(plan["planned_updates"], 0 if arm == "adabn" else plan["planned_steps"])

    def test_checkpoint_rejects_v6_wrong_fold_seed_balance_and_best(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(train, "load_cache", side_effect=self._cache):
            args = train.build_parser().parse_args(["--arm", "ce_K19", "--output", directory])
            config, _, _, _, _ = train.prepare_plan(args)
        payload = {**config, "arm": "joint", "checkpoint_kind": "last", "epoch": 30,
                   "checkpoint_epoch": 30, "last_epoch": 30,
                   "steps": config["steps_per_epoch"] * 30, "updates": config["steps_per_epoch"] * 30}
        train.validate_source_checkpoint(payload, config)
        for key, value in (("protocol", "v6"), ("fold", 1), ("seed", 1),
                           ("class_balance", False), ("checkpoint_kind", "best"),
                           ("epoch", 29), ("source_file_count", 141), ("representation", "other")):
            with self.subTest(key=key):
                invalid = {**payload, key: value}
                with self.assertRaises(ValueError):
                    train.validate_source_checkpoint(invalid, config)

    def test_frozen_lists_and_cli_mode_conflicts(self):
        config, selection = deepcopy(self.config), deepcopy(self.selection)
        train._check_frozen(config, selection)
        selection["budgets"]["K19"]["pool"][0] = selection["budgets"]["K19"]["test"][0]
        with self.assertRaises(ValueError):
            train._check_frozen(config, selection)
        selection = deepcopy(self.selection)
        selection["budgets_status"] = "pending"
        with self.assertRaises(ValueError):
            train._check_frozen(config, selection)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                train.build_parser().parse_args(["--arm", "joint", "--execute", "--dry-run"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
