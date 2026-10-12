# -*- coding: utf-8 -*-
"""固定阶段二采样、损失、源验证和训练；调用训练函数须另获执行授权。"""
from copy import deepcopy
import csv
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluate import aggregate_records, aggregate_units
from model import classwise_mmd_loss, mmd_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHARED_CLASSES = (0, 2, 3)


def seed_everything(seed):
    """沿用 v6 的随机性设置。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_window_weights(feature):
    """照搬 v6 四步均衡：通道、类别、记录、记录内窗口。"""
    if len(feature.y) == 0:
        raise ValueError("空特征集不能计算采样权重")
    keys = np.asarray(feature.record_keys, dtype=object)
    channels = np.asarray(feature.channels, dtype=object)
    labels = feature.y.detach().cpu().numpy()
    if np.any(labels < 0):
        raise ValueError("无标签目标不能计算真值均衡权重")
    record_pairs = {(str(channel), str(key), int(label))
                    for channel, key, label in zip(channels, keys, labels)}
    record_counts, window_counts = {}, {}
    for channel, key, label in record_pairs:
        group = (str(channel), int(label))
        record_counts[group] = record_counts.get(group, 0) + 1
        record_key = (str(channel), str(key))
        window_counts[record_key] = int(np.sum((channels == channel) & (keys == key)))
    weights = np.empty(len(labels), dtype=np.float64)
    for index, (channel, key, label) in enumerate(zip(channels, keys, labels)):
        group = (str(channel), int(label))
        record_key = (str(channel), str(key))
        weights[index] = 1.0 / (record_counts[group] * window_counts[record_key])
    return weights


def make_weighted_loader(feature, batch_size, num_samples, seed,
                         class_balance=True, labeled=True):
    """固定样本数、有放回；关闭均衡或无标签时只使用均匀窗口权重。"""
    if len(feature.X) == 0 or batch_size <= 0 or num_samples < batch_size:
        raise ValueError("特征不能为空，采样数须不少于正数 batch_size")
    weights = torch.as_tensor(balanced_window_weights(feature), dtype=torch.double) \
        if labeled and class_balance else torch.ones(len(feature.X), dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights, num_samples=int(num_samples), replacement=True,
        generator=torch.Generator().manual_seed(int(seed)))
    # 无标签分支不访问 feature.y，防止采样器接触目标真值。
    labels = feature.y if labeled else torch.full((len(feature.X),), -1, dtype=torch.long)
    loader = DataLoader(TensorDataset(feature.X, labels), batch_size=batch_size,
                        sampler=sampler, drop_last=True)
    return loader, weights


def steps_per_epoch(s_ref, batch_size):
    """只以本折源训练 DE 窗口数确定所有臂的固定预算。"""
    if s_ref <= 0 or batch_size <= 0:
        raise ValueError("s_ref 和 batch_size 必须为正数")
    return int(math.ceil(s_ref / batch_size))


def compute_losses(arm, logits_source, labels_source, features_source=None,
                   logits_target=None, labels_target=None, features_target=None,
                   lambda_mmd=0.3, sigma=(1, 5, 10), estimator="biased"):
    """只计算损失；marginal 不读取目标标签，也不计算目标 CE。"""
    ce_source = F.cross_entropy(logits_source, labels_source)
    ce_target, align = ce_source.new_zeros(()), ce_source.new_zeros(())
    if arm == "mmd_marginal":
        align = mmd_loss(features_source, features_target, estimator=estimator, sigma=sigma)
    elif arm in ("ce_K19", "ce_K38", "ce_K116", "ce_mmd"):
        ce_target = F.cross_entropy(logits_target, labels_target)
        if arm == "ce_mmd":
            align = classwise_mmd_loss(
                features_source, labels_source, features_target, labels_target,
                shared_classes=SHARED_CLASSES, estimator=estimator, sigma=sigma)
    elif arm not in ("joint", "source_continue"):
        raise ValueError(f"没有梯度损失的臂：{arm}")
    loss = ce_source + ce_target + float(lambda_mmd) * align
    return {"loss": loss, "ce_source": ce_source, "ce_target": ce_target, "align": align}


def _class_metrics(true, pred):
    """源记录级四类指标，宏 F1 只对有真实样本的类别平均。"""
    confusion = np.zeros((4, 4), dtype=np.int64)
    for actual, guessed in zip(true, pred):
        confusion[actual, guessed] += 1
    support = confusion.sum(axis=1)
    f1 = []
    for label in range(4):
        denominator = support[label] + confusion[:, label].sum()
        f1.append(None if support[label] == 0 else
                  float(2 * confusion[label, label] / denominator))
    present = [value for value in f1 if value is not None]
    return {"accuracy": float(np.mean(np.asarray(true) == np.asarray(pred))) if true else 0.0,
            "macro_f1": float(np.mean(present)) if present else None,
            "f1": f1, "support": support.tolist(), "confusion_matrix": confusion.tolist()}


def source_validation(model, feature, batch_size=256):
    """仅源验证：记录均值后 DE/FE 文件等权融合；不增加前向。"""
    if not len(feature.X):
        raise ValueError("源验证特征为空")
    model.eval()
    device = next(model.parameters()).device
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(feature.X), batch_size):
            logits = model(feature.X[start:start + batch_size].to(device))[0]
            if not torch.isfinite(logits).all():
                raise ValueError("源验证输出包含非有限值")
            probabilities.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(probabilities).numpy()
    record_predictions = aggregate_records(feature.records, probs)
    unit_predictions = aggregate_units(record_predictions)
    groups, by_source = {}, {}
    for record in record_predictions:
        actual = int(record["label"])
        guessed = int(np.asarray(record["probabilities"]).argmax())
        channel, source = record["channel"], record["source"]
        groups.setdefault(channel, ([], []))
        by_source.setdefault(source, {}).setdefault(channel, ([], []))
        for true, pred in (groups[channel], by_source[source][channel]):
            true.append(actual)
            pred.append(guessed)
    if not {"DE", "FE"} <= groups.keys():
        raise ValueError("源验证必须同时包含 DE 和 FE")
    metrics = {channel: _class_metrics(*values) for channel, values in groups.items()}
    metrics["by_source"] = {
        source: {channel: _class_metrics(*values) for channel, values in channels.items()}
        for source, channels in by_source.items()}
    metrics["file_fused"] = _class_metrics(
        [row["label"] for row in unit_predictions],
        [int(np.asarray(row["probabilities"]).argmax()) for row in unit_predictions])
    for source, values in metrics["by_source"].items():
        units = [row for row in unit_predictions if row["source"] == source]
        values["file_fused"] = _class_metrics(
            [row["label"] for row in units],
            [int(np.asarray(row["probabilities"]).argmax()) for row in units])
    return metrics


def select_validation_score(metrics):
    return (float(np.mean([metrics[c]["macro_f1"] for c in ("DE", "FE")])),
            float(np.mean([metrics[c]["accuracy"] for c in ("DE", "FE")])))


def _set_bn_train_only(model):
    """仅供获授权后的 AdaBN：Dropout 关闭，只开启 BN 统计更新。"""
    model.eval()
    found = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            found = True
            if module.momentum != 0.1:
                raise ValueError("AdaBN 沿用 BN momentum=0.1，不重置统计")
            module.train()
    if not found:
        raise ValueError("模型没有 BN 层")


def _assert_only_bn_buffers_changed(before, model):
    allowed = {f"{name}.{buffer}" for name, module in model.named_modules()
               if isinstance(module, nn.BatchNorm1d)
               for buffer in ("running_mean", "running_var", "num_batches_tracked")}
    for key, value in model.state_dict().items():
        if key not in allowed and not torch.equal(before[key], value.detach().cpu()):
            raise AssertionError(f"AdaBN 改变了非 BN 缓冲区：{key}")


def save_checkpoint(path, state_dict, metadata):
    """训练目录由入口独占；每轮保存普通权重和元数据，不做断点续训。"""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({**metadata, "state_dict": state_dict}, temporary)
    temporary.replace(path)


def load_checkpoint(path, model=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("checkpoint 缺少 state_dict")
    if model is not None:
        model.load_state_dict(payload["state_dict"])
    return payload


def write_history(path, history):
    fields = list(dict.fromkeys(key for row in history for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def train_model(model, source_feature, target_feature, val_feature, arm,
                config, output, s_ref, seed):
    """获授权后才调用：固定步数跑满，各轮只记录源验证，正式返回 last。"""
    params = config["parameters"]
    batch_size, epochs = params["batch_size"], params["epochs"]
    steps = steps_per_epoch(s_ref, batch_size)
    balanced = config["class_balance"]
    source_loader = None
    if arm != "adabn":
        source_loader, _ = make_weighted_loader(
            source_feature, batch_size, steps * batch_size, seed, balanced)
    target_loader = None
    if target_feature is not None:
        target_loader, _ = make_weighted_loader(
            target_feature, batch_size, steps * batch_size, seed + 1, class_balance=True,
            labeled=arm not in ("adabn", "mmd_marginal"))
    device = next(model.parameters()).device
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()} \
        if arm == "adabn" else None
    optimizer = None if arm == "adabn" else torch.optim.Adam(
        model.parameters(), lr=params["joint_lr"] if arm == "joint" else params["adapt_lr"])
    best, history = None, []
    for epoch in range(1, epochs + 1):
        totals = {"ce_source": 0.0, "ce_target": 0.0, "align": 0.0}
        target_iterator = iter(target_loader) if target_loader is not None else None
        if arm == "adabn":
            _set_bn_train_only(model)
            with torch.no_grad():
                for xb, _ in target_iterator:
                    model(xb.to(device))
            _assert_only_bn_buffers_changed(before, model)
        else:
            model.train()
            for xb_s, yb_s in source_loader:
                logits_s, _, feat_s = model(xb_s.to(device), return_feat=True)
                logits_t, feat_t, yb_t = None, None, None
                if target_iterator is not None:
                    xb_t, yb_t = next(target_iterator)
                    logits_t, _, feat_t = model(xb_t.to(device), return_feat=True)
                    # marginal 不移动、不使用目标占位标签。
                    yb_t = yb_t.to(device) if arm != "mmd_marginal" else None
                losses = compute_losses(
                    arm, logits_s, yb_s.to(device), feat_s, logits_t, yb_t, feat_t,
                    params["lambda_mmd"], tuple(params["mmd_sigma"]), params["mmd_estimator"])
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"第 {epoch} 轮损失非有限，停止更新")
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()
                for key in totals:
                    totals[key] += float(losses[key].detach())
        metrics = source_validation(model, val_feature)
        score, accuracy = select_validation_score(metrics)
        row = {"epoch": epoch, **{key: value / steps for key, value in totals.items()},
               "val_macro_f1_mean": score, "val_accuracy_mean": accuracy,
               "updates": 0 if arm == "adabn" else steps * epoch,
               "target_forward_batches_cum": steps * epoch if target_loader is not None else 0}
        for channel in ("DE", "FE", "file_fused"):
            row[f"val_macro_f1_{channel}"] = metrics[channel]["macro_f1"]
            row[f"val_accuracy_{channel}"] = metrics[channel]["accuracy"]
        for source, channels in metrics["by_source"].items():
            for channel, values in channels.items():
                row[f"val_macro_f1_{source}_{channel}"] = values["macro_f1"]
                row[f"val_accuracy_{source}_{channel}"] = values["accuracy"]
        history.append(row)
        last = {"epoch": epoch, "score": score, "accuracy": accuracy,
                "state_dict": {key: value.detach().cpu().clone()
                               for key, value in model.state_dict().items()}, "metrics": metrics}
        if best is None or (score, accuracy) > (best["score"], best["accuracy"]):
            best = deepcopy(last)
        for kind, snapshot in (("best", best), ("last", last)):
            checkpoint_epoch = snapshot["epoch"]
            metadata = {**config, "checkpoint_kind": kind, "epoch": checkpoint_epoch,
                        "checkpoint_epoch": checkpoint_epoch, "last_epoch": epoch,
                        "best_epoch": best["epoch"], "steps_per_epoch": steps,
                        "steps": steps * checkpoint_epoch,
                        "updates": 0 if arm == "adabn" else steps * checkpoint_epoch,
                        "target_forwards": steps * checkpoint_epoch if target_loader is not None else 0,
                        "validation_metrics": snapshot["metrics"],
                        "validation_score": snapshot["score"]}
            save_checkpoint(Path(output) / f"{kind}.pt", snapshot["state_dict"], metadata)
        write_history(Path(output) / "history.csv", history)
        print(f"{arm} 第 {epoch}/{epochs} 轮：源验证 F1={score:.4f}，"
              f"best={best['epoch']}，正式权重=last", flush=True)
    model.eval()
    return {"model": model, "best": best, "last": last, "history": history,
            "steps_per_epoch": steps, "updates": 0 if arm == "adabn" else steps * epochs}
