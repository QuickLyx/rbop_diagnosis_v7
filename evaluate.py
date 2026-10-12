# -*- coding: utf-8 -*-
"""冻结阶段二评估：只读前向、记录等权聚合和普通 JSON/CSV 报告。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["Normal", "Ball", "IR", "OR"]
UNIT_FIELDS = ("unit_key", "dataset", "label", "source", "bearing_id", "condition",
               "origin", "severity", "src_fs")


def _probabilities(values):
    probabilities = np.asarray(values, dtype=np.float64)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 4
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("预测必须是有限、归一化的四类概率")
    return probabilities


def aggregate_records(records, window_probs):
    """同记录窗口概率取均值；区间必须恰好覆盖当前子集的全部窗口。"""
    if not records and len(window_probs) == 0:
        return []
    probabilities = _probabilities(window_probs)
    result, seen, offset = [], set(), 0
    for record in records:
        key = record["record_key"]
        start, stop = record["start"], record["stop"]
        if key in seen:
            raise ValueError(f"记录重复：{key}")
        if (type(start) is not int or type(stop) is not int or start != offset
                or stop <= start or stop > len(probabilities)):
            raise ValueError(f"记录窗口区间缺失、重叠或越界：{key}")
        seen.add(key)
        result.append({**record, "probabilities": probabilities[start:stop].mean(axis=0).tolist(),
                       "window_count": stop - start})
        offset = stop
    if offset != len(probabilities):
        raise ValueError("记录区间未覆盖全部窗口概率")
    return result


def aggregate_units(record_predictions):
    """PU 按 20 条记录（KA08 特例 19 条），CW 按 DE/FE 两记录等权聚合。"""
    groups, seen = {}, set()
    for record in record_predictions:
        key = record["record_key"]
        if key in seen:
            raise ValueError(f"记录重复：{key}")
        seen.add(key)
        label = record["label"]
        if label is not None and (type(label) is not int or label not in range(4)):
            raise ValueError(f"记录标签无效：{key}")
        if type(record["window_count"]) is not int or record["window_count"] <= 0:
            raise ValueError(f"记录窗口数无效：{key}")
        groups.setdefault(record["unit_key"], []).append(record)
    result = []
    for unit, rows in groups.items():
        first = rows[0]
        if any(any(row.get(field) != first.get(field) for field in UNIT_FIELDS) for row in rows):
            raise ValueError(f"单元内标签或元数据不一致：{unit}")
        channels = Counter(row["channel"] for row in rows)
        if first["dataset"] == "paderborn":
            expected = 19 if unit == "KA08/N15_M01_F10" else 20
            if (unit != f"{first['bearing_id']}/{first['condition']}"
                    or channels != Counter({"vibration_1": expected})
                    or len({str(row['path']) for row in rows}) != expected):
                raise ValueError(f"PU 单元记录缺失、重复或通道错误：{unit}")
        elif first["dataset"] == "cwru":
            if (channels != Counter({"DE": 1, "FE": 1}) or first["label"] is None
                    or any(unit != row['source'] + '/' + row['path'].replace('\\', '/').rsplit('/', 1)[-1]
                           or row["record_key"] != f"{unit}::{row['channel']}" for row in rows)):
                raise ValueError(f"CWRU 单元记录缺失或通道错误：{unit}")
        else:
            raise ValueError(f"未知数据集：{first['dataset']}")
        probabilities = _probabilities([row["probabilities"] for row in rows])
        result.append({**{field: first.get(field) for field in UNIT_FIELDS},
                       "channels": sorted(channels), "probabilities": probabilities.mean(axis=0).tolist(),
                       "record_count": len(rows), "window_count": sum(row["window_count"] for row in rows)})
    return result


def _classification_metrics(labels, predicted):
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, guess in zip(labels, predicted):
        confusion[truth, guess] += 1
    support = confusion.sum(axis=1)
    recall, f1 = [], []
    for index in range(4):
        tp = int(confusion[index, index])
        recall.append(tp / int(support[index]) if support[index] else None)
        denominator = int(support[index] + confusion[:, index].sum())
        f1.append(2 * tp / denominator if support[index] else None)
    present = [index for index in range(4) if support[index]]
    macro_recall = float(np.mean([recall[index] for index in present])) if present else None
    return {"n": len(labels), "accuracy": float(np.trace(confusion) / len(labels)) if len(labels) else None,
            "macro_f1": float(np.mean([f1[index] for index in present])) if present else None,
            "macro_recall": macro_recall, "recall": recall,
            "confusion": confusion.tolist(), "support": support.tolist(),
            "classes": CLASS_NAMES, "averaged_classes": [CLASS_NAMES[index] for index in present]}


def _score(rows):
    labeled = [row for row in rows if row["label"] is not None]
    if any(type(row["label"]) is not int or row["label"] not in range(4) for row in labeled):
        raise ValueError("评分标签必须是 0/1/2/3 或 None")
    probabilities = (_probabilities([row["probabilities"] for row in labeled])
                     if labeled else np.empty((0, 4)))
    labels = [row["label"] for row in labeled]
    e1 = probabilities.argmax(axis=1) if labeled else np.array([], dtype=int)
    masked = probabilities.copy()
    masked[:, 1] = 0
    e2 = masked.argmax(axis=1) if labeled else np.array([], dtype=int)
    return {"n": len(labeled), "total_count": len(rows), "unlabeled_count": len(rows) - len(labeled),
            "E1": _classification_metrics(labels, e1), "E2": _classification_metrics(labels, e2),
            "P_ball": float(np.mean(e1 == 1)) if labeled else None,
            "denominator": "有真实标签的全部样本；E1/E2/P_ball 使用相同分母"}


def score_predictions(predictions, pool_units=None):
    """真实支持类别宏平均；PU 按工况、来源及监督池轴承是否见过并报。"""
    result = _score(predictions)
    primary = [row for row in predictions if row.get("dataset") == "paderborn" and row["label"] is not None]
    source = [row for row in predictions if row.get("dataset") == "cwru"]
    result["source"] = _score(source)
    result["source_by_source"] = {
        name: _score([row for row in source if row["source"] == name])
        for name in sorted({row["source"] for row in source})}
    result["source_by_channel"] = {
        name: _score([row for row in source if row.get("channel") == name])
        for name in sorted({row["channel"] for row in source if "channel" in row})}
    result["groups"] = {}
    for field in ("condition", "origin"):
        values = sorted({row.get(field) for row in primary}, key=str)
        result["groups"][field] = {str(value): _score([row for row in primary if row.get(field) == value])
                                    for value in values}
    if pool_units is None:
        result["groups"]["seen_unseen_bearing"] = {"unknown": _score(primary)}
    else:
        seen = {unit.split("/", 1)[0] for unit in pool_units}
        result["groups"]["seen_unseen_bearing"] = {
            name: _score([row for row in primary if (row["bearing_id"] in seen) == is_seen])
            for name, is_seen in (("seen", True), ("unseen", False))}
    kb = [row for row in predictions if row["label"] is None]
    kb_report = {"n": len(kb), "prediction_counts": [0] * 4, "prediction_fractions": None,
                 "by_bearing": {}, "note": "KB 无单类真值，不入 E1/E2/P_ball 分母；官方主导指派仅作参考"}
    references = {"KB23": 2, "KB24": 2, "KB27": 3}
    for bearing in sorted({row.get("bearing_id") for row in kb}, key=str):
        rows = [row for row in kb if row.get("bearing_id") == bearing]
        predicted = _probabilities([row["probabilities"] for row in rows]).argmax(axis=1)
        counts = np.bincount(predicted, minlength=4)
        reference = references.get(bearing)
        kb_report["by_bearing"][str(bearing)] = {
            "n": len(rows), "prediction_counts": counts.tolist(),
            "prediction_fractions": (counts / len(rows)).tolist(),
            "official_dominant_reference": reference,
            "reference_agreement": float(np.mean(predicted == reference)) if reference is not None else None}
        kb_report["prediction_counts"] = (np.asarray(kb_report["prediction_counts"]) + counts).tolist()
    if kb:
        kb_report["prediction_fractions"] = (np.asarray(kb_report["prediction_counts"]) / len(kb)).tolist()
    result["KB"] = kb_report
    return result


def infer_records(model, feature, batch_size=256, device=None):
    """只读前向：显式 eval/no_grad，检查所有 BN buffers 未改变。"""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if device is None:
        device = next(model.parameters(), torch.empty(0)).device
    model.to(device)
    model.eval()
    bn_before = {f"{name}.{key}": value.detach().clone()
                 for name, module in model.named_modules()
                 if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                 for key, value in module.named_buffers()}
    output = []
    try:
        with torch.no_grad():
            for start in range(0, len(feature.X), batch_size):
                inputs = torch.as_tensor(feature.X[start:start + batch_size]).to(device)
                logits = model(inputs)
                # 复用网络默认返回 (logits, attention)，小模型也可直接返回 logits。
                if isinstance(logits, tuple):
                    logits = logits[0]
                if (not isinstance(logits, torch.Tensor) or logits.shape != (len(inputs), 4)
                        or not torch.isfinite(logits).all()):
                    raise ValueError("模型必须输出有限的 (batch,4) logits")
                output.append(torch.softmax(logits, dim=1).cpu().numpy())
    finally:
        bn_after = {f"{name}.{key}": value for name, module in model.named_modules()
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                    for key, value in module.named_buffers()}
        if bn_before.keys() != bn_after.keys() or any(
                not torch.equal(value, bn_after[key]) for key, value in bn_before.items()):
            raise RuntimeError("测试前向改变了 BN buffers，拒绝报告")
    probabilities = np.concatenate(output) if output else np.empty((0, 4))
    return aggregate_records(feature.records, probabilities)


def _write_predictions(rows, path, unit=False):
    fields = (["unit_key"] if unit else ["record_key", "unit_key", "path", "channel", "start", "stop"])
    fields += ["dataset", "source", "bearing_id", "condition", "origin", "severity", "src_fs", "label",
               "prob_Normal", "prob_Ball", "prob_IR", "prob_OR", "prediction_E1", "prediction_E2", "window_count"]
    if unit:
        fields += ["record_count", "channels"]
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            probabilities = _probabilities([row["probabilities"]])[0]
            masked = probabilities.copy()
            masked[1] = 0
            values = {**row, **dict(zip(("prob_Normal", "prob_Ball", "prob_IR", "prob_OR"),
                                      probabilities.tolist())),
                      "prediction_E1": int(probabilities.argmax()), "prediction_E2": int(masked.argmax())}
            if unit:
                values["channels"] = ";".join(row.get("channels", []))
            writer.writerow(values)


def save_reports(record_predictions, unit_predictions, output, pool_units=None, eval_info=None):
    """仅创建四份报告；允许目录已有 rule.json，不覆盖已有同名产物。"""
    output = Path(output)
    names = ("metrics.json", "predictions.csv", "unit_predictions.csv", "eval_info.json")
    for name in names:
        if (output / name).exists():
            raise FileExistsError(f"报告已存在，拒绝覆盖：{output / name}")
    info = dict(eval_info or {})
    info.setdefault("protocol", "v7_stage2")
    info["inference"] = "eval/no_grad；无训练、梯度更新或 BN 重估"
    info["supervised_pool_units"] = None if pool_units is None else sorted(pool_units)
    metrics = {"record_level": score_predictions(record_predictions, pool_units),
               "unit_level": score_predictions(unit_predictions, pool_units)}
    primary = [row for row in unit_predictions if row.get("dataset") == "paderborn" and row["label"] is not None]
    primary_metrics = _score(primary)
    e1, e2 = primary_metrics["E1"]["macro_recall"], primary_metrics["E2"]["macro_recall"]
    full = len(primary) == 116 and len({row["unit_key"] for row in primary}) == 116
    metrics["D3"] = {"n": len(primary), "required_n": 116, "E1": e1, "E2": e2,
                     "E1_min": 0.8, "E2_min": 0.9,
                     "passed": bool(e1 >= 0.8 and e2 >= 0.9) if full else None,
                     "literature_anchor_E2": 0.75,
                     "interpretation": info.get("evaluation_scope", "监督池未知，不能判为泛化证据"),
                     "note": "全 116 单元真实支持类宏平均；含监督池时仅描述，K116 仅 in_sample"}
    for name, units in info.get("evaluation_splits", {}).items():
        wanted = set(units)
        overlap = len(wanted & set(pool_units or []))
        metrics.setdefault("splits", {})[name] = {
            "requested_unit_count": len(wanted), "supervised_pool_overlap_unit_count": overlap,
            "interpretation": "no_heldout" if not wanted else (
                "source_validation" if name == "source_validation" else (
                    info.get("evaluation_scope", "监督池未知") if name in ("all_primary", "source_all") else (
                        "descriptive_contains_training_units" if overlap else (
                            "transductive_labels_heldout_signals_seen" if info.get("transductive") else
                            info.get("heldout_scope", "监督池未知"))))),
            "unit_level": score_predictions([row for row in unit_predictions if row["unit_key"] in wanted], pool_units),
            "record_level": score_predictions([row for row in record_predictions if row["unit_key"] in wanted], pool_units)}
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", metrics), ("eval_info.json", info)):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    _write_predictions(record_predictions, output / "predictions.csv")
    _write_predictions(unit_predictions, output / "unit_predictions.csv", unit=True)
    return metrics


def _evaluation_info(checkpoint, selection, budget):
    """普通字符串分支明确区分监督留出、传导适配和零目标使用。"""
    if selection.get("protocol") != "v7_stage2" or selection.get("k_selected") != 19:
        raise ValueError("只接受正式 v7 的 K19 冻结选标")
    all_units = selection["all"]
    if len(all_units) != 116 or len(set(all_units)) != 116:
        raise ValueError("冻结主口径必须是 116 个不同单元")
    splits = selection["budgets"]
    for name, count in (("K19", 19), ("K38", 38), ("K116", 116)):
        pool, test = splits[name]["pool"], splits[name]["test"]
        if (len(pool) != count or len(test) != 116 - count
                or len(set(pool)) != len(pool) or len(set(test)) != len(test)
                or set(pool) & set(test) or set(pool) | set(test) != set(all_units)):
            raise ValueError(f"冻结池和测试列表不完整或重叠：{name}")
    if not set(splits["K19"]["pool"]) <= set(splits["K38"]["pool"]) <= set(splits["K116"]["pool"]):
        raise ValueError("冻结监督池必须嵌套")
    arm = checkpoint["arm"]
    if arm in ("ce_K19", "ce_K38", "ce_K116", "ce_mmd"):
        expected = "K19" if arm == "ce_mmd" else arm[3:]
        if budget != expected:
            raise ValueError(f"预算与监督臂不一致：{arm} / {budget}")
        pool = splits[budget]["pool"]
        scope = "in_sample_only_no_heldout" if budget == "K116" else "descriptive_contains_training_units"
        heldout = splits[budget]["test"]
    elif arm in ("adabn", "mmd_marginal"):
        pool, heldout, scope = [], all_units, "transductive_test_signals_seen_labels_unused"
    elif arm in ("joint", "source_continue", "phys_envelope"):
        pool, heldout, scope = [], all_units, "zero_shot_all_primary"
    else:
        raise ValueError(f"未知评估臂：{arm}")
    if "supervised_pool_units" in checkpoint and set(checkpoint["supervised_pool_units"]) != set(pool):
        raise ValueError("checkpoint 监督池与冻结口径不一致")
    if "target_label_usage" in checkpoint and checkpoint["target_label_usage"] != len(pool):
        raise ValueError("checkpoint 目标标签用量与监督池不一致")
    if "target_train_units" in checkpoint:
        expected_train = all_units if arm in ("adabn", "mmd_marginal") else pool
        if set(checkpoint["target_train_units"]) != set(expected_train):
            raise ValueError("checkpoint 目标信号参与范围与冻结口径不一致")
    info = {"protocol": "v7_stage2", "arm": arm, "fold": checkpoint["fold"],
            "seed": checkpoint["seed"], "budget": budget, "checkpoint_kind": checkpoint["checkpoint_kind"],
            "repr_config": checkpoint["repr_config"], "model": checkpoint["model"],
            "evaluation_scope": scope, "target_label_unit_count": len(pool),
            "unlabeled_target_label_usage": 0,
            "transductive": arm in ("adabn", "mmd_marginal"),
            "adaptation_signal_unit_count": 116 if arm in ("adabn", "mmd_marginal") else len(pool),
            "heldout_scope": "no_heldout" if not heldout else (
                "transductive_labels_heldout_signals_seen" if arm in ("adabn", "mmd_marginal") else (
                    "zero_shot" if not pool else "supervised_pool_outside")),
            "evaluation_splits": {"all_primary": all_units, "natural_heldout": heldout,
                                  "common_K19_test": splits["K19"]["test"],
                                  "common_K38_test": splits["K38"]["test"]},
            "pairing_note": "共同测试列表只用于适用臂配对；K116 已使用这些监督标签，不作泛化比较"}
    for name, split in splits.items():
        info["evaluation_splits"][f"{name}_test"] = split["test"]
    return info, pool


def main():
    parser = argparse.ArgumentParser(description="阶段二只读 checkpoint 评估（无训练或 BN 重估）")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("cwru", "paderborn"), required=True)
    parser.add_argument("--selection", default=str(ROOT / "outputs/selection/selection.json"))
    parser.add_argument("--budget", choices=("K19", "K38", "K116", "19", "38", "116"), default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if any((output / name).exists() for name in
           ("metrics.json", "predictions.csv", "unit_predictions.csv", "eval_info.json")):
        raise FileExistsError(f"评估报告已存在，重跑请使用新目录：{output}")
    # 只读取本地可信权重；不调用 v6 评估入口或训练模块。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (not isinstance(checkpoint, dict) or checkpoint.get("protocol") != "v7_stage2"
            or checkpoint.get("checkpoint_kind") != "last" or "state_dict" not in checkpoint):
        raise ValueError("正式评估只接受 v7_stage2 的 last checkpoint，不接受 v6 或 best")
    if any(checkpoint.get(field) != 30 for field in ("epoch", "checkpoint_epoch", "last_epoch")):
        raise ValueError("正式评估只接受完成 30 轮的 last；epoch、checkpoint_epoch、last_epoch 必须均为 30")
    from features import feature_subset, load_cache
    from model import MSRSABiLSTM
    from preprocessing import config_from_saved
    cache = load_cache(args.dataset, args.cache_dir)
    if config_from_saved(checkpoint["repr_config"], require_complete=True) != cache.repr_config:
        raise ValueError("checkpoint 表征与冻结缓存不一致")
    model_config = checkpoint["model"]
    if (model_config.get("n_channels") != 1 or model_config.get("n_classes") != 4
            or model_config.get("frontend", "spectral_v1") != "spectral_v1"):
        raise ValueError("只允许固定的单通道四类 spectral_v1 网络")
    model = MSRSABiLSTM(**{name: model_config[name] for name in
                          ("n_channels", "n_classes", "lstm_hidden", "use_se") if name in model_config})
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    saved_budget = checkpoint.get("target_budget", checkpoint.get("budget"))
    budget = args.budget or saved_budget or (
        checkpoint["arm"][3:] if checkpoint["arm"].startswith("ce_K") else "K19")
    budget = str(budget)
    if not budget.startswith("K"):
        budget = f"K{budget}"
    if saved_budget is not None:
        saved_budget = str(saved_budget)
        saved_budget = saved_budget if saved_budget.startswith("K") else f"K{saved_budget}"
        if saved_budget != budget and checkpoint["arm"] in ("ce_K19", "ce_K38", "ce_K116", "ce_mmd"):
            raise ValueError("命令行预算与 checkpoint 不一致")
    info, pool = _evaluation_info(checkpoint, selection, budget)
    info.update(checkpoint=str(Path(args.checkpoint).resolve()), dataset=args.dataset,
                selection=str(Path(args.selection).resolve()), cache_version=cache.version)
    if args.dataset == "paderborn":
        primary_units = {row["unit_key"] for row in cache.records if row["label"] is not None}
        if primary_units != set(selection["all"]):
            raise ValueError("PU 缓存主口径与冻结 116 单元不一致")
    else:
        # CW 全源报告含源训练文件，另外并报本折未训练验证文件。
        from datasets.cwru import split_source_units
        source_split = checkpoint.get("source_split") or split_source_units(
            [SimpleNamespace(**row) for row in cache.records], split_seed=0)[checkpoint["fold"]]
        info["evaluation_scope"] = "source_all_descriptive_contains_training_units"
        info["evaluation_splits"] = {"source_all": sorted({row["unit_key"] for row in cache.records}),
                                    "source_validation": source_split["val"]}
        info["heldout_scope"] = "source_validation"
    feature = feature_subset(cache, labeled=False)
    records = infer_records(model, feature, device="cuda" if torch.cuda.is_available() else "cpu")
    units = aggregate_units(records)
    metrics = save_reports(records, units, output, pool_units=pool, eval_info=info)
    print(f"评估已落盘：{output}，记录分母 {metrics['record_level']['n']}，单元分母 {metrics['unit_level']['n']}")


if __name__ == "__main__":
    main()
