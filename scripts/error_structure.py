#!/usr/bin/env python3
# 零算力错误结构分析：只读 outputs/eval/ 下 96 份 PU 评估 + phys_envelope，
# 汇总错分结构（类别/工况/损伤来源/塌类/P_ball）、分组并报与六对配对链比较。
# 不训练、不改任何已有产物；结果写 outputs/summary_2026-10-13/error_structure.json。

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "outputs", "eval")
SUMMARY = os.path.join(ROOT, "outputs", "summary_2026-10-13")
OUT_JSON = os.path.join(SUMMARY, "error_structure.json")

MODEL_ARMS = [
    "joint", "source_continue", "adabn", "mmd_marginal",
    "ce_K19", "ce_K38", "ce_K116", "ce_mmd",
]
PAIR_CHAINS = [
    ("joint", "mmd_marginal"),
    ("adabn", "mmd_marginal"),
    ("mmd_marginal", "ce_K19"),
    ("joint", "ce_K19"),
    ("source_continue", "ce_K19"),
    ("ce_mmd", "ce_K19"),
]
CONDITIONS = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10"]
ORIGINS = ["none", "artificial", "real"]
CLASSES = ["Normal", "Ball", "IR", "OR"]
LAB = {0: "Normal", 1: "Ball", 2: "IR", 3: "OR"}


def macro_acc(df, pred_col):
    accs = []
    for c in (0, 2, 3):
        sub = df[df["label"] == c]
        if len(sub):
            accs.append(float((sub[pred_col] == c).mean()))
    return float(np.mean(accs)) if accs else None


def confusion(df, pred_col):
    cm = np.zeros((4, 4), dtype=int)
    for t in (0, 1, 2, 3):
        for p in (0, 1, 2, 3):
            cm[t, p] = int(((df["label"] == t) & (df[pred_col] == p)).sum())
    return cm.tolist()


def p_ball(df, pred_col="prediction_E1"):
    return float((df[pred_col] == 1).mean()) if len(df) else None


def load_grid(arm, fold, seed):
    d = os.path.join(EVAL, f"{arm}_f{fold}_s{seed}_paderborn")
    up = pd.read_csv(os.path.join(d, "unit_predictions.csv"))
    info = json.load(open(os.path.join(d, "eval_info.json"), encoding="utf-8"))
    metrics = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
    return up, info, metrics


def grid_stats(up, info, metrics):
    es = info["evaluation_splits"]
    held = es["natural_heldout"]
    pool = info.get("supervised_pool_units") or []
    pool_set = set(pool)
    seen_bearings = {u.split("/")[0] for u in pool}
    k19_test = es["common_K19_test"]

    h = up[up["unit_key"].isin(set(held))].copy()
    k19 = up[up["unit_key"].isin(set(k19_test))].copy()
    pool_df = up[up["unit_key"].isin(pool_set)].copy()

    def unseen_bearing_group(df):
        return df[~df["bearing_id"].isin(seen_bearings)]

    st = {
        "D3_E1": metrics["D3"]["E1"],
        "D3_E2": metrics["D3"]["E2"],
        "D3_passed": metrics["D3"]["passed"],
        "held_n": len(h),
        "held_E1": macro_acc(h, "prediction_E1"),
        "held_E2": macro_acc(h, "prediction_E2"),
        "held_P_ball": p_ball(h),
        "pool_E1": macro_acc(pool_df, "prediction_E1") if len(pool_df) else None,
        "k19_E1": macro_acc(k19, "prediction_E1"),
        "k19_E2": macro_acc(k19, "prediction_E2"),
        "k19_n": len(k19),
        "held_confusion_E1": confusion(h, "prediction_E1"),
        "held_confusion_E2": confusion(h, "prediction_E2"),
        "held_by_condition_E1": {
            c: macro_acc(h[h["condition"] == c], "prediction_E1") for c in CONDITIONS
        },
        "held_by_condition_E2": {
            c: macro_acc(h[h["condition"] == c], "prediction_E2") for c in CONDITIONS
        },
        "held_by_origin_E1": {
            o: macro_acc(h[h["origin"] == o], "prediction_E1") for o in ORIGINS
        },
        "held_by_class_recall_E1": {
            LAB[c]: (float((h[h["label"] == c]["prediction_E1"] == c).mean())
                     if (h["label"] == c).any() else None)
            for c in (0, 2, 3)
        },
        # 每工况预测分布（判塌类用，含 Ball 列）
        "held_pred_dist_by_condition": {
            c: {
                LAB[p]: int((h[h["condition"] == c]["prediction_E1"] == p).sum())
                for p in (0, 1, 2, 3)
            }
            for c in CONDITIONS
        },
        "unseen_bearing": None,
    }
    ug = unseen_bearing_group(h)
    if len(ug) and seen_bearings:
        st["unseen_bearing"] = {
            "n_units": int(len(ug)),
            "n_bearings": int(ug["bearing_id"].nunique()),
            "E1": macro_acc(ug, "prediction_E1"),
            "E2": macro_acc(ug, "prediction_E2"),
        }
    # 记录级（并报口径）
    return st


def record_level_confusion(arm, fold, seed):
    d = os.path.join(EVAL, f"{arm}_f{fold}_s{seed}_paderborn")
    pr = pd.read_csv(os.path.join(d, "predictions.csv"))
    info = json.load(open(os.path.join(d, "eval_info.json"), encoding="utf-8"))
    held = set(info["evaluation_splits"]["natural_heldout"])
    pr = pr[pr["unit_key"].isin(held)]
    return confusion(pr, "prediction_E1"), confusion(pr, "prediction_E2")


def agg_arm():
    out = {}
    for arm in MODEL_ARMS:
        grids = []
        rec_cm_e1 = np.zeros((4, 4), dtype=int)
        rec_cm_e2 = np.zeros((4, 4), dtype=int)
        for fold in range(4):
            for seed in range(3):
                up, info, metrics = load_grid(arm, fold, seed)
                st = grid_stats(up, info, metrics)
                grids.append(st)
                c1, c2 = record_level_confusion(arm, fold, seed)
                rec_cm_e1 += np.array(c1)
                rec_cm_e2 += np.array(c2)

        def ms(key):
            vals = [g[key] for g in grids if g[key] is not None]
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            } if vals else None

        conf_e1 = np.sum([np.array(g["held_confusion_E1"]) for g in grids], axis=0)
        conf_e2 = np.sum([np.array(g["held_confusion_E2"]) for g in grids], axis=0)

        # 工况/来源/类召回：12 格平均
        def avg_dict(key):
            d = {}
            for k in grids[0][key]:
                vals = [g[key][k] for g in grids if g[key][k] is not None]
                d[k] = float(np.mean(vals)) if vals else None
            return d

        dist = {}
        for c in CONDITIONS:
            dist[c] = {
                LAB[p]: int(np.sum([g["held_pred_dist_by_condition"][c][LAB[p]]
                                    for g in grids]))
                for p in (0, 1, 2, 3)
            }

        unseens = [g["unseen_bearing"] for g in grids if g["unseen_bearing"]]
        out[arm] = {
            "n_grids": len(grids),
            "D3_E1": ms("D3_E1"), "D3_E2": ms("D3_E2"),
            "D3_pass_count": int(sum(g["D3_passed"] for g in grids)),
            "held_n": grids[0]["held_n"],
            "held_E1": ms("held_E1"), "held_E2": ms("held_E2"),
            "held_P_ball": ms("held_P_ball"),
            "pool_E1": ms("pool_E1"),
            "k19_common_E1": ms("k19_E1"), "k19_common_E2": ms("k19_E2"),
            "held_confusion_E1_sum": conf_e1.tolist(),
            "held_confusion_E2_sum": conf_e2.tolist(),
            "record_confusion_E1_sum": rec_cm_e1.tolist(),
            "record_confusion_E2_sum": rec_cm_e2.tolist(),
            "held_by_condition_E1": avg_dict("held_by_condition_E1"),
            "held_by_condition_E2": avg_dict("held_by_condition_E2"),
            "held_by_origin_E1": avg_dict("held_by_origin_E1"),
            "held_class_recall_E1": avg_dict("held_by_class_recall_E1"),
            "held_pred_dist_by_condition_sum": dist,
            "unseen_bearing": ({
                "n_units": unseens[0]["n_units"],
                "n_bearings": unseens[0]["n_bearings"],
                "E1_mean": float(np.mean([u["E1"] for u in unseens])),
                "E2_mean": float(np.mean([u["E2"] for u in unseens])),
            } if unseens else None),
            "per_grid_held_E1": [g["held_E1"] for g in grids],
        }
    return out


def paired_chains():
    out = {}
    cache = {}
    for arm in {a for pair in PAIR_CHAINS for a in pair}:
        per_grid = []
        for fold in range(4):
            for seed in range(3):
                up, info, metrics = load_grid(arm, fold, seed)
                k19 = set(info["evaluation_splits"]["common_K19_test"])
                sub = up[up["unit_key"].isin(k19)]
                per_grid.append({
                    "E1": macro_acc(sub, "prediction_E1"),
                    "E2": macro_acc(sub, "prediction_E2"),
                })
        cache[arm] = per_grid
    for a, b in PAIR_CHAINS:
        for metric in ("E1", "E2"):
            diffs = [ga[metric] - gb[metric] for ga, gb in zip(cache[a], cache[b])]
            wins = int(sum(d > 0 for d in diffs))
            out[f"{a} vs {b} ({metric}, common K19 test, n=97)"] = {
                f"{a}_mean": float(np.mean([g[metric] for g in cache[a]])),
                f"{b}_mean": float(np.mean([g[metric] for g in cache[b]])),
                "diff_mean": float(np.mean(diffs)),
                "diff_std": float(np.std(diffs)),
                "win_count_of_12": wins,
            }
    return out


def phys_baseline():
    d = os.path.join(EVAL, "phys_envelope")
    up = pd.read_csv(os.path.join(d, "unit_predictions.csv"))
    metrics = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
    bh = json.load(open(os.path.join(d, "budget_heldout_metrics.json"),
                       encoding="utf-8"))
    return {
        "D3_E1": metrics["D3"]["E1"],
        "D3_E2": metrics["D3"]["E2"],
        "all116_confusion_E1": confusion(up, "prediction_E1"),
        "by_condition_E1": {
            c: macro_acc(up[up["condition"] == c], "prediction_E1")
            for c in CONDITIONS
        },
        "by_origin_E1": {
            o: macro_acc(up[up["origin"] == o], "prediction_E1") for o in ORIGINS
        },
        "class_recall_E1": {
            LAB[c]: (float((up[up["label"] == c]["prediction_E1"] == c).mean())
                     if (up["label"] == c).any() else None)
            for c in (0, 2, 3)
        },
        "P_ball": p_ball(up),
        "budget_heldout": {
            k: (v["unit_level"]["E1"]["macro_recall"] if v.get("unit_level") else None)
            for k, v in bh.items() if isinstance(v, dict)
        },
    }


def main():
    os.makedirs(SUMMARY, exist_ok=True)
    result = {
        "arm_summary": agg_arm(),
        "paired_chains": paired_chains(),
        "phys_envelope": phys_baseline(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"written: {OUT_JSON}")
    # 控制台速览
    def fmt(x, nd=4):
        return "None" if not x else ("%.*f±%.*f" % (nd, x["mean"], nd, x["std"]))
    for arm, v in result["arm_summary"].items():
        print(f"{arm:16s} held_E1={fmt(v['held_E1'])} "
              f"held_E2={'None' if not v['held_E2'] else format(v['held_E2']['mean'], '.4f')} "
              f"P_ball={'None' if not v['held_P_ball'] else format(v['held_P_ball']['mean'], '.4f')} "
              f"pool_E1={'-' if not v['pool_E1'] else format(v['pool_E1']['mean'], '.4f')}")
    print("phys:", result["phys_envelope"]["D3_E1"])


if __name__ == "__main__":
    sys.exit(main())
