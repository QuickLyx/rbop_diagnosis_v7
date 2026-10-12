# -*- coding: utf-8 -*-
"""PU 单元级无标签选标：固定 k-means、medoid 与绝对标注预算。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from bearing_facts import PRIMARY_BEARINGS

ROOT = Path(__file__).resolve().parent
K_SEARCH = tuple(range(6, 21))
KMEANS_N_INIT = 25
RANDOM_STATE = 0
SILHOUETTE_TOLERANCE = 0.95


def unit_level_features(cache):
    """116 单元内全部窗口等权均值，不用标签或损伤元数据选代表。"""
    groups = {}
    for record in cache.records:
        if record["bearing_id"] not in PRIMARY_BEARINGS:
            continue
        groups.setdefault(record["unit_key"], []).append(record)
    keys = sorted(groups)
    if len(keys) != 116:
        raise ValueError(f"主口径必须恰有 116 个单元，实际为 {len(keys)}")
    matrix = []
    for key in keys:
        records = groups[key]
        total = sum(record["stop"] - record["start"] for record in records)
        if total <= 0:
            raise ValueError(f"单元没有可用窗口：{key}")
        summed = sum(np.asarray(cache.X[r["start"]:r["stop"], 0], dtype=np.float64).sum(axis=0)
                     for r in records)
        matrix.append(summed / total)
    return keys, np.stack(matrix)


def select_units(keys, matrix):
    """输入只含键和信号特征；返回完整的无标签代表序列及簇诊断。"""
    if len(set(keys)) != len(keys) or matrix.shape != (len(keys), 256):
        raise ValueError("选标键必须唯一，特征形状必须为 (单元数,256)")
    if not np.isfinite(matrix).all():
        raise ValueError("选标特征包含非有限值")
    z = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-12)
    curve, fits = {}, {}
    with threadpool_limits(limits=1):
        for k in K_SEARCH:
            fitted = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
            labels = fitted.fit_predict(z)
            if len(set(labels)) != k:
                raise ValueError(f"k={k} 出现退化簇，不能冻结选标结果")
            curve[k] = float(silhouette_score(z, labels))
            fits[k] = (labels, fitted.cluster_centers_)
    k_max = max(curve, key=lambda k: (curve[k], -k))
    target = SILHOUETTE_TOLERANCE * curve[k_max]
    eligible = [k for k in K_SEARCH if curve[k] >= target]
    if not eligible:
        raise ValueError("silhouette 全为负且无 k 满足冻结的 95% 规则，请复核，不自动换规则")
    selected = min(eligible)
    labels, centers = fits[selected]
    distances = np.linalg.norm(z - centers[labels], axis=1)
    members = {cluster: sorted(np.flatnonzero(labels == cluster),
                               key=lambda i: (float(distances[i]), keys[i]))
               for cluster in range(selected)}
    # 先各簇一个 medoid，再按同一簇顺序轮转追加；各预算是同一序列的前缀。
    order = sorted(members, key=lambda c: (len(members[c]) == 1, c))
    ranked = []
    for depth in range(max(map(len, members.values()))):
        for cluster in order:
            if depth < len(members[cluster]):
                ranked.append(keys[members[cluster][depth]])
    assert len(ranked) == len(keys) and len(set(ranked)) == len(keys)
    return {
        "k_selected": selected,
        "k_at_max_silhouette": k_max,
        "silhouette_curve": {str(k): curve[k] for k in K_SEARCH},
        "silhouette_target": target,
        "cluster_sizes": {str(c): len(members[c]) for c in members},
        "medoids": [keys[members[c][0]] for c in order],
        "ranked_units": ranked,
        "cluster_labels": {key: int(labels[i]) for i, key in enumerate(keys)},
        "distance_to_centroid": {key: float(distances[i]) for i, key in enumerate(keys)},
        "all": keys,
    }


def budget_splits(payload, budgets):
    """每个 K 对应冻结序列前缀，池与测试并集为全部 116 个单元。"""
    all_keys = payload["all"]
    ranked = payload["ranked_units"]
    if len(set(ranked)) != len(all_keys) or set(ranked) != set(all_keys):
        raise ValueError("代表序列必须不重不漏地覆盖全部单元")
    result = {}
    for k in sorted(set(budgets)):
        if not isinstance(k, int) or not payload["k_selected"] <= k <= len(all_keys):
            raise ValueError(f"绝对预算必须在 k={payload['k_selected']} 至 116 之间：{k}")
        pool = sorted(ranked[:k])
        test = sorted(set(all_keys) - set(pool))
        assert not set(pool) & set(test)
        assert set(pool) | set(test) == set(all_keys)
        result[f"K{k}"] = {"pool": pool, "test": test,
                            "label_unit_count": len(pool), "heldout_unit_count": len(test)}
    return result


def diagnostics(cache, splits):
    """池落定后才读真值，仅记录覆盖情况，不回流到选标规则。"""
    by_unit = {r["unit_key"]: r for r in cache.records if r["bearing_id"] in PRIMARY_BEARINGS}
    result = {}
    for name, split in splits.items():
        pool, test = split["pool"], split["test"]
        seen = {by_unit[key]["bearing_id"] for key in pool}
        result[name] = {
            "pool_class_counts": dict(Counter(str(by_unit[k]["label"]) for k in pool)),
            "test_class_counts": dict(Counter(str(by_unit[k]["label"]) for k in test)),
            "pool_origin_counts": dict(Counter(by_unit[k]["origin"] for k in pool)),
            "pool_condition_counts": dict(Counter(by_unit[k]["condition"] for k in pool)),
            "seen_bearings": sorted(seen),
            "unseen_bearings": sorted(set(PRIMARY_BEARINGS) - seen),
            "unseen_test_units": [k for k in test if by_unit[k]["bearing_id"] not in seen],
            "missing_pool_classes": sorted({0, 2, 3} - {by_unit[k]["label"] for k in pool}),
        }
    return result


def build_payload(cache, budgets=None):
    keys, matrix = unit_level_features(cache)
    result = select_units(keys, matrix)
    final_budgets = budgets if budgets is not None else [result["k_selected"], 116]
    result.update({
        "protocol": "v7_stage2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "budgets_status": "frozen" if budgets is not None else "pending",
        "cache_version": cache.version,
        "repr_config": cache.repr_config,
        "rule": {
            "feature": "单元内全部 logspec512 窗口等权均值，候选集按维 z-score(ddof=0)",
            "k_search": list(K_SEARCH), "n_init": KMEANS_N_INIT,
            "random_state": RANDOM_STATE, "silhouette_tolerance": SILHOUETTE_TOLERANCE,
            "representative": "各簇 medoid，同分按单元键升序；单例簇最后",
            "additional_budget": "按簇顺序轮转追加各簇距簇心最近的剩余单元；预算为嵌套前缀",
            "metadata_used_in_selection": False,
            "short_record_filter": "不应用：可用 PU 记录均不少于 91 窗",
        },
    })
    result["budgets"] = budget_splits(result, final_budgets)
    result["diagnostics"] = diagnostics(cache, result["budgets"])
    return result


def main():
    parser = argparse.ArgumentParser(description="PU 单元级无标签簇覆盖选标（不加载模型、不训练）")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--budgets", nargs="+", type=int, default=None,
                        help="选标跑出 k 后明确指定绝对预算；未传时仅生成待确认候选")
    parser.add_argument("--out", default=str(ROOT / "outputs/selection/candidate.json"))
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"不覆盖已有选标文件：{out}")
    from features import load_cache
    payload = build_payload(load_cache("paderborn", args.cache_dir), args.budgets)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"选标完成：k={payload['k_selected']}，簇大小={payload['cluster_sizes']}")
    print(f"预算状态：{payload['budgets_status']}，写入：{out}")
    for name, split in payload["budgets"].items():
        print(f"  {name}：池 {len(split['pool'])}，测试 {len(split['test'])}；"
              f"未覆盖池类别 {payload['diagnostics'][name]['missing_pool_classes']}")


if __name__ == "__main__":
    main()
