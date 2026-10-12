# -*- coding: utf-8 -*-
"""冻结的零训练包络谱分类器；规则不根据目标标签或结果调节。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly

from bearing_facts import BY_ID, CONDITIONS

ROOT = Path(__file__).resolve().parent
#: 冻结选标的正式位置；预算档位以 selection.json 为准，此处不重复硬编码校验。
SELECTION_PATH = ROOT / "outputs" / "selection" / "selection.json"
RULE = {
    "protocol": "v7_stage2",
    "band_hz": [2000.0, 5000.0], "filter_order": 4,
    "target_fs": 12000, "harmonics": 5, "tolerance_hz": 2.0,
    "floor_band_hz": [0.0, 500.0], "dominance_ratio": 1.5,
    "classification": "BPFO > 1.5*BPFI 判 OR；BPFI > 1.5*BPFO 判 IR；否则判 Normal",
    "aggregation": "记录预测转为四类 one-hot；单元内记录概率等权均值",
    "information": "使用完整原始记录的包络谐波比，不是深度模型逐窗标准化的谱形特征",
    "label_usage": "标签只用于最后评分；不估计阈值、不调带、不选谐波",
    "training": "零训练：不估计参数、不适配目标数据、非 transductive",
}
def _envelope_functions():
    # 先让 v7 自己的 preprocessing shim（内部走 load_v6 隔离名）与 _v6 进 sys.modules，
    # 脚本内的 `import preprocessing` 命中缓存，不会把 v6 同名模块留在全局名下；
    # 导入后只恢复 sys.path，不收集 sys.modules（误删已登记的合法模块比污染更危险）。
    import _v6, preprocessing  # noqa: F401  预登记缓存，防抢名
    before_path = sys.path[:]
    try:
        from scripts.check_paderborn_pipeline import envelope_spectrum, harmonic_ratio
    finally:
        sys.path[:] = before_path
    return envelope_spectrum, harmonic_ratio


def classify_ratios(bpfo_ratio, bpfi_ratio):
    """严格大于冻结阈值；相等或恰在边界时归为 Normal。"""
    if not math.isfinite(bpfo_ratio) or not math.isfinite(bpfi_ratio):
        raise ValueError("包络谐波比必须有限")
    if bpfo_ratio < 0 or bpfi_ratio < 0:
        raise ValueError("包络谐波比不能为负数")
    if bpfo_ratio > RULE["dominance_ratio"] * bpfi_ratio:
        return 3
    if bpfi_ratio > RULE["dominance_ratio"] * bpfo_ratio:
        return 2
    return 0


def recording_prediction(record):
    """用单记录原始振动与逐轴承几何计算预测，不读取真值作判断。"""
    from datasets.paderborn import read_signal
    if record.dataset != "paderborn" or record.channel != "vibration_1":
        raise ValueError("物理基线只接受 PU 的 vibration_1 通道")
    envelope_spectrum, harmonic_ratio = _envelope_functions()
    signal = read_signal(record)
    divisor = math.gcd(record.src_fs, RULE["target_fs"])
    signal = resample_poly(signal, RULE["target_fs"] // divisor, record.src_fs // divisor)
    if len(signal) < 512 or not np.isfinite(signal).all():
        raise ValueError(f"物理基线输入不足一窗或非有限：{record.record_key}")
    freqs, spectrum = envelope_spectrum(signal)
    bpfo_order, bpfi_order = BY_ID[record.bearing_id].fault_orders
    shaft_hz = CONDITIONS[record.condition][0]
    bpfo = harmonic_ratio(freqs, spectrum, bpfo_order * shaft_hz)
    bpfi = harmonic_ratio(freqs, spectrum, bpfi_order * shaft_hz)
    prediction = classify_ratios(bpfo, bpfi)
    probabilities = [float(i == prediction) for i in range(4)]
    row = asdict(record)
    row["path"] = str(record.path)
    row.update({"probabilities": probabilities, "window_count": len(signal) // 512,
                "bpfo_ratio": bpfo, "bpfi_ratio": bpfi,
                "physical_prediction": prediction})
    return row


def run(output, selection_path=None):
    """全量计算后落盘；不接触权重、不构造优化器、不修改原始数据。"""
    from datasets.paderborn import enumerate_recordings
    from evaluate import aggregate_units, save_reports
    selection_path = Path(selection_path or SELECTION_PATH)
    if not selection_path.is_file():
        raise FileNotFoundError(f"冻结选标尚未落盘，无法按预算并报 heldout：{selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    budgets = selection["budgets"]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"物理基线输出目录已有内容，重跑请用 _retry1 后缀：{output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "rule.json").write_text(json.dumps(RULE, ensure_ascii=False, indent=2), encoding="utf-8")
    records = enumerate_recordings(include_kb=True)
    predictions = []
    for index, record in enumerate(records, 1):
        predictions.append(recording_prediction(record))
        if index % 80 == 0 or index == len(records):
            print(f"物理基线：已计算 {index}/{len(records)} 条记录", flush=True)
    units = aggregate_units(predictions)
    info = {"arm": "phys_envelope", "protocol": "v7_stage2",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_scope": "zero_shot_all_primary", "trained": False,
            "frozen_budgets": [len(split["pool"]) for split in budgets.values()],
            "note": "零训练物理规则：无参数估计、无目标域适配、非 transductive；"
                    "标签只用于最后评分，不进入任何预测判断"}
    save_reports(predictions, units, output, pool_units=[], eval_info=info)
    with (output / "physical_ratios.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["record_key", "unit_key", "bearing_id", "condition", "origin", "severity",
                  "label", "bpfo_ratio", "bpfi_ratio", "physical_prediction", "window_count"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    if selection is not None:
        from evaluate import score_predictions
        heldout = {}
        for name, split in budgets.items():
            wanted = set(split["test"])
            heldout[name] = {
                "scope": "heldout" if wanted else "no_heldout",
                "unit_count": len(wanted),
                "unit_level": score_predictions([u for u in units if u["unit_key"] in wanted],
                                                 pool_units=split["pool"]) if wanted else None,
                "record_level": score_predictions([r for r in predictions if r["unit_key"] in wanted],
                                                   pool_units=split["pool"]) if wanted else None,
            }
        (output / "budget_heldout_metrics.json").write_text(
            json.dumps(heldout, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"物理基线已落盘：{output}", flush=True)
    return output


def _self_check():
    assert classify_ratios(3.0, 2.0) == 0
    assert classify_ratios(3.01, 2.0) == 3
    assert classify_ratios(2.0, 3.01) == 2
    assert classify_ratios(0.0, 0.0) == 0
    print("物理分类规则自检通过，未训练")


def main():
    parser = argparse.ArgumentParser(description="零训练包络谱物理基线")
    parser.add_argument("--output", default=str(ROOT / "outputs/eval/phys_envelope"))
    parser.add_argument("--selection", default=None)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args.output, args.selection)


if __name__ == "__main__":
    main()
