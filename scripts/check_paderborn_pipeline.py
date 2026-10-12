#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paderborn 全量通路验证：解包后的 .mat -> 12 kHz -> 512 点窗 -> logspec512。

目的：**在写任何训练代码之前**，确认全部 32 个轴承能走通 v6 的特征口径，并用包络谱确认
标签与官方事实表一致。特征计算直接 `import` v6 的 `preprocessing.extract_logspec`，不重写。

用法（必须用 fault-detect 环境）：
    E:/tool/anaconda/envs/fault-detect/python.exe scripts/check_paderborn_pipeline.py
    ... --fast     # 每个工况只取前 5 次测量，快速冒烟
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert, resample_poly

ROOT = Path(__file__).resolve().parent.parent
V6 = ROOT.parent / "rbop_diagnosis_v6"
sys.path.insert(0, str(V6))
sys.path.insert(0, str(ROOT))
import preprocessing as v6prep  # noqa: E402  复用 v6 口径，禁止另写一份
from bearing_facts import BEARINGS, BY_ID, CONDITIONS, TRUNCATED_RECORDINGS  # noqa: E402

PU_FS = 64000
TARGET_FS = 12000
SEG_LEN = 512
EXTRACTED = ROOT / "Data" / "paderborn" / "extracted"
ENVELOPE_BAND = (2000.0, 5000.0)


# ---------------------------------------------------------------------------
# 读数据
# ---------------------------------------------------------------------------
def read_recording(path: Path) -> dict[str, np.ndarray]:
    """读一个 PU `.mat` 的全部通道（顶层键 = 文件名词干，`Y` 是 {Name, Data} 列表）。"""
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    stem = path.stem
    if stem not in raw:
        raise KeyError(f"{path.name} 顶层键不是文件名")
    return {channel.Name: np.asarray(channel.Data, dtype=np.float64).ravel()
            for channel in raw[stem].Y}


def to_windows(signal: np.ndarray) -> np.ndarray:
    """64 kHz -> 12 kHz（精确 3/16）-> 不重叠 512 点窗；末尾不足一窗丢弃。"""
    g = np.gcd(TARGET_FS, PU_FS)
    resampled = resample_poly(signal, TARGET_FS // g, PU_FS // g)
    n_windows = len(resampled) // SEG_LEN
    return np.stack([resampled[i * SEG_LEN:(i + 1) * SEG_LEN] for i in range(n_windows)])


# ---------------------------------------------------------------------------
# 包络谱判据（PU 原论文验证数据用的就是包络谱，PHME16 Fig 6/7）
# ---------------------------------------------------------------------------
def envelope_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nyquist = TARGET_FS / 2
    b, a = butter(4, [ENVELOPE_BAND[0] / nyquist, ENVELOPE_BAND[1] / nyquist], btype="band")
    envelope = np.abs(hilbert(filtfilt(b, a, signal)))
    envelope = envelope - envelope.mean()
    spectrum = np.abs(np.fft.rfft(envelope * np.hanning(len(envelope))))
    return np.fft.rfftfreq(len(envelope), 1 / TARGET_FS), spectrum


def harmonic_ratio(freqs, spectrum, base_hz, n_harmonics=5, tol_hz=2.0, band=500.0) -> float:
    """基频及前 n 次谐波处的谱幅均值 / 0–band Hz 内中位谱幅（归一化，消除量级）。"""
    floor = np.median(spectrum[freqs <= band]) + 1e-12
    peaks = []
    for k in range(1, n_harmonics + 1):
        target = base_hz * k
        window = (freqs >= target - tol_hz) & (freqs <= target + tol_hz)
        if window.any():
            peaks.append(spectrum[window].max())
    return float(np.mean(peaks) / floor) if peaks else float("nan")


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def verify_shape(per_condition: int | None) -> tuple[dict, list[str]]:
    """逐文件检查窗数与特征形状；返回 (统计, 异常清单)。"""
    stats, problems = {}, []
    print("=" * 100)
    print("1) 形状自检：每个文件走 64 kHz -> resample_poly(3,16) -> 12 kHz -> 512 点窗 -> logspec512")
    print("=" * 100)
    print(f"{'轴承':6s} {'类别':12s} {'来源':10s} {'文件':>4s} {'窗数分布':28s} {'特征':16s} {'窗内均值':>11s}")
    for bearing in BEARINGS:
        folder = EXTRACTED / bearing.bearing_id
        files = sorted(folder.glob("*.mat"))
        if not files:
            problems.append(f"{bearing.bearing_id}: 无文件")
            continue
        counts, shapes = [], Counter()
        last_feature = None
        skipped = []
        for path in files:
            if path.stem in TRUNCATED_RECORDINGS:
                skipped.append(path.name)  # 官方截断文件，已在 §4 单独报，不混进统计
                continue
            try:
                windows = to_windows(read_recording(path)["vibration_1"])
            except Exception as error:  # noqa: BLE001
                problems.append(f"{bearing.bearing_id}/{path.name}: {type(error).__name__}")
                continue
            counts.append(len(windows))
            last_feature = v6prep.extract_logspec(windows)
            shapes[tuple(last_feature.shape[1:])] += 1
        tally = dict(sorted(Counter(counts).items()))
        stats[bearing.bearing_id] = {"files": len(files), "windows": counts, "tally": tally,
                                     "skipped": skipped}
        shape_note = "OK" if set(shapes) == {(1, 256)} else f"异常{set(shapes)}"
        print(f"{bearing.bearing_id:6s} {bearing.fault_type:12s} {bearing.origin:10s} "
              f"{len(counts):4d} {str(tally):28s} {shape_note:16s} {last_feature.mean():+.2e}"
              + (f"  跳过 {len(skipped)}（官方截断）" if skipped else ""))
        if set(shapes) != {(1, 256)}:
            problems.append(f"{bearing.bearing_id}: 特征形状 {set(shapes)}")
        # 窗数实测范围 91–109（众数 93）。超出此范围才是真异常，不是 93–96。
        if min(counts) < 91 or max(counts) > 109:
            problems.append(f"{bearing.bearing_id}: 窗数越界 {sorted(set(counts))}")
    return stats, problems


def verify_physics(per_condition: int | None) -> tuple[dict, list[str]]:
    """逐轴承 × 逐工况算包络谱 BPFO/BPFI 谐波比，并与官方 fault_type 对照。

    ⚠️ 这个判据是**零训练的粗糙启发式**（固定单带 2000–5000 Hz、固定 5 次谐波、±2 Hz 容差），
    不是分类器。它对弱损伤（L1/L2 人工电刻笔、小压痕）本来就会失效，**失效不等于标签错误**。
    它真正的价值是：(a) 证明通道/采样率/几何/标签链路对；(b) 给出一个**难度排序**。
    因此不一致只作为信息列出，不算异常。
    """
    print()
    print("=" * 100)
    print("2) 物理判据：包络谱谐波比 vs 官方标签（带通 2000–5000 Hz，5 次谐波，归一化到 0–500 Hz 中位谱幅）")
    print("=" * 100)
    print(f"{'轴承':6s} {'官方类别':12s} {'来源':10s} {'严重度':>4s} | "
          f"{'BPFO比':>26s} | {'BPFI比':>26s} | 判读")
    results, mismatches = {}, []
    agree_count, judgeable = 0, 0
    for bearing in BEARINGS:
        folder = EXTRACTED / bearing.bearing_id
        bpfo_ord, bpfi_ord = bearing.fault_orders
        per_condition_ratios = {}
        for condition, (shaft_hz, *_rest) in CONDITIONS.items():
            files = sorted(folder.glob(f"{condition}_{bearing.bearing_id}_*.mat"))
            if per_condition:
                files = files[:per_condition]
            fo, fi = [], []
            for path in files:
                if path.stem in TRUNCATED_RECORDINGS:
                    continue  # 官方截断文件单独报，不混进统计
                signal = resample_poly(read_recording(path)["vibration_1"], 3, 16)
                freqs, spectrum = envelope_spectrum(signal)
                fo.append(harmonic_ratio(freqs, spectrum, bpfo_ord * shaft_hz))
                fi.append(harmonic_ratio(freqs, spectrum, bpfi_ord * shaft_hz))
            per_condition_ratios[condition] = (float(np.median(fo)), float(np.median(fi)))
        results[bearing.bearing_id] = per_condition_ratios

        fo_all = [v[0] for v in per_condition_ratios.values()]
        fi_all = [v[1] for v in per_condition_ratios.values()]
        # 判读用 1500 rpm 工况的中位（避开 N09 的低速衰减，见 §2.9）
        fast = [c for c in CONDITIONS if c.startswith("N15")]
        fo_fast = float(np.median([per_condition_ratios[c][0] for c in fast]))
        fi_fast = float(np.median([per_condition_ratios[c][1] for c in fast]))
        if fo_fast > 1.5 * fi_fast:
            verdict = "BPFO 主导"
        elif fi_fast > 1.5 * fo_fast:
            verdict = "BPFI 主导"
        else:
            verdict = "两者相近"
        expect = {"normal": "两者相近", "outer": "BPFO 主导", "inner": "BPFI 主导",
                  "inner+outer": "任意（复合故障）"}[bearing.fault_type]
        if bearing.is_kb:
            agree = "—"
        else:
            judgeable += 1
            agree = "✓" if verdict == expect else "✗"
            if agree == "✓":
                agree_count += 1
            else:
                mismatches.append((bearing.bearing_id, bearing.fault_type, bearing.origin,
                                   bearing.severity, verdict, fo_fast, fi_fast))
        print(f"{bearing.bearing_id:6s} {bearing.fault_type:12s} {bearing.origin:10s} "
              f"L{bearing.severity:<3d} | "
              f"{' '.join(f'{v:5.1f}' for v in fo_all):>26s} | "
              f"{' '.join(f'{v:5.1f}' for v in fi_all):>26s} | {verdict:9s} 期望 {expect:16s} {agree}")

    print(f"\n判据命中 {agree_count}/{judgeable}（主口径 29 个轴承，KB 不参与判定）")
    print("未命中的都是弱损伤或异常方向，**不构成标签错误**，但构成难度排序与待查清单：")
    for bearing_id, fault_type, origin, severity, verdict, fo, fi in mismatches:
        print(f"  {bearing_id:5s} 官方 {fault_type:6s} / {origin:10s} / L{severity}  "
              f"判读 {verdict:9s}（1500 rpm 中位 BPFO {fo:5.1f} vs BPFI {fi:5.1f}）")
    return results, []


def verify_condition_effect(results: dict) -> None:
    """逐工况聚合，确认 N09（900 rpm）低速衰减。"""
    print()
    print("=" * 100)
    print("3) 逐工况聚合：N09（900 rpm）的低速衰减是否如预期")
    print("=" * 100)
    print(f"{'工况':13s} {'转频':>6s} {'BPFO 周期/窗':>12s} | "
          f"{'外圈轴承 BPFO比 中位':>22s} {'内圈轴承 BPFI比 中位':>22s} {'健康 BPFO比':>12s}")
    outer = [b.bearing_id for b in BEARINGS if b.fault_type == "outer"]
    inner = [b.bearing_id for b in BEARINGS if b.fault_type == "inner"]
    healthy = [b.bearing_id for b in BEARINGS if b.fault_type == "normal"]
    rows = []
    for condition, (shaft_hz, *_rest) in CONDITIONS.items():
        bpfo_cycles = 3.071 * shaft_hz * SEG_LEN / TARGET_FS
        outer_ratio = float(np.median([results[b][condition][0] for b in outer]))
        inner_ratio = float(np.median([results[b][condition][1] for b in inner]))
        healthy_ratio = float(np.median([results[b][condition][0] for b in healthy]))
        rows.append((condition, outer_ratio))
        print(f"{condition:13s} {shaft_hz:5.1f}Hz {bpfo_cycles:12.2f} | "
              f"{outer_ratio:22.2f} {inner_ratio:22.2f} {healthy_ratio:12.2f}")
    best = max(rows, key=lambda item: item[1])
    worst = min(rows, key=lambda item: item[1])
    print(f"\n外圈判据最强/最弱工况：{best[0]}（{best[1]:.2f}） / {worst[0]}（{worst[1]:.2f}）"
          f"，相差 {best[1]/worst[1]:.1f} 倍")
    print("-> 两个方向都要标注：低转速（N09，单窗只覆盖 1.97 个 BPFO 周期）"
          "与小径向力（F04，400 N）都会明显削弱外圈特征。")


def verify_truncated() -> None:
    """官方截断文件的实际状态。"""
    print()
    print("=" * 100)
    print("4) 官方截断文件")
    print("=" * 100)
    for stem in sorted(TRUNCATED_RECORDINGS):
        condition, bearing_id, _rep = stem.rsplit("_", 2)
        path = EXTRACTED / bearing_id / f"{stem}.mat"
        if not path.is_file():
            print(f"  {stem}: 文件不存在")
            continue
        print(f"  {stem}: {path.stat().st_size / 1048576:.2f} MiB（正常约 8.3 MiB）")
        try:
            channels = read_recording(path)
            print(f"    scipy.loadmat 成功，通道 {sorted(channels)}，"
                  f"vibration_1 {len(channels['vibration_1'])} 点")
        except Exception as error:  # noqa: BLE001
            print(f"    scipy.loadmat 失败（预期）：{type(error).__name__} {str(error)[:80]}")
            print("    -> 已冻结：整条丢弃，原文件保留；KA08/N15_M01_F10 按 19 条记录聚合")


def verify_cwru_distance() -> None:
    """与 CWRU 的早期距离（仅定性）。"""
    print()
    print("=" * 100)
    print("5) 与 CWRU 的指纹距离（256 维均值谱欧氏距离，仅定性，不构成结论）")
    print("=" * 100)
    import dataset as v6ds  # noqa: E402

    def cwru(dataset_name: str, filename: str, channel: str = "DE") -> np.ndarray | None:
        records = v6ds.enumerate_files(datasets=(dataset_name,))
        keys = [r["file_key"] for r in records if r["filename"] == filename]
        if not keys:
            return None
        return v6prep.build_record_features(records, keys, channel).X[:, 0, :].numpy().mean(axis=0)

    refs = {}
    for label, dataset_name, filename in (("CWRU-Normal", "Normal", "Normal_1.mat"),
                                          ("CWRU-OR(FE)", "12k_FE", "OR007@6_1.mat"),
                                          ("CWRU-IR(FE)", "12k_FE", "IR007_1.mat"),
                                          ("CWRU-B(FE)", "12k_FE", "B007_1.mat")):
        vector = cwru(dataset_name, filename)
        if vector is not None:
            refs[label] = vector

    pu = {}
    for bearing in BEARINGS:
        folder = EXTRACTED / bearing.bearing_id
        parts = []
        for path in sorted(folder.glob("*.mat"))[:8]:
            if path.stem in TRUNCATED_RECORDINGS:
                continue
            parts.append(v6prep.extract_logspec(to_windows(read_recording(path)["vibration_1"]))[:, 0, :])
        if parts:
            pu[bearing.bearing_id] = np.concatenate(parts, axis=0).mean(axis=0)

    print(f"{'轴承':6s} {'类别':12s} " + "".join(f"{k:>14s}" for k in refs))
    for bearing in BEARINGS:
        if bearing.bearing_id not in pu:
            continue
        row = "".join(f"{np.linalg.norm(pu[bearing.bearing_id] - v):14.2f}" for v in refs.values())
        print(f"{bearing.bearing_id:6s} {bearing.fault_type:12s} {row}")

    healthy = [pu[b.bearing_id] for b in BEARINGS if b.fault_type == "normal" and b.bearing_id in pu]
    outer = [pu[b.bearing_id] for b in BEARINGS if b.fault_type == "outer" and b.bearing_id in pu]
    inner = [pu[b.bearing_id] for b in BEARINGS if b.fault_type == "inner" and b.bearing_id in pu]
    def spread(group):
        return float(np.median([np.linalg.norm(a - b) for i, a in enumerate(group) for b in group[i + 1:]]))
    print(f"\nPU 内部类内/类间距离中位："
          f"健康 {spread(healthy):.2f}、外圈 {spread(outer):.2f}、内圈 {spread(inner):.2f}")
    cross = [np.linalg.norm(a - b) for a in outer for b in inner]
    print(f"外圈↔内圈 跨类距离中位：{np.median(cross):.2f}")
    to_cwru = [np.linalg.norm(pu[k] - v) for k in pu for v in refs.values()]
    print(f"PU↔CWRU 跨域距离中位：{np.median(to_cwru):.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paderborn 全量通路验证")
    parser.add_argument("--fast", action="store_true", help="每个工况只取前 5 次测量")
    parser.add_argument("--skip-cwru", action="store_true", help="跳过与 CWRU 的对照")
    args = parser.parse_args()
    per_condition = 5 if args.fast else None

    stats, problems = verify_shape(per_condition)
    results, physics_problems = verify_physics(per_condition)
    verify_condition_effect(results)
    verify_truncated()
    if not args.skip_cwru:
        verify_cwru_distance()

    print()
    print("=" * 100)
    total_files = sum(s["files"] for s in stats.values())
    all_windows = [w for s in stats.values() for w in s["windows"]]
    print(f"汇总：{len(stats)} 个轴承、{total_files} 个文件、{sum(all_windows)} 个窗"
          f"（窗数 {min(all_windows)}–{max(all_windows)}，众数 {Counter(all_windows).most_common(1)[0][0]}）")
    problems = problems + physics_problems
    if problems:
        print(f"⚠️ 发现 {len(problems)} 处异常：")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("ALL PADERBORN PIPELINE CHECKS PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
