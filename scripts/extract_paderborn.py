#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paderborn `.rar` 批量解包（幂等）。

用法：
    python scripts/extract_paderborn.py            # 解包 _raw 下所有尚未解包的 .rar
    python scripts/extract_paderborn.py K001 KA04  # 只解包指定轴承
    python scripts/extract_paderborn.py --check    # 只检查已解包情况，不解包

实测要点：
- **不需要 unrar / 7z**：anaconda 自带的 `bsdtar 3.8.2` / `libarchive 3.8.2` 能读 RAR4
  （PU 的 rar 签名是 `Rar!\\x1a\\x07\\x00`）。
- 解包后是**两层**：`<bearing>/<工况>_<bearing>_<rep>.mat`（方案初稿写的三层是错的）。
- 每个 rar 里除 80 个 `.mat` 还含 `<bearing>.pdf`（官方事实表）与 `measuring_log_<bearing>.pdf`。
- rar 不保序，**必须 glob + sort**，不能依赖解包顺序。
- `KA08` 的 `N15_M01_F10_KA08_2.mat` 是官方截断文件，解包后单独打标。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Data" / "paderborn" / "_raw"
OUT_DIR = ROOT / "Data" / "paderborn" / "extracted"

sys.path.insert(0, str(ROOT))
from bearing_facts import BY_ID, CONDITIONS, TRUNCATED_RECORDINGS  # noqa: E402

EXPECTED_MAT = len(CONDITIONS) * 20  # 4 工况 × 20 次 = 80
#: 解包后每个 rar 应产生的条目：1 个目录 + 80 个 .mat + 2 个 PDF
EXPECTED_ENTRIES = EXPECTED_MAT + 3


def find_bsdtar() -> str:
    """定位 bsdtar；找不到就报错并给出替代方案。"""
    found = shutil.which("bsdtar") or shutil.which("bsdtar.exe")
    if found:
        return found
    for candidate in (
        r"E:\tool\anaconda\Library\bin\bsdtar.exe",
        r"C:\Program Files\Git\usr\bin\bsdtar.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "找不到 bsdtar。不要改用 unrar/7z（本机未安装）。"
        "请把 anaconda 的 Library/bin 加入 PATH，或安装 libarchive。"
    )


def listing(bsdtar: str, archive: Path) -> list[str]:
    result = subprocess.run([bsdtar, "-tf", str(archive)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"列出 {archive.name} 失败：{result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def mat_files(bearing: str) -> list[Path]:
    folder = OUT_DIR / bearing
    return sorted(folder.glob("*.mat")) if folder.is_dir() else []


def extract_one(bsdtar: str, bearing: str, force: bool = False) -> tuple[str, int]:
    """解包单个轴承；返回 (状态, mat 数)。已完整解包的直接跳过。"""
    archive = RAW_DIR / f"{bearing}.rar"
    if not archive.is_file():
        return "缺少 .rar", 0

    existing = mat_files(bearing)
    if len(existing) == EXPECTED_MAT and not force:
        return "已解包，跳过", len(existing)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [bsdtar, "-xf", str(archive), "-C", str(OUT_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"解包失败：{result.stderr.strip()[:120]}", len(mat_files(bearing))

    files = mat_files(bearing)
    return ("解包完成" if len(files) == EXPECTED_MAT else f"⚠️ mat 数 {len(files)} != {EXPECTED_MAT}"), len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量解包 Paderborn .rar")
    parser.add_argument("bearings", nargs="*", help="轴承号；省略则处理 _raw 下所有 .rar")
    parser.add_argument("--force", action="store_true", help="即使已解包也重新解包")
    parser.add_argument("--check", action="store_true", help="只检查，不解包")
    args = parser.parse_args()

    if args.bearings:
        targets = [b.upper() for b in args.bearings]
        unknown = [b for b in targets if b not in BY_ID]
        if unknown:
            print(f"未知轴承号：{unknown}")
            return 2
    else:
        targets = sorted(p.stem for p in RAW_DIR.glob("*.rar")) if RAW_DIR.is_dir() else []
    if not targets:
        print(f"{RAW_DIR} 下没有 .rar，先跑 scripts/fetch_paderborn.py")
        return 1

    if args.check:
        print(f"{'轴承':6s} {'类别':12s} {'mat 数':>6s}  {'PDF':>4s}  状态")
        bad = []
        for bearing in targets:
            files = mat_files(bearing)
            pdfs = sorted((OUT_DIR / bearing).glob("*.pdf")) if (OUT_DIR / bearing).is_dir() else []
            ok = len(files) == EXPECTED_MAT and len(pdfs) == 2
            print(f"{bearing:6s} {BY_ID[bearing].fault_type:12s} {len(files):6d}  {len(pdfs):4d}  "
                  f"{'ok' if ok else '不完整'}")
            if not ok:
                bad.append(bearing)
        print(f"\n{len(targets) - len(bad)}/{len(targets)} 完整")
        if bad:
            print(f"不完整：{bad}")
            return 1
        return 0

    bsdtar = find_bsdtar()
    print(f"bsdtar: {bsdtar}")
    print(f"待处理 {len(targets)} 个轴承 -> {OUT_DIR}\n")

    failures = []
    total_mat = 0
    for index, bearing in enumerate(targets, 1):
        status, count = extract_one(bsdtar, bearing, force=args.force)
        total_mat += count
        fact = BY_ID[bearing]
        marker = "⚠️ " if "⚠️" in status or "失败" in status else ""
        print(f"[{index:2d}/{len(targets)}] {bearing} {fact.fault_type:12s} "
              f"{fact.origin:10s} L{fact.severity}  {marker}{status}（{count} mat）")
        if "失败" in status or "⚠️" in status:
            failures.append(bearing)

    print(f"\n合计 {total_mat} 个 .mat（期望 {len(targets) * EXPECTED_MAT}）")
    for name in sorted(TRUNCATED_RECORDINGS):
        print(f"注意：{name} 是官方截断文件，需在加载时单独处理")

    if failures:
        print(f"未完成：{failures}")
        return 1
    print("全部解包完成。下一步：python scripts/check_paderborn_pipeline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
