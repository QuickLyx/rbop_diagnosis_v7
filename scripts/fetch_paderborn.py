#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paderborn (KAt) 数据获取：分阶段下载 + sha256 校验 + 断点续传。

用法：
    python scripts/fetch_paderborn.py P0          # K001 + KA04 + KI04（健康/外圈真实/内圈真实）
    python scripts/fetch_paderborn.py P1          # 全部真实损伤 14 + 健康 6
    python scripts/fetch_paderborn.py P2          # 全部人工损伤 12
    python scripts/fetch_paderborn.py all         # 32 个轴承，5.4 GB
    python scripts/fetch_paderborn.py P0 --verify # 只校验已下文件，不下载

官方源 https://groups.uni-paderborn.de/kat/BearingDataCenter/ ，许可 CC-BY-NC-4.0。
文件清单与 sha256 取自官方索引 + bearing-datasets 的 files.lock.json（已核对尺寸）。
引用：Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016). PHM Society
European Conference, 3(1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Data" / "paderborn" / "_raw"

# --- 32 个轴承的事实表（官网索引核准；厂商/来源/严重度/损伤方式见 docs/阶段二方案_v1.md §2.1）---
# bearing_id -> (类别, 损伤来源, 严重度, 损伤方式, 制造商)
BEARING_FACTS = {
    "K001": ("normal", "-", 0, "none", "IBU"),
    "K002": ("normal", "-", 0, "none", "IBU"),
    "K003": ("normal", "-", 0, "none", "IBU"),
    "K004": ("normal", "-", 0, "none", "IBU"),
    "K005": ("normal", "-", 0, "none", "IBU"),
    "K006": ("normal", "-", 0, "none", "IBU"),
    "KA01": ("outer", "artificial", 1, "EDM", "MTK"),
    "KA03": ("outer", "artificial", 2, "electric engraver", "MTK"),
    "KA05": ("outer", "artificial", 1, "electric engraver", "IBU"),
    "KA06": ("outer", "artificial", 2, "electric engraver", "IBU"),
    "KA07": ("outer", "artificial", 1, "drilling", "IBU"),
    "KA08": ("outer", "artificial", 2, "drilling", "IBU"),
    "KA09": ("outer", "artificial", 2, "drilling", "IBU"),
    "KA04": ("outer", "real", 1, "pitting", "FAG"),
    "KA15": ("outer", "real", 1, "indentation", "FAG"),
    "KA16": ("outer", "real", 2, "pitting", "MTK"),
    "KA22": ("outer", "real", 1, "pitting", "IBU"),
    "KA30": ("outer", "real", 1, "indentation", "MTK"),
    "KI01": ("inner", "artificial", 1, "EDM", "MTK"),
    "KI03": ("inner", "artificial", 1, "electric engraver", "MTK"),
    "KI05": ("inner", "artificial", 1, "electric engraver", "IBU"),
    "KI07": ("inner", "artificial", 2, "electric engraver", "IBU"),
    "KI08": ("inner", "artificial", 2, "electric engraver", "IBU"),
    "KI04": ("inner", "real", 1, "pitting", "MTK"),
    "KI14": ("inner", "real", 1, "pitting", "MTK"),
    "KI16": ("inner", "real", 3, "pitting", "FAG"),
    "KI17": ("inner", "real", 1, "pitting", "MTK"),
    "KI18": ("inner", "real", 2, "pitting", "MTK"),
    "KI21": ("inner", "real", 1, "pitting", "FAG"),
    "KB23": ("inner+outer", "real", 2, "pitting", "MTK"),
    "KB24": ("inner+outer", "real", 3, "pitting", "FAG"),
    "KB27": ("inner+outer", "real", 1, "indentation", "FAG"),
}

# --- sha256 与字节数（bearing-datasets/files.lock.json，已与官网索引的尺寸逐项核对）---
# 键为轴承号；值为 (字节数, sha256)。32 项全部来自 bearing-datasets/files.lock.json，
# 已与官方索引页的尺寸逐项核对（官方只给 MB 级粗粒度，这里给字节级）。
SHA256 = {
    "K001": (173881721, "0f119ebdb28fb2f4d9fac1beb1319429f63f7ae1256c23c872f280f3560918e5"),
    "K002": (161981588, "1040da8b74d169c4e8c8545afa335d7a3a320bcaf36a471250c3e434bc4caffd"),
    "K003": (173194389, "d7d1edf6047fac088efe118198abd0856c5280dd8653496fbac9a5d877180f1e"),
    "K004": (163337851, "97242542046cf3ce286c13e12d8df11f618a53365efe592750e2047256345333"),
    "K005": (167366765, "2faf0ee4ee66055bcc066e5b6d9ed9fc44638b2c1ea397a279261103aa83caaa"),
    "K006": (175997605, "250b8e0ef18082163ae4003cd7f50e6daa3061b3a095e226378910da55b78b94"),
    "KA01": (166571438, "6a6be1e11132730cc6f560d51eacedcbfd5fd74b829e9d8d3728c6c8a7cd4c0e"),
    "KA03": (176196084, "21f188899e46183a29752e5dbc830e3132e0fee4d7eb5507450d1730a5b264b5"),
    "KA04": (180920703, "6fc317157e2cd4cb55bc54d6fac1b1753f8ab31abcccd1ac68cbfb78e5e8299d"),
    "KA05": (163644977, "880ad18fae8f1deef2c99643cb74e2dbd284cdf1fe490caff9418717729c4dd1"),
    "KA06": (164139086, "a0f3e8b167b1a1cee63f904566dff6ee8b0d905ffb67796809dbfc7f279c5f60"),
    "KA07": (160346192, "0c081e1fe273d17cf26f052bd99c742fc9a1fcbbc76667b4c0c394f3a8aaf95b"),
    "KA08": (169188540, "aa5c16a379859f67018e23818d671bd12a5c820eb75d02d3a037a16b8a06fd7f"),
    "KA09": (171467156, "6a4fb9b05c95324d9f4ee38d3855106dba21052309f25c30c866dbae9305ce9b"),
    "KA15": (164268273, "2914d003a7a7a23425a825f03f09b2dede8e0f53c266e0b3ad90aa6b98958b7b"),
    "KA16": (167165704, "3e300d2e3402b47a18ce9c75694d3146f6e4b5f86bcd1d159f4938221fb261b6"),
    "KA22": (159054470, "95790035ebe5a69bb298f93a7f4fc64769f0baeffd04a2cbb3ff5b4af8608642"),
    "KA30": (163625195, "cbe1810281a7cbd28d5ad0372cc392c910744709d46d4db501a45eeea18dd518"),
    "KB23": (170933916, "a07e976b285438c33492650c4f5b28250a49e7855990191b30b6bcb541831c6c"),
    "KB24": (186677308, "ae5928ee2ed02773e3787aeceaae73074b15ff3ae158cd932c7a6865fbc9c1ff"),
    "KB27": (162452979, "738b9c332b31038ab051f60a7e1ceec5612f28449254cccda6154ae14655e4aa"),
    "KI01": (175265779, "b1dd6d99bb64d556f889fefaedb7e6e672900f5f015125615feaae776f055348"),
    "KI03": (159217524, "725010ee4158256eef9ef7f58da835fc2150579d6e97b5d48bcd2ee70795d77e"),
    "KI04": (172644279, "8fde861a9ceac4b41fe8a030b4c5bbb4b9b9af0590d9338a8de0150b4661be77"),
    "KI05": (160359281, "5a936f170f4188e38c537b05a260a3cf60cc3b671f90502a008636aac6c66bf4"),
    "KI07": (160646876, "5f3089d19db77ae071a6731198ef116c800e03d3628316086f4672d029d5a2ed"),
    "KI08": (163760131, "f00d6dca7360ed6ec30b78c0e186161913852db4c209551975c5a0998447d514"),
    "KI14": (163581312, "16911b5f8c723b568b6aa69397b00796fea41e43bf627a1a0158a7294f2b7a60"),
    "KI16": (166348305, "f7f4ce4feffc3ae799f92289cdea7708d2cf6e154ad5382f9f4c2c8bed890c44"),
    "KI17": (167363476, "d80dbdb612cfaeb7ba1ffe6387c52c4d938084b2466e55d96a5528aff4d8b084"),
    "KI18": (161226797, "37b9b17c392d033f0b14c35bf0fd19ea878909299cdc8c517e2226f794048615"),
    "KI21": (164882839, "1e3c563fa1d1564e3586736189b2e0807688389b549944af4a06908485c1d5bf"),
}

STAGES = {
    "P0": ["K001", "KA04", "KI04"],
    "P1": [b for b, f in BEARING_FACTS.items() if f[0] == "normal" or f[1] == "real"],
    "P2": [b for b, f in BEARING_FACTS.items() if f[1] == "artificial"],
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(bearing: str, retries: int = 4) -> Path:
    """下载单个 `.rar`，支持断点续传；完成后校验 sha256（有记录时）。

    返回最终路径。已存在且校验通过的文件直接跳过。
    """
    name = f"{bearing}.rar"
    dest = RAW_DIR / name
    expect = SHA256.get(bearing)

    if dest.is_file() and expect and dest.stat().st_size == expect[0]:
        if _sha256(dest) == expect[1]:
            print(f"[skip] {name} 已存在且校验通过")
            return dest
        print(f"[warn] {name} 尺寸对但 sha256 不符，重新下载")

    url = BASE_URL + name
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.is_file() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        mode = "ab" if have else "wb"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                total = have + int(response.headers.get("Content-Length", 0))
                started, done = time.time(), have
                with dest.open(mode) as fh:
                    for block in iter(lambda: response.read(1 << 20), b""):
                        fh.write(block)
                        done += len(block)
                        elapsed = time.time() - started
                        if elapsed > 0 and done % (8 << 20) < (1 << 20):
                            rate = (done - have) / elapsed / 1048576
                            print(f"  {name} {done/1048576:7.1f}/{total/1048576:.0f} MiB"
                                  f"  {rate:.2f} MiB/s", end="\r", flush=True)
            print()
        except Exception as error:  # noqa: BLE001 - 网络异常一律重试，打印原因
            print(f"[retry {attempt}/{retries}] {name}: {type(error).__name__} {error}")
            time.sleep(5 * attempt)
            continue

        if expect is None:
            print(f"[ok] {name} 下载完成（无 sha256 记录，未校验）")
            return dest
        if dest.stat().st_size != expect[0]:
            print(f"[retry {attempt}/{retries}] {name} 尺寸 {dest.stat().st_size} != {expect[0]}")
            continue
        if _sha256(dest) != expect[1]:
            print(f"[retry {attempt}/{retries}] {name} sha256 不符")
            continue
        print(f"[ok] {name} {expect[0]/1048576:.1f} MiB sha256 校验通过")
        return dest

    raise RuntimeError(f"{name} 下载失败（已重试 {retries} 次）")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 Paderborn (KAt) 轴承数据")
    parser.add_argument("stage", choices=sorted(STAGES) + ["all"])
    parser.add_argument("--verify", action="store_true", help="只校验已下文件，不下载")
    args = parser.parse_args()

    bearings = sorted(BEARING_FACTS) if args.stage == "all" else STAGES[args.stage]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"阶段 {args.stage}：{len(bearings)} 个轴承 -> {RAW_DIR}")
    known = sum(SHA256[b][0] for b in bearings if b in SHA256)
    print(f"已记录尺寸的合计约 {known/1073741824:.2f} GB\n")

    failures = []
    for index, bearing in enumerate(bearings, 1):
        fault = BEARING_FACTS[bearing]
        print(f"[{index}/{len(bearings)}] {bearing}  {fault[0]} / {fault[1]} / L{fault[2]} / {fault[4]}")
        dest = RAW_DIR / f"{bearing}.rar"
        if args.verify:
            if not dest.is_file():
                print("  [missing]")
                failures.append(bearing)
            elif bearing in SHA256:
                ok = dest.stat().st_size == SHA256[bearing][0] and _sha256(dest) == SHA256[bearing][1]
                print(f"  [{'ok' if ok else 'BAD'}] {dest.stat().st_size/1048576:.1f} MiB")
                if not ok:
                    failures.append(bearing)
            continue
        try:
            download(bearing)
        except Exception as error:  # noqa: BLE001
            print(f"  [FAIL] {error}")
            failures.append(bearing)

    if failures:
        print(f"\n未完成：{failures}")
        return 1
    print("\n全部完成。下一步：解包 -> 校验 .mat 结构与通道名。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
