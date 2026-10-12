#!/usr/bin/env bash
# v7 非训练准备流水线（顺序脚本，不是调度器）：
#   数据事实自检 -> 单元测试 -> 特征缓存 -> PU 选标候选 -> 零训练物理基线。
# 不含任何训练、适配或 BN 重估入口；每步幂等（已有产物明确跳过，不覆盖、不静默重跑）。
# 单步失败时按 agents.md 单独重跑该步，产物留 _retry1 后缀，不重跑整条脚本。
set -euo pipefail

PYTHON="${PYTHON:-E:/tool/anaconda/envs/fault-detect/python.exe}"

# 根目录定位：本脚本所在目录的父目录即 v7 工作根
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# 与选标 k-means 的单线程纪律一致（Windows 小样本稳定复现）
export OMP_NUM_THREADS=1

echo "==[1/5] bearing_facts.py：32 轴承事实表自检"
"$PYTHON" bearing_facts.py

echo "==[2/5] unittest discover tests/：数据层与规则单元测试"
"$PYTHON" -m unittest discover -s tests -v

echo "==[3/5] features.py --dataset all：logspec512 特征缓存"
# features.py 对已有缓存幂等（命中即只读加载并校验口径版本），重复执行安全
"$PYTHON" features.py --dataset all

echo "==[4/5] selection.py：PU 116 单元无标签选标候选"
if [ -f outputs/selection/candidate.json ]; then
  echo "已存在 outputs/selection/candidate.json，明确跳过（selection.py 本身也拒绝覆盖）。"
  echo "若需带显式预算的正式 selection，请人工确认后另行指定 --budgets 生成正式文件。"
else
  "$PYTHON" selection.py
fi

echo "==[5/5] physical_baseline.py：零训练包络谱物理基线（D8）"
if [ -d outputs/eval/phys_envelope ] && [ -n "$(ls -A outputs/eval/phys_envelope 2>/dev/null)" ]; then
  echo "已存在 outputs/eval/phys_envelope 且非空，明确跳过（physical_baseline.py 对非空输出目录也拒绝重跑）。"
else
  if [ -f outputs/selection/selection.json ]; then
    "$PYTHON" physical_baseline.py --selection outputs/selection/selection.json \
        --output outputs/eval/phys_envelope
  elif [ -f outputs/selection/candidate.json ]; then
    # 优先正式清单；只有未冻结时才用候选清单，不用池真值调规则
    "$PYTHON" physical_baseline.py --selection outputs/selection/candidate.json \
        --output outputs/eval/phys_envelope
  else
    "$PYTHON" physical_baseline.py --output outputs/eval/phys_envelope
  fi
fi

echo "==非训练准备全部完成；训练（train.py --execute）不在本脚本范围内。=="
