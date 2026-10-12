#!/usr/bin/env bash
# 固定矩阵顺序执行；默认只做无模型 dry-run，显式 --execute 才启动训练。
set -euo pipefail
PYTHON="${PYTHON:-E:/tool/anaconda/envs/fault-detect/python.exe}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1

MODE="--dry-run"
OUTPUT_ROOT="outputs/preflight"
if [ "${1:-}" = "--execute" ]; then
  MODE="--execute"
  OUTPUT_ROOT="outputs/train"
elif [ "$#" -gt 0 ]; then
  echo "用法：bash scripts/run_matrix.sh [--execute]；默认不训练。" >&2
  exit 2
fi
if [ "$#" -gt 1 ]; then
  echo "只接受一个可选参数 --execute。" >&2
  exit 2
fi
ARMS=(joint source_continue adabn mmd_marginal ce_K19 ce_K38 ce_K116 ce_mmd)
for fold in 0 1 2 3; do
  for seed in 0 1 2; do
    for arm in "${ARMS[@]}"; do
      args=(--config configs/stage2.json --arm "$arm" --fold "$fold" --seed "$seed"
            --output "$OUTPUT_ROOT/${arm}_f${fold}_s${seed}" "$MODE")
      if [ "$arm" != "joint" ]; then
        args+=(--source-checkpoint "outputs/train/joint_f${fold}_s${seed}/last.pt")
      fi
      "$PYTHON" train.py "${args[@]}"
    done
  done
done
echo "固定矩阵检查/执行完成，模式：$MODE。失败格请单独重跑并使用 _retry1 输出目录。"
