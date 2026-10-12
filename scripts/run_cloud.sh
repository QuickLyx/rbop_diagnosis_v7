#!/usr/bin/env bash
# v7 云端一键训练与评估（Cloud Studio 单 GPU 工作区，T4/A10 均可）。
# 固定矩阵的薄封装，不是调度器：12 个 joint 先行 -> 84 个后续臂 -> 108 份冻结评估。
# 并行机制：同一块 GPU 上跑多个独立训练进程（每格一个小模型，单进程喂不满卡），
#           每进程独立固定种子 + cuDNN 确定性算法，并行不影响每格结果。
# 用法：
#   bash scripts/run_cloud.sh check    # 环境/v6 依赖/特征缓存/dry-run 自检
#   bash scripts/run_cloud.sh train    # 96 格训练（续跑安全：已有 last.pt 的格自动跳过）
#   bash scripts/run_cloud.sh eval     # 108 份评估（96 份 PU + 12 份 joint 的 CWRU 源验证）
#   bash scripts/run_cloud.sh all      # check + train + eval
#   bash scripts/run_cloud.sh status   # 看进度
# 环境变量：PAR=并行进程数（默认 2；A10/20核可设 3，T4/8核别超过 3）
#           PYTHON=解释器（默认 python3）
# 注意：train/eval/all 全部带 --execute 启动训练；在云端运行即视为已获训练授权。
set -uo pipefail
PYTHON="${PYTHON:-python3}"
PAR="${PAR:-2}"
[ "$PAR" -ge 1 ] || PAR=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/train outputs/eval
export OMP_NUM_THREADS=1
ARMS=(joint source_continue adabn mmd_marginal ce_K19 ce_K38 ce_K116 ce_mmd)

log() { echo "[$(date +%H:%M:%S)] $*"; }
cell_done() { [ -f "outputs/train/$1/last.pt" ]; }
eval_done() { [ -f "outputs/eval/$1/metrics.json" ]; }

check() {
  $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('设备:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
  [ -f ../rbop_diagnosis_v6/model.py ] || {
    echo "缺少只读 v6 仓库：../rbop_diagnosis_v6/（含 Data/ 与 3 个被隔离加载的模块）"; exit 1; }
  local f
  for f in outputs/features/cwru.json outputs/features/paderborn.json \
           outputs/selection/selection.json outputs/features/checksums.md5; do
    [ -f "$f" ] || { echo "缺少 $f"; exit 1; }
  done
  # paderborn.npy 234MB 超单文件限制，随仓库走分卷；缺主文件时自动合并
  if [ ! -f outputs/features/paderborn.npy ]; then
    log "从分卷合并 paderborn.npy ..."
    cat outputs/features/paderborn.npy.part-* > outputs/features/paderborn.npy
  fi
  ( cd outputs/features && md5sum -c --status checksums.md5 ) || {
    echo "特征缓存 md5 校验失败，不要使用，重新 clone 或重新分卷"; exit 1; }
  # joint dry-run：完整校验冻结配置、selection、双数据集缓存与 153/116 口径（不建模型、不训练）
  rm -rf logs/cloud_check   # scratch 目录，避免重复运行时被“非空目录”检查挡住
  $PYTHON train.py --arm joint --fold 0 --seed 0 --dry-run --output logs/cloud_check >/dev/null || {
    echo "dry-run 校验未通过，见上方报错"; exit 1; }
  # 无训练单元测试（约 5 秒）：提前暴露云端 numpy/scipy 版本问题；用退出码判断
  if ! $PYTHON -m unittest discover -s tests >/dev/null 2>&1; then
    echo "单元测试未全部通过，先解决环境再开训（复现：$PYTHON -m unittest discover -s tests）"; exit 1
  fi
  log "自检通过：CUDA / v6 目录 / 特征缓存 / selection / 冻结口径 全部就绪"
}

# 内部子命令：_cell <arm> <fold> <seed>
_cell() {
  local arm="$1" fold="$2" seed="$3" cell="${1}_f${2}_s${3}"
  if cell_done "$cell"; then log "跳过 $cell（已有 last.pt）"; return 0; fi
  # 上次失败残留的目录挪走留现场（不删除），保证本格能重跑
  if [ -d "outputs/train/$cell" ]; then
    mv "outputs/train/$cell" "outputs/train/${cell}_failed_$(date +%m%d%H%M%S)"
  fi
  local t0=$SECONDS
  local args=(--config configs/stage2.json --arm "$arm" --fold "$fold" --seed "$seed"
              --output "outputs/train/$cell" --execute)
  [ "$arm" != joint ] && args+=(--source-checkpoint "outputs/train/joint_f${fold}_s${seed}/last.pt")
  log "开始 $cell"
  if $PYTHON train.py "${args[@]}"; then
    log "完成 $cell，耗时 $((SECONDS - t0)) 秒"
  else
    log "失败 $cell（现场在 outputs/train/${cell}_failed_*，重跑 train 会自动续上其余格）"
    echo "$cell" >> logs/failed_train.txt
    return 1
  fi
}

# 内部子命令：_evalcell <cell> <dataset>
_evalcell() {
  local cell="$1" dataset="$2" dir="outputs/eval/${1}_${2}"
  if eval_done "$dir"; then log "跳过评估 $dir"; return 0; fi
  if [ -d "$dir" ]; then mv "$dir" "${dir}_failed_$(date +%m%d%H%M%S)"; fi
  local t0=$SECONDS
  log "开始评估 $cell（$dataset）"
  if $PYTHON evaluate.py --checkpoint "outputs/train/$cell/last.pt" --dataset "$dataset" --output "$dir"; then
    log "完成评估 $dir，耗时 $((SECONDS - t0)) 秒"
  else
    log "评估失败 $dir"; echo "$dir" >> logs/failed_eval.txt
    return 1
  fi
}

# run_jobs <子命令> <任务行...>：每行一个任务的空格分隔参数，经 xargs 并行
run_jobs() {
  local sub="$1"; shift
  [ "$#" -eq 0 ] && return 0
  printf '%s\n' "$@" | xargs -P "$PAR" -L 1 bash "$ROOT/scripts/run_cloud.sh" "$sub"
}

train_all() {
  rm -f logs/failed_train.txt
  local jobs=() fold seed arm
  for fold in 0 1 2 3; do for seed in 0 1 2; do jobs+=("joint $fold $seed"); done; done
  log "== 阶段 1/2：12 个 joint（84 个后续臂的依赖）=="
  run_jobs _cell "${jobs[@]}"
  jobs=()
  for fold in 0 1 2 3; do for seed in 0 1 2; do for arm in "${ARMS[@]:1}"; do
    jobs+=("$arm $fold $seed")
  done; done; done
  log "== 阶段 2/2：84 个后续臂（依赖同折同种子 joint 完整末轮）=="
  run_jobs _cell "${jobs[@]}"
  if [ -s logs/failed_train.txt ]; then
    log "训练有失败格：$(paste -sd, logs/failed_train.txt)"
    return 1
  fi
  log "训练 96/96 全部完成"
}

eval_all() {
  rm -f logs/failed_eval.txt
  local jobs=() fold seed arm ds
  for fold in 0 1 2 3; do for seed in 0 1 2; do
    jobs+=("joint_f${fold}_s${seed} paderborn" "joint_f${fold}_s${seed} cwru")
    for arm in "${ARMS[@]:1}"; do jobs+=("${arm}_f${fold}_s${seed} paderborn"); done
  done; done
  log "== 评估：108 份（96 份 PU 主口径 + 12 份 joint 的 CWRU 源验证）=="
  run_jobs _evalcell "${jobs[@]}"
  if [ -s logs/failed_eval.txt ]; then
    log "评估有失败项：$(paste -sd, logs/failed_eval.txt)"
    return 1
  fi
  log "评估全部完成，产物在 outputs/eval/"
}

status() {
  local t=0 d=0 fold seed arm ds
  for fold in 0 1 2 3; do for seed in 0 1 2; do for arm in "${ARMS[@]}"; do
    t=$((t+1)); cell_done "${arm}_f${fold}_s${seed}" && d=$((d+1))
  done; done; done
  echo "训练：$d/$t 格完成（起点权重 12 joint -> 84 后续臂）"
  t=0; d=0
  for fold in 0 1 2 3; do for seed in 0 1 2; do
    t=$((t+1)); eval_done "joint_f${fold}_s${seed}_paderborn" && d=$((d+1))
    t=$((t+1)); eval_done "joint_f${fold}_s${seed}_cwru" && d=$((d+1))
    for arm in "${ARMS[@]:1}"; do
      t=$((t+1)); eval_done "${arm}_f${fold}_s${seed}_paderborn" && d=$((d+1))
    done
  done; done
  echo "评估：$d/$t 份完成（96 PU + 12 CWRU）"
  [ -s logs/failed_train.txt ] && echo "训练失败格：$(paste -sd, logs/failed_train.txt)"
  [ -s logs/failed_eval.txt ] && echo "评估失败项：$(paste -sd, logs/failed_eval.txt)"
  return 0
}

case "${1:-}" in
  check) check ;;
  train) train_all ;;
  eval) eval_all ;;
  all) check; train_all; eval_all ;;
  status) status ;;
  _cell) shift; _cell "$@" ;;
  _evalcell) shift; _evalcell "$@" ;;
  *) echo "用法：bash scripts/run_cloud.sh {check|train|eval|all|status}"; exit 2 ;;
esac
