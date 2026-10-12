#!/usr/bin/env bash
# v7 云端一键训练与评估（Cloud Studio 单 GPU 工作区，T4/A10 均可）。
# 固定矩阵的薄封装，不是调度器：12 个 joint 先行 -> 84 个后续臂 -> 108 份冻结评估。
# 并行机制：同一块 GPU 上跑多个独立训练进程（每格一个小模型，单进程喂不满卡），
#           每进程独立固定种子；CUDA/LSTM 逐位复现仍须在实际环境验证。
# 用法：
#   bash scripts/run_cloud.sh check    # 环境/v6 依赖/特征缓存/dry-run 自检
#   bash scripts/run_cloud.sh train    # 96 格训练（仅跳过成功完成且产物齐全的格）
#   bash scripts/run_cloud.sh eval     # 108 份评估（96 份 PU + 12 份 joint 的 CWRU 源验证）
#   bash scripts/run_cloud.sh all      # check + train + eval
#   bash scripts/run_cloud.sh status   # 看进度
# 环境变量：PAR=并行进程数（默认 2；A10/20核可设 3，T4/8核别超过 3）
#           PYTHON=解释器（默认 python3）
# 注意：train/all 带 --execute；eval 只读前向，不训练或重估 BN。
set -uo pipefail
PYTHON="${PYTHON:-python3}"
PAR="${PAR:-2}"
[[ "$PAR" =~ ^[1-9][0-9]*$ ]] || { echo "PAR 必须是正整数"; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$ROOT" || exit 1
mkdir -p logs outputs/train outputs/eval || exit 1
export OMP_NUM_THREADS=1
# PyTorch 2.6 的 LSTM 文档要求 CUDA 10.2+ 在启动进程前设置工作区。
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:16:8}"
ARMS=(joint source_continue adabn mmd_marginal ce_K19 ce_K38 ce_K116 ce_mmd)

log() { echo "[$(date +%H:%M:%S)] $*"; }
cell_done() {
  local dir="outputs/train/$1" f
  [ -f "$dir/.complete" ] || return 1
  for f in config.json split.json last.pt best.pt history.csv; do
    [ -s "$dir/$f" ] || return 1
  done
}
eval_done() {
  local dir="outputs/eval/$1" f
  [ -f "$dir/.complete" ] || return 1
  for f in metrics.json eval_info.json predictions.csv unit_predictions.csv; do
    [ -s "$dir/$f" ] || return 1
  done
}

check() {
  "$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('设备:', torch.cuda.get_device_name(0), '| torch', torch.__version__)" || {
    echo "CUDA 检查失败，禁止开训；请检查 GPU 和 CUDA 版 torch"; return 1; }
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
    cat outputs/features/paderborn.npy.part-* > outputs/features/paderborn.npy.tmp &&
      mv outputs/features/paderborn.npy.tmp outputs/features/paderborn.npy || {
        echo "特征分卷合并失败，未发布主缓存"; return 1; }
  fi
  ( cd outputs/features && md5sum -c --status checksums.md5 ) || {
    echo "特征缓存 md5 校验失败，不要使用，重新 clone 或重新分卷"; exit 1; }
  # joint dry-run：完整校验冻结配置、selection、双数据集缓存与 153/116 口径（不建模型、不训练）
  rm -rf logs/cloud_check || return 1   # scratch 目录，避免重复运行时被“非空目录”检查挡住
  "$PYTHON" train.py --arm joint --fold 0 --seed 0 --dry-run --output logs/cloud_check >/dev/null || {
    echo "dry-run 校验未通过，见上方报错"; exit 1; }
  # 无训练单元测试；保存日志便于查看具体失败项。
  if ! "$PYTHON" -m unittest discover -s tests > logs/cloud_tests.log 2>&1; then
    echo "单元测试未全部通过，见 logs/cloud_tests.log；先解决环境再开训"; exit 1
  fi
  log "自检通过：CUDA / v6 目录 / 特征缓存 / selection / 冻结口径 全部就绪"
}

# 内部子命令：_cell <arm> <fold> <seed>
_cell() {
  local arm="$1" fold="$2" seed="$3" cell="${1}_f${2}_s${3}"
  if cell_done "$cell"; then log "跳过 $cell（已成功完成，产物齐全）"; return 0; fi
  # 上次失败残留的目录挪走留现场（不删除），保证本格能重跑。
  if [ -d "outputs/train/$cell" ]; then
    mv "outputs/train/$cell" "outputs/train/${cell}_failed_$(date +%Y%m%d%H%M%S)_$$" || return 1
  fi
  local t0=$SECONDS
  local args=(--config configs/stage2.json --arm "$arm" --fold "$fold" --seed "$seed"
              --output "outputs/train/$cell" --execute)
  [ "$arm" != joint ] && args+=(--source-checkpoint "outputs/train/joint_f${fold}_s${seed}/last.pt")
  log "开始 $cell"
  if "$PYTHON" train.py "${args[@]}" &&
      touch "outputs/train/$cell/.complete" && cell_done "$cell"; then
    log "完成 $cell，耗时 $((SECONDS - t0)) 秒"
  else
    log "失败 $cell（现场在 outputs/train/$cell；再次重跑时归档后从头训练本格）"
    echo "$cell" >> logs/failed_train.txt
    return 1
  fi
}

# 内部子命令：_evalcell <cell> <dataset>
_evalcell() {
  local cell="$1" dataset="$2" dir="outputs/eval/${1}_${2}"
  if eval_done "${cell}_${dataset}"; then log "跳过评估 $dir"; return 0; fi
  if [ -d "$dir" ]; then mv "$dir" "${dir}_failed_$(date +%Y%m%d%H%M%S)_$$" || return 1; fi
  local t0=$SECONDS
  log "开始评估 $cell（$dataset）"
  if "$PYTHON" evaluate.py --checkpoint "outputs/train/$cell/last.pt" --dataset "$dataset" --output "$dir" &&
      touch "$dir/.complete" && eval_done "${cell}_${dataset}"; then
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
  rm -f logs/failed_train.txt || return 1
  local jobs=() fold seed arm
  for fold in 0 1 2 3; do for seed in 0 1 2; do jobs+=("joint $fold $seed"); done; done
  log "== 阶段 1/2：12 个 joint（84 个后续臂的依赖）=="
  run_jobs _cell "${jobs[@]}" || {
    log "joint 阶段失败，停止后续臂；查看日志并重跑 train"; return 1; }
  jobs=()
  for fold in 0 1 2 3; do for seed in 0 1 2; do for arm in "${ARMS[@]:1}"; do
    jobs+=("$arm $fold $seed")
  done; done; done
  log "== 阶段 2/2：84 个后续臂（依赖同折同种子 joint 完整末轮）=="
  run_jobs _cell "${jobs[@]}" || {
    log "后续臂阶段失败，停止评估；查看日志并重跑 train"; return 1; }
  if [ -s logs/failed_train.txt ]; then
    log "训练有失败格：$(paste -sd, logs/failed_train.txt)"
    return 1
  fi
  log "训练 96/96 全部完成"
}

eval_all() {
  rm -f logs/failed_eval.txt || return 1
  local jobs=() fold seed arm
  for fold in 0 1 2 3; do for seed in 0 1 2; do
    for arm in "${ARMS[@]}"; do
      cell_done "${arm}_f${fold}_s${seed}" || {
        log "训练未全部完成：${arm}_f${fold}_s${seed}；请先重跑 train"; return 1; }
    done
    jobs+=("joint_f${fold}_s${seed} paderborn" "joint_f${fold}_s${seed} cwru")
    for arm in "${ARMS[@]:1}"; do jobs+=("${arm}_f${fold}_s${seed} paderborn"); done
  done; done
  log "== 评估：108 份（96 份 PU 主口径 + 12 份 joint 的 CWRU 源验证）=="
  run_jobs _evalcell "${jobs[@]}" || {
    log "评估阶段失败，查看日志并重跑 eval"; return 1; }
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
  all) check && train_all && eval_all ;;
  status) status ;;
  _cell) shift; _cell "$@" ;;
  _evalcell) shift; _evalcell "$@" ;;
  *) echo "用法：bash scripts/run_cloud.sh {check|train|eval|all|status}"; exit 2 ;;
esac
