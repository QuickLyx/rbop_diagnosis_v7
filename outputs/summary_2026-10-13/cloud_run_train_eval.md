# 阶段二全批执行记录：96 格训练 + 108 份评估（2026-10-13）

固定矩阵、判据、池与配对链均未改动，本记录只写实际执行与已落盘数字。

## 命令、环境与时间

- 命令：`nohup bash scripts/run_cloud.sh all > logs/cloud_all.log 2>&1 &`（`all` = `check` + `train` + `eval`）
- 环境：Cloud Studio / Tesla T4（16 GB）、torch 2.6.0+cu124；`OMP_NUM_THREADS=1`、`CUBLAS_WORKSPACE_CONFIG=:16:8`
- 并行度：3（`logs/cloud_all.log` 实测同时最多 3 个训练格进程；3 进程合计显存约 1.0 GB）
- 时间：07:48:17 自检通过启动 → **训练 96/96 收尾 14:43:19** → **评估 108/108 收尾 14:58:52**；
  训练段 6 小时 55 分，评估段再 15 分，整批墙钟约 7 小时 11 分

## 完成情况与产物

- 训练 96/96：每格 `.complete` + `config.json`/`split.json`/`last.pt`/`best.pt`/`history.csv`，
  `outputs/train/` 整目录 744 MB（不入库）；正式权重取末轮 `last.pt`（`epoch=checkpoint_epoch=last_epoch=30`）
- 评估 108/108：96 份 PU 主口径 + 12 份 joint 的 CWRU 源验证，每份 `.complete` + `metrics.json`/
  `eval_info.json`/`predictions.csv`/`unit_predictions.csv`，`outputs/eval/` 整目录 131 MB（需回传）
- `logs/failed_train.txt`、`logs/failed_eval.txt` 均不存在 → **本轮无失败格，无需 `_retry1` 重跑**
- 归档现场：`outputs/train/joint_*_failed_<时间戳>_<进程号>` 共 24 个，来自本批之前 joint 的两次中断尝试
  （07:43 一批只落了 `config.json`/`split.json`；07:44 一批跑到约第 4 轮被中断），其中 12 个留有部分
  `last.pt`，**不是完成格**；成功的一批即日志所记录的 07:48:17 起跑
- 单格训练耗时（每臂 12 格均值，秒）：`adabn` 142、`source_continue` 438、`ce_K116` 740、`ce_K38` 763、
  `ce_K19` 767、`mmd_marginal` 777、`joint` 1246、`ce_mmd` 1295；单份评估 7–32 s

## 评估产物 D3 字段汇总（96 份 PU，未重新计算判据）

| 臂 | D3.E1 均值 | D3.E2 均值 | P_ball 均值 | 过线格 |
| --- | ---: | ---: | ---: | ---: |
| `ce_K116` | 96.90% | 96.90% | 0.00% | 11/12（in-sample，无留出） |
| `ce_K38` | 69.61%（自然留出 78 单元 56.74%） | 69.61% | 0.00% | 0/12 |
| `ce_mmd` | 50.14%（自然留出 97 单元 42.93%） | 50.14% | 0.00% | 0/12 |
| `ce_K19` | 49.79%（自然留出 97 单元 42.55%） | 49.79% | 0.00% | 0/12 |
| `joint` | 27.78% | 28.33% | 2.16% | 0/12 |
| `source_continue` | 26.70% | 27.43% | 1.87% | 0/12 |
| `mmd_marginal` | 20.51% | 24.98% | 16.02% | 0/12 |
| `adabn` | 15.69% | 21.18% | 31.47% | 0/12 |

- 口径：`D3.E1/E2` 为单元级、按真实支持类（Normal/IR/OR）宏平均，`P_ball` 为单元级均值。
  四个零标注臂（`joint`/`source_continue`/`adabn`/`mmd_marginal`）没有监督池，自然留出就是全部
  116 单元，表中数字即留出值；`ce_K*` 与 `ce_mmd` 的全体 116 单元统计含各自监督池、只作描述，
  括号内的留出数字取自 `splits.natural_heldout`（`ce_mmd`/`ce_K19` 为 97 单元、`ce_K38` 为 78 单元）。
- 判据（E1 ≥ 80% 且 E2 ≥ 90%）下 **96 格仅 11 格过线且全部属于 `ce_K116`**；K116 全标、无留出单元，
  按协议只作全监督天花板参照，**不作为泛化过线证据**。留出集上最好的是 `ce_K38`（78 单元 56.74%）。
- 最好单格：`ce_K116_f0_s0`，E1 = E2 = 99.24%（in-sample）。

## 未执行（下一步）

零算力错误结构分析（`predictions.csv` 的错分类别/工况/损伤来源、是否整档塌成一类、`P_ball` 占比）、
六对配对链比较、分组并报（工况 / 损伤来源 / 未见轴承）、KB 事后分析。
按 `agents.md`：不临时加臂、不临时改判据、不加新特征。
