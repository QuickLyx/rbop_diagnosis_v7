# rbop_diagnosis_v7

源域 = CWRU 排 028 后 153 文件（四类）；目标域 = Paderborn / KAt（PU，三类 + 无真值 KB）。
零目标标注外部测试 + 少样本适配。方案与全部冻结决策见 `docs/阶段二方案_v1.md`，
数据事实见 `docs/数据事实_CWRU与Paderborn.md`，工程规则见 `agents.md`。

## 当前状态（2026-10-13）

固定矩阵已在云端执行完毕（Cloud Studio / Tesla T4，torch 2.6.0+cu124，
`nohup bash scripts/run_cloud.sh all > logs/cloud_all.log 2>&1 &`，并行度 3，
07:48:17 自检通过启动，14:58:52 收尾，墙钟约 7 小时 11 分）：

- **训练 96/96 格完成**（14:43:19）：每格带 `.complete` 与五件产物
  （`config.json`/`split.json`/`last.pt`/`best.pt`/`history.csv`），`outputs/train/` 整目录 744 MB；
  `logs/failed_train.txt` 不存在，无失败格。同目录另有 24 个
  `joint_*_failed_<时间戳>_<进程号>` 归档目录（joint 早期两次尝试被中断的现场，其中 12 个留有
  未完成的部分权重），**不是完成格**；
- **评估 108/108 份完成**（14:58:52）：96 份 PU 主口径 + 12 份 joint 的 CWRU 源验证，
  每份带 `.complete` 与四份报告，`outputs/eval/` 整目录 131 MB；`logs/failed_eval.txt` 不存在；
- 逐臂耗时、单份评估耗时、命令与环境等执行明细见
  `outputs/summary_2026-10-13/cloud_run_train_eval.md`。

**结果状态（判据未改动）**：96 份 PU 评估的 `D3` 字段已落盘，其中 **11 格过线、全部属于
`ce_K116`**——K116 是全标臂、没有留出单元，按协议只算 in-sample 参照，**不作为泛化过线证据**；
其余 85 格未过线，自然留出集上最好的是 `ce_K38`（78 单元宏平均 E1 56.74%）。
零算力错误结构分析、六对配对链比较与分组并报**已于 2026-10-13 执行完毕**，
总结与结论见 `outputs/summary_2026-10-13/results_summary.md`（逐项数字在
`outputs/summary_2026-10-13/error_structure.json`，脚本 `scripts/error_structure.py`）。

## 工作根与只读依赖

- **工作根**：本目录（`rbop_diagnosis_v7/`）。v5、v6 的目录与输出**只读**，不写入、不覆盖。
- **v6 依赖**：`preprocessing.extract_logspec`（logspec512 口径）与 `model.py`（MSRS-ABiLSTM，
  `n_classes=4`，不改）经 `_v6.py` 的 `load_v6()` 隔离加载，路径固定为
  `../rbop_diagnosis_v6/`；CWRU 数据同样直接指到 `../rbop_diagnosis_v6/Data/`
  （`datasets/cwru.py` 的 `DEFAULT_DATA_ROOT`），**不复制数据**。
- **PU 数据**：`Data/paderborn/_raw/`（32 个 .rar）与 `extracted/`（2560 个 .mat）不入库
  （见 `.gitignore`），由 `scripts/fetch_paderborn.py` / `extract_paderborn.py` 重建。
- **解释器**：`E:/tool/anaconda/envs/fault-detect/python.exe`（torch 2.6.0 + cu124，
  与 `requirements.txt` 一致，不引入新依赖）。

## 数据口径（冻结，代码会报错兜底）

- **官方截断文件** `N15_M01_F10_KA08_2.mat`：**整条丢弃**（方案 D12），丢弃落盘记录、不静默跳过。
  单元 `KA08×N15_M01_F10` 按 19 条记录聚合，其余 115 个单元 20 条（D11）。
- **PU 主口径**（29 轴承，排 KB23/24/27）：**116 单元 / 2,319 记录 / 216,350 窗**。
- **全 32 轴承口径**：2,559 记录 / 238,752 窗，**仅用于 KB 事后分析**——KB23/24/27 是复合故障、
  类别由官方按主导损伤人为指派，在四类空间**无真值**（`label=None`），不得进 E1/E2 分母。
- 评估三级聚合：**窗口概率 → 记录级等权平均 → 单元级等权平均**（v6 evaluate.py 纪律；
  单元内记录数 19/20 不符或同单元真值不一致时直接报错）。argmax 在单元级概率上做。

## 评估口径：E1 / E2 / P_ball（D6 冻结）

三个数都在**单元级（116）**上算、宏平均，记录级（2,319）并报：

- **E1**（主判据）：四路 argmax 准确率，判成 Ball 即错；
- **E2**：**三类掩码重判**，在三级聚合后的四路概率中把 Ball 项置零，全部样本重新
  argmax，分母与 E1 相同。这等价于完全排除 Ball 候选；**不能直接把原始负 logit 设为 0**，
  否则 Ball 仍可能胜出。也不是只统计“未判成 Ball 的样本”的条件准确率。
- **P_ball**：4 路 argmax 下被判成 Ball 的比例。

判据（D3，冻结，不得事后改动）：**全体 116 单元宏平均 E1 ≥ 80% 且 E2 ≥ 90%**，并报对标
PU 原论文 75.0%（对 E2 口径）。两条互补的解读规则：

- **全体 116 的阈值是冻结的主判据**，不暗改为 test-only；
- 但全体 116 中含适配池单元（`ce_K{k}` 的池在 116 之内），这部分**只能作描述性统计**，
  **归纳式泛化结论只能下在未参与适配的 test 上**：各预算档的 `budgets[K].test` 指标必须并报
  （physical_baseline 的 `budget_heldout_metrics.json`、evaluate 的分组都按此拆）。
  **AdaBN 与 mmd_marginal 是例外**：用户确认使用全部 116 个单元的无标签信号，属于
  **传导式适配**。其池外指标是标签留出，测试信号已见过，不称完全未见测试。
  极端情形 `K116`（全标）**没有任何留出单元**，只有 in-sample 数字，**不能**作为
  泛化过线证据，只当全监督天花板参照。
- **v6 的 12 个旧权重**（56 文件上训）只能当初始化，**不能充当 153 文件源域基线**
  （源域构成全变，必须重训，方案 §1 代价 3）。

## 非训练准备（训练前的准备步骤，已全部完成）

训练之外的所有步骤一条脚本跑完（顺序 shell，不是调度器框架）：

```bash
bash scripts/prepare_nontraining.sh
```

步骤（每步幂等，已有产物明确跳过、不覆盖不静默重跑）：

1. `python bearing_facts.py` —— 32 轴承事实表自检（任何数据改动后必须跑通）；
2. `python -m unittest discover -s tests -v` —— 数据层 / 物理 / 选标 / 评估 / dry-run 无训练测试；
3. `python features.py --dataset all` —— logspec512 特征缓存（约 245 MB，带口径版本
   `v7_logspec512_12k_drop_ka08_v1`；已有缓存命中时只读加载并校验，过期拒绝静默使用，D9）；
4. `python selection.py` —— PU 116 单元无标签 k-means medoid 选标，生成
   `outputs/selection/candidate.json`（`budgets_status=pending`，**不覆盖已有文件**）；
5. `python physical_baseline.py --selection outputs/selection/selection.json
   --output outputs/eval/phys_envelope` —— 零训练包络谱物理下界（D8）；
   优先使用正式清单；尚未冻结时脚本才回退到 `candidate.json`，只读 `budgets[K].test` 算留出格指标。

环境变量可覆盖解释器：`PYTHON=/path/to/python bash scripts/prepare_nontraining.sh`。
单步失败：按 `agents.md` 单独重跑该步并留 `_retry1` 后缀，不重跑整条脚本。

### 显式自检 / 单步运行

```bash
E:/tool/anaconda/envs/fault-detect/python.exe bearing_facts.py          # 事实表自检
E:/tool/anaconda/envs/fault-detect/python.exe features.py --self-check  # 枚举与抽样自检
E:/tool/anaconda/envs/fault-detect/python.exe physical_baseline.py --self-check
E:/tool/anaconda/envs/fault-detect/python.exe -m unittest discover -s tests -v
```

## 标注预算与 selection

- 正式冻结产物：`outputs/selection/selection.json`，实际 **k=19**；用户确认预算
  **K19、K38、K116**，池/测试分别 **19/97、38/78、116/0**。候选文件保留作选标过程记录。
- 所有池来自同一无标签 medoid 代表序列的嵌套前缀，标签只用于选完后的覆盖诊断。
- `configs/stage2.json` 固定 4 折、3 种子、8 模型臂，共 **96 格**（2026-10-13 已全部训练完成）；物理基线只运行一次。
- AdaBN / mmd_marginal 的无标签信号范围是全部 116 主口径单元，不含 KB，不按目标真值采样。

## 训练与评估（2026-10-13 已执行完毕）

- 云端一条命令跑完全批：`nohup bash scripts/run_cloud.sh all > logs/cloud_all.log 2>&1 &`
  （`all` = `check` + `train` + `eval`，本轮并行度 3）。结果：**训练 96/96、评估 108/108，无失败格**。
- 训练入口带 `--execute` 才会真正开跑。默认或显式 `--dry-run` 只检查计划、
  打印并写 `dry_run.json`，不构造模型、优化器或训练特征副本，不落盘权重。
- 单格 dry-run 的准确命令如下；必须使用空的新输出目录，已有目录非空时改用下一重试后缀。

```bash
E:/tool/anaconda/envs/fault-detect/python.exe train.py --help
E:/tool/anaconda/envs/fault-detect/python.exe train.py --arm joint --fold 0 --seed 0 \
  --selection outputs/selection/selection.json --dry-run \
  --output outputs/preflight/joint_f0_s0_retry1
```

`bash scripts/run_matrix.sh` 默认检查全部 96 格；`outputs/preflight/` 中已有这 96 份计划，
直接重跑会拒绝覆盖，无需再跑。12 个 joint 与 84 个后续臂的 v7 完整末轮权重均已落盘
（`outputs/train/`，不入库），因此再跑 `train` / `eval` 只会跳过带 `.complete` 的格，是空操作。
执行明细与实测耗时见 `outputs/summary_2026-10-13/cloud_run_train_eval.md`。

正式评估只接受 v7 的 `last`，`epoch`、`checkpoint_epoch`、`last_epoch` 必须全部为 **30**；
中断时保存的部分 `last`、缺失轮数字段、v6 权重和 `best` 均被拒绝。AdaBN / mmd_marginal
的目标 PU 指标保留传导式解释，CWRU 的 `source_validation` 单独标为源域验证留出。
续修与验证结果见 `outputs/summary_2026-10-13/nontraining_evalfix.md`，此前暂停记录保留。

## 结果分析（已执行，2026-10-13）

判据是事先写死的（见上）：全体 116 单元宏平均 E1 ≥ 80% 且 E2 ≥ 90%。本轮 96 格中只有
`ce_K116` 的 11 格过线（全标、无留出单元，in-sample 参照），其余 85 格未过线。
零算力错误结构分析、六对配对链比较与分组并报已按冻结路径执行完毕，未临时加臂、
未改判据、未加新特征；结论与全部数字见 `outputs/summary_2026-10-13/results_summary.md`。
KB23/24/27 事后分析（需前向推理，非零算力）尚未执行。
