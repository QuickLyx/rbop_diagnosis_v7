# -*- coding: utf-8 -*-
"""阶段二固定矩阵入口；默认只检查计划，显式 --execute 才调用训练。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bearing_facts import CONDITIONS, PRIMARY_BEARINGS
from datasets import cwru
from features import feature_subset, load_cache
from model import MSRSABiLSTM
from preprocessing import build_repr_config
from transfer import DEVICE, load_checkpoint, seed_everything, steps_per_epoch, train_model

ROOT = Path(__file__).resolve().parent
ARMS = ("joint", "source_continue", "adabn", "mmd_marginal",
        "ce_K19", "ce_K38", "ce_K116", "ce_mmd")


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_frozen(config, selection):
    """仅检查已经冻结的实验值，不引入协议 schema 或数据指纹。"""
    if (config["protocol"] != "v7_stage2" or config["status"] != "frozen"
            or selection.get("protocol") != "v7_stage2"
            or selection.get("budgets_status") != "frozen"):
        raise ValueError("配置和标注预算必须是正式冻结的 v7_stage2")
    if (config["arms"] != list(ARMS) or config["folds"] != [0, 1, 2, 3]
            or config["seeds"] != [0, 1, 2] or config["budgets"] != [19, 38, 116]
            or config["split_seed"] != 0 or config["selection_rule"] != "last"):
        raise ValueError("矩阵、划分或末轮选择规则与冻结值不一致")
    expected = {"batch_size": 64, "epochs": 30, "joint_lr": 0.001,
                "adapt_lr": 0.0001, "lambda_mmd": 0.3,
                "mmd_sigma": [1, 5, 10], "mmd_estimator": "biased"}
    if config["parameters"] != expected:
        raise ValueError("训练参数与冻结值不一致")
    if (config["representation"] != "logspec512"
            or config["model"] != {"n_channels": 1, "n_classes": 4, "frontend": "spectral_v1"}
            or not isinstance(config["class_balance"], bool)):
        raise ValueError("网络、表征或类均衡设置不符合冻结配置")
    if (config["unlabeled_target_scope"] != "all_primary"
            or config["unlabeled_target_label_usage"] != 0
            or config["unlabeled_evaluation_regime"] != "transductive"):
        raise ValueError("无标签迁移必须用全部 116 主单元，且目标标签使用为 0")
    primary = {f"{bearing}/{condition}" for bearing in PRIMARY_BEARINGS for condition in CONDITIONS}
    all_units = selection["all"]
    if len(all_units) != 116 or set(all_units) != primary or selection["k_selected"] != 19:
        raise ValueError("选标全集必须恰为排 KB 的 116 单元，k 必须为 19")
    previous = set()
    for k in (19, 38, 116):
        budget = selection["budgets"][f"K{k}"]
        pool, test = budget["pool"], budget["test"]
        if (len(pool) != k or len(set(pool)) != k or len(test) != 116 - k
                or len(set(test)) != len(test) or set(pool) & set(test)
                or set(pool) | set(test) != primary or not previous <= set(pool)):
            raise ValueError(f"K{k} 池/测试必须嵌套、不重不漏地覆盖主口径")
        previous = set(pool)
    if set(selection["budgets"]["K19"]["pool"]) != set(selection["medoids"]):
        raise ValueError("K19 池必须使用冻结 medoid 清单")


def _window_count(cache, units, channels=None):
    units = set(units)
    return sum(r["stop"] - r["start"] for r in cache.records
               if r["unit_key"] in units and (channels is None or r["channel"] in channels))


def prepare_plan(args):
    """只读缓存/元数据和现有划分，不构造模型、优化器或特征训练副本。"""
    config_path = _project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection_path = _project_path(args.selection or config["selection_path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _check_frozen(config, selection)
    if args.arm not in config["arms"] or args.fold not in config["folds"] or args.seed not in config["seeds"]:
        raise ValueError("arm、fold 或 seed 不在冻结矩阵内")
    if args.no_class_balance:
        config["class_balance"] = False
    cache_dir = _project_path(args.cache_dir or config["cache_dir"])
    source, target = load_cache("cwru", cache_dir), load_cache("paderborn", cache_dir)
    repr_config = build_repr_config()
    if (source.repr_config != repr_config or target.repr_config != repr_config
            or selection.get("repr_config") != repr_config
            or selection.get("cache_version") != target.version):
        raise ValueError("缓存、选标与冻结表征不一致")
    recordings = cwru.enumerate_recordings()
    source_units = {r.unit_key for r in recordings}
    if (len(source_units) != 153 or len(recordings) != 306
            or {r["unit_key"] for r in source.records} != source_units):
        raise ValueError("源域必须是排 028 的全部 153 文件，拒绝旧 v6 源集")
    primary = set(selection["all"])
    cached_primary = {r["unit_key"] for r in target.records if r["bearing_id"] in PRIMARY_BEARINGS}
    if cached_primary != primary:
        raise ValueError("目标缓存没有恰好覆盖冻结 116 主单元")
    source_split = cwru.split_source_units(recordings, split_seed=config["split_seed"])[args.fold]
    s_ref = _window_count(source, source_split["train"], ("DE",))
    steps = steps_per_epoch(s_ref, config["parameters"]["batch_size"])
    budget_name = args.arm.removeprefix("ce_") if args.arm.startswith("ce_K") else "K19"
    budget = selection["budgets"][budget_name]
    unlabeled = args.arm in ("adabn", "mmd_marginal")
    supervised = args.arm in ("ce_K19", "ce_K38", "ce_K116", "ce_mmd")
    target_train = sorted(primary) if unlabeled else list(budget["pool"]) if supervised else []
    target_split = {"all": list(selection["all"]), "pool": list(budget["pool"]),
                    "test": list(budget["test"]), "train": target_train,
                    "budget": budget_name, "labeled": supervised,
                    "common_K19_test": list(selection["budgets"]["K19"]["test"]),
                    "common_K38_test": list(selection["budgets"]["K38"]["test"])}
    split = {"source": source_split, "target": target_split}
    checkpoint = None if args.arm == "joint" else _project_path(
        args.source_checkpoint or f"outputs/train/joint_f{args.fold}_s{args.seed}/last.pt")
    if args.arm == "joint" and args.source_checkpoint:
        raise ValueError("joint 必须从头训练，不能传入源 checkpoint")
    output = _project_path(args.output or
                           f"outputs/{'train' if args.execute else 'preflight'}/{args.arm}_f{args.fold}_s{args.seed}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"不覆盖非空输出目录，请另用 _retry1：{output}")
    config.update({"arm": args.arm, "fold": args.fold, "seed": args.seed,
                   "config_path": str(config_path), "selection_path": str(selection_path),
                   "cache_dir": str(cache_dir), "repr_config": repr_config,
                   "source_checkpoint": str(checkpoint) if checkpoint else None,
                   "initialization": "scratch" if args.arm == "joint" else "joint_last",
                   "source_split": source_split, "s_ref": s_ref,
                   "steps_per_epoch": steps, "source_file_count": 153,
                   "budget": budget_name, "target_train_units": target_train,
                   "supervised_pool_units": list(budget["pool"]) if supervised else [],
                   "target_label_usage": len(target_train) if supervised else 0,
                   "evaluation_regime": "transductive" if unlabeled else
                       "supervised_contains_training_pool" if supervised else "source_only",
                   "selected_checkpoint": "last.pt"})
    plan = {"protocol": config["protocol"], "arm": args.arm, "fold": args.fold, "seed": args.seed,
            "mode": "execute" if args.execute else "dry_run", "class_balance": config["class_balance"],
            "status": "dependency_pending" if checkpoint and not checkpoint.is_file() else "ready",
            "source_checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_metadata_checked": False, "output": str(output),
            "representation": config["representation"], "repr_config": repr_config,
            "model": config["model"], "parameters": config["parameters"],
            "s_ref": s_ref, "steps_per_epoch": steps,
            "num_samples_per_epoch": steps * config["parameters"]["batch_size"],
            "planned_steps": steps * config["parameters"]["epochs"],
            "planned_updates": 0 if args.arm == "adabn" else steps * config["parameters"]["epochs"],
            "source_file_count": 153, "source_train_unit_count": len(source_split["train"]),
            "source_validation_unit_count": len(source_split["val"]),
            "source_train_window_count": _window_count(source, source_split["train"]),
            "source_validation_window_count": _window_count(source, source_split["val"]),
            "target_train_unit_count": len(target_train),
            "target_train_window_count": _window_count(target, target_train),
            "target_sampling": "none" if not target_train else
                "uniform" if unlabeled else "balanced",
            "target_label_usage": config["target_label_usage"],
            "evaluation_regime": config["evaluation_regime"], "selection_rule": "last",
            "criterion": config["criterion"], "paired_chain": config["paired_chain"], "split": split}
    return config, split, plan, source, target


def validate_source_checkpoint(payload, config):
    """后续臂只接受本协议同折同 seed 同均衡口径的 joint 完整末轮。"""
    required = {"protocol": "v7_stage2", "arm": "joint", "fold": config["fold"],
                "seed": config["seed"], "class_balance": config["class_balance"],
                "representation": config["representation"], "repr_config": config["repr_config"],
                "model": config["model"], "checkpoint_kind": "last", "selection_rule": "last",
                "epoch": config["parameters"]["epochs"],
                "checkpoint_epoch": config["parameters"]["epochs"],
                "last_epoch": config["parameters"]["epochs"], "split_seed": config["split_seed"],
                "source_file_count": 153, "source_split": config["source_split"],
                "s_ref": config["s_ref"], "steps_per_epoch": config["steps_per_epoch"],
                "steps": config["steps_per_epoch"] * config["parameters"]["epochs"],
                "updates": config["steps_per_epoch"] * config["parameters"]["epochs"]}
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"源 checkpoint 的 {key} 不匹配，必须用同折同种子的 v7 joint last")
    if payload.get("parameters") != config["parameters"] or payload.get("source") != config["source"]:
        raise ValueError("源 checkpoint 必须按阶段二 153 文件冻结配置重训")


def build_parser():
    parser = argparse.ArgumentParser(description="阶段二固定矩阵，默认仅 dry-run（无模型/无训练）")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/stage2.json")
    parser.add_argument("--selection", default=None, help="默认读取配置中的冻结 selection_path")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--no-class-balance", action="store_true", help="只关闭源域窗口均衡，监督目标仍保持均衡")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只检查并写 dry_run.json，不构造模型")
    mode.add_argument("--execute", action="store_true", help="须获训练授权后才使用")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config, split, plan, source, target = prepare_plan(args)
    output = Path(plan["output"])
    if not args.execute:
        output.mkdir(parents=True, exist_ok=True)
        _dump_json(output / "dry_run.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    source_payload = None
    if args.arm != "joint":
        if plan["status"] == "dependency_pending":
            raise FileNotFoundError(f"尚无本折 joint 末轮：{plan['source_checkpoint']}")
        source_payload = load_checkpoint(plan["source_checkpoint"])
        validate_source_checkpoint(source_payload, config)
    # 在模型、优化器和训练前固定所有划分及配置；非空目录在 prepare_plan 已拒绝。
    output.mkdir(parents=True, exist_ok=True)
    _dump_json(output / "split.json", split)
    _dump_json(output / "config.json", config)
    seed_everything(args.seed)
    model = MSRSABiLSTM(n_channels=config["model"]["n_channels"],
                       n_classes=config["model"]["n_classes"]).to(DEVICE)
    if source_payload is not None:
        model.load_state_dict(source_payload["state_dict"])
    train_source = None if args.arm == "adabn" else feature_subset(source, split["source"]["train"])
    val_source = feature_subset(source, split["source"]["val"])
    train_target = feature_subset(target, split["target"]["train"], labeled=split["target"]["labeled"]) \
        if split["target"]["train"] else None
    return train_model(model, train_source, train_target, val_source, args.arm,
                       config, output, plan["s_ref"], args.seed)


if __name__ == "__main__":
    main()
