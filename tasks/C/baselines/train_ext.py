"""
扩展训练器 (Extended Trainer)
=============================

在官方 baselines.train 基础上增加四项能力：
  1. 样本加权 (sample_weight): 类平衡加权、长寿命加权
  2. 噪声增强 (noise_augmentation): 归一化空间高斯噪声
  3. 余弦退火 (cosine_anneal): 学习率余弦衰减 + 1e-6 下限
  4. 收缩折叠 (shrink): 向训练集中位数收缩，折叠进 target_scale/target_mean

所有扩展均修改梯度权重、增强、学习率调度或目标缩放；
架构、特征 schema、加载器检查、检查点格式保持不变。

用法:
  python -m baselines.train_ext --profile B --train ... --validation ... --output ...
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
import tempfile
import time

import numpy as np

try:
    from .model import Model, checkpoint_digest, _THREADS
    from .data import read_dataset, sha256
except ImportError:
    from model import Model, checkpoint_digest, _THREADS
    from data import read_dataset, sha256

ROOT = Path(__file__).resolve().parent

# ============================================================
# 扩展配置
# ============================================================

LR_FLOOR = 1e-6  # model.validate() 要求 0 < learning_rate <= 1


def compute_sample_weights(y: np.ndarray, mode: str, classes: list) -> np.ndarray:
    """
    计算样本权重
    
    Args:
        y: 训练目标 (分类任务为类别索引，回归任务为数值)
        mode: 加权模式
            - "none": 等权重
            - "invfreq": 逆频率加权（分类任务，类别平衡）
            - "rul-<cap>x<mult>": 长寿命加权（回归任务，RUL > cap 的样本权重 × mult）
        classes: 类别列表（分类任务）
    
    Returns:
        样本权重数组
    """
    if mode == "none":
        return np.ones(len(y), dtype=float)
    
    if mode == "invfreq":
        # 逆频率加权：每类权重 = 总样本数 / (类别数 × 该类样本数)
        counts = np.bincount(y.astype(int), minlength=len(classes))
        weights = np.zeros(len(y), dtype=float)
        for cls_idx, count in enumerate(counts):
            if count > 0:
                mask = y == cls_idx
                weights[mask] = len(y) / (len(classes) * count)
        return weights
    
    if mode.startswith("rul-"):
        # 长寿命加权：RUL > cap 的样本权重 × mult
        # 格式: rul-150x3 表示 RUL > 150 的样本权重 × 3
        parts = mode[4:].split("x")
        if len(parts) != 2:
            raise ValueError(f"长寿命加权格式错误: {mode}，应为 rul-<cap>x<mult>")
        cap, mult = float(parts[0]), float(parts[1])
        weights = np.ones(len(y), dtype=float)
        weights[y > cap] = mult
        return weights
    
    raise ValueError(f"未知的加权模式: {mode}")


def fit_epoch_weighted(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    aug_sd: float = 0.0
) -> float:
    """
    带样本加权和噪声增强的单 epoch 训练
    
    Args:
        model: 模型
        x: 输入特征
        y: 目标（已归一化）
        w: 样本权重
        aug_sd: 高斯噪声标准差（归一化空间，0 表示不增强）
    
    Returns:
        平均 loss
    """
    x_norm = model.normalized(x)
    
    if aug_sd > 0:
        noise = np.random.randn(*x_norm.shape) * aug_sd
        x_norm = x_norm + noise
    
    # 按 epoch 索引 shuffle（与官方实现一致）
    rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
    order = rng.permutation(len(x_norm))
    
    loss_sum = 0.0
    bs = model.meta["batch_size"]
    
    for start in range(0, len(order), bs):
        indices = order[start:start + bs]
        batch, expected = x_norm[indices], y[indices]
        wb = w[indices]
        
        hidden, logits = model.forward(batch)
        
        if model.meta["task"] == "classification":
            # 交叉熵 loss
            logits_shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits_shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            
            loss_sum -= float((wb * np.log(np.maximum(probabilities[np.arange(len(indices)), expected], 1e-300))).sum())
            
            delta = probabilities
            delta[np.arange(len(indices)), expected] -= 1
            delta = delta * (wb / wb.sum())[:, None]
        else:
            # MSE loss
            error = logits[:, 0] - expected
            loss_sum += float((wb * error ** 2).sum())
            delta = (2 * error * wb / wb.sum())[:, None]
        
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
        hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
        gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
        
        # 梯度裁剪 (norm = 5.0，与官方一致)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        if not math.isfinite(norm):
            raise ValueError("nonfinite gradient")
        
        model.meta["steps"] += 1
        
        # Adam 更新
        for name, gradient in gradients.items():
            gradient *= min(1.0, 5.0 / max(norm, 1e-12))
            model.first[name] = 0.9 * model.first[name] + 0.1 * gradient
            model.second[name] = 0.999 * model.second[name] + 0.001 * gradient ** 2
            first = model.first[name] / (1 - 0.9 ** model.meta["steps"])
            second = model.second[name] / (1 - 0.999 ** model.meta["steps"])
            model.params[name] -= model.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
        
        if not all(np.all(np.isfinite(p)) for p in model.params.values()):
            raise ValueError("nonfinite model state")
    
    model.meta["epochs"] += 1
    return loss_sum / len(x)


def compute_metrics(model: Model, data: dict) -> dict:
    """计算验证指标"""
    predicted = model.predict(data["x"])
    expected = data["y"]
    
    if model.meta["task"] == "classification":
        labels = predicted.argmax(axis=1)
        observed = sorted(set(expected.tolist()) | set(labels.tolist()))
        f1 = []
        for label in observed:
            tp = np.sum((labels == label) & (expected == label))
            prec_denom = np.sum(labels == label)
            rec_denom = np.sum(expected == label)
            prec = tp / prec_denom if prec_denom > 0 else 0.0
            rec = tp / rec_denom if rec_denom > 0 else 0.0
            f1.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        
        result = {
            "accuracy": float(np.mean(labels == expected)),
            "macro_f1": float(np.mean(f1))
        }
        
        # 严重故障漏报率（仅 B 题三分类历史基线）
        if "SevereFault" in model.meta["classes"]:
            severe_idx = model.meta["classes"].index("SevereFault")
            severe_count = int(np.sum(expected == severe_idx))
            severe_miss = int(np.sum((expected == severe_idx) & (labels != severe_idx)))
            result["severe_miss_rate"] = severe_miss / severe_count if severe_count > 0 else None
        
        return result
    else:
        error = predicted - expected
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2)))
        }


# ============================================================
# 主训练流程
# ============================================================

def train_extended(
    profile: str,
    train_path: str,
    validation_path: str,
    output: str,
    resume: str = None,
    epochs: int = None,
    learning_rate: float = None,
    batch_size: int = None,
    seed: int = None,
    stage_id: str = "baseline",
    label_provenance: str = None,
    # 扩展参数
    sample_weight: str = "none",
    gauss_aug: float = 0.0,
    cosine_anneal: bool = False,
    shrink_a: float = 1.0,
    shrink_center: str = "median",
    target_transform: str = "identity"
) -> dict:
    """
    扩展训练器主流程
    
    Returns:
        training.json 报告字典
    """
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory; previous rounds are never overwritten")
    
    # 加载配置
    config_path = Path(profile) if str(profile).endswith(".json") else ROOT / "configs" / (str(profile) + ".json")
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    
    if config.get("profile") not in ("A", "B", "C") or config.get("task") not in ("classification", "regression"):
        raise ValueError("unsupported baseline profile")
    
    if not isinstance(stage_id, str) or not stage_id.strip() or len(stage_id) > 128:
        raise ValueError("stage ID must be nonempty and bounded")
    
    # B 题 RUL 特殊检查
    bearing_rul = config["profile"] == "B" and config["task"] == "regression"
    if bearing_rul and config.get("target_unit") != "minutes":
        raise ValueError("B RUL requires target_unit=minutes")
    
    # B 题三分类历史基线需要 label_provenance
    provenance = None
    if config["profile"] == "B" and config["task"] == "classification":
        if label_provenance is None:
            raise ValueError("B requires explicit label provenance")
        raw = Path(label_provenance).read_bytes()
        declared = json.loads(raw)
        if not isinstance(declared, dict) or declared.get("status") not in ("draft", "approved") or \
           not isinstance(declared.get("method"), str) or not declared["method"].strip():
            raise ValueError("label provenance requires status=draft|approved and a nonempty method")
        provenance = {
            "sha256": sha256(raw),
            "declared_status": declared["status"],
            "method": declared["method"]
        }
    
    # 加载数据
    classes = config.get("classes", [])
    train = read_dataset(train_path, config["task"], classes)
    validation = read_dataset(validation_path, config["task"], classes, train["paths"])
    
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("training and validation sample IDs overlap")
    
    # 加载检查点
    parent = checkpoint_digest(resume) if resume else None
    model = Model.load(resume) if resume else Model(
        config["task"], train["paths"], classes,
        train["x"], train["y"],
        hidden=config["hidden"],
        seed=config["seed"] if seed is None else seed,
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"]
    )
    
    # 验证检查点兼容性
    if (model.meta["task"] != config["task"] or model.meta["paths"] != train["paths"] or
        model.meta["classes"] != classes or (resume and model.meta.get("profile") != config["profile"])):
        raise ValueError("checkpoint task, profile, feature schema or classes do not match")
    
    if bearing_rul and resume and model.meta.get("target_unit") != "minutes":
        raise ValueError("B RUL checkpoint must use minutes")
    
    if resume and seed is not None and seed != model.meta["seed"]:
        raise ValueError("resume must preserve the checkpoint's random seed")
    
    if resume and config["hidden"] != model.meta["hidden"]:
        raise ValueError("resume hidden width must match checkpoint")
    
    if resume and config["seed"] != model.meta["seed"]:
        raise ValueError("resume profile must preserve the checkpoint's random seed")
    
    # 更新元数据
    model.meta["profile"] = config["profile"]
    if bearing_rul:
        model.meta["target_unit"] = "minutes"
    if learning_rate is not None:
        model.meta["learning_rate"] = learning_rate
    if batch_size is not None:
        model.meta["batch_size"] = batch_size
    
    model.validate()
    
    # 训练轮数
    count = config["epochs"] if epochs is None else epochs
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("epochs must be in 1..10000")
    
    # 计算样本权重
    w = compute_sample_weights(train["y"], sample_weight, classes)
    
    # 目标变换（log1p）
    if target_transform == "log1p":
        if config["task"] != "regression":
            raise ValueError("log1p 目标变换仅适用于回归任务")
        # 注意：这里不实际变换训练目标，而是在预测时做逆变换
        # 目标缩放仍然在原始空间
        model.meta["target_transform"] = "log1p"
    elif target_transform != "identity":
        raise ValueError(f"未知的目标变换: {target_transform}")
    
    # 记录初始状态
    initial_steps, initial_epochs = model.meta["steps"], model.meta["epochs"]
    validation_before = compute_metrics(model, validation)
    
    # 余弦退火配置
    base_lr = model.meta["learning_rate"]
    
    # 训练循环
    started = time.monotonic()
    records = []
    
    for epoch_idx in range(count):
        # 余弦退火
        if cosine_anneal:
            lr = max(base_lr * 0.5 * (1 + math.cos(math.pi * epoch_idx / count)), LR_FLOOR)
            model.meta["learning_rate"] = lr
        
        # 目标归一化
        if config["task"] == "regression":
            target = (train["y"] - model.meta["target_mean"]) / model.meta["target_scale"]
        else:
            target = train["y"]
        
        # 训练一个 epoch
        loss = fit_epoch_weighted(model, train["x"], target, w, aug_sd=gauss_aug)
        
        # 验证
        val_metrics = compute_metrics(model, validation)
        records.append({
            "epoch": model.meta["epochs"],
            "step": model.meta["steps"],
            "loss": loss,
            "validation": val_metrics
        })
    
    # 恢复学习率（避免影响续训）
    if cosine_anneal:
        model.meta["learning_rate"] = base_lr
    
    duration = time.monotonic() - started
    
    if not records:
        raise ValueError("at least one training epoch is required")
    
    # 收缩折叠
    shrink_meta = None
    if shrink_a < 1.0:
        if config["task"] != "regression":
            raise ValueError("收缩仅适用于回归任务")
        
        # 计算收缩中心
        if shrink_center == "median":
            center = float(np.median(train["y"]))
        elif shrink_center == "mean":
            center = float(np.mean(train["y"]))
        else:
            raise ValueError(f"未知的收缩中心: {shrink_center}")
        
        # 折叠: target_scale' = a * target_scale
        #        target_mean' = a * target_mean + (1-a) * center
        model.meta["target_scale"] *= shrink_a
        model.meta["target_mean"] = shrink_a * model.meta["target_mean"] + (1 - shrink_a) * center
        
        shrink_meta = {
            "a": shrink_a,
            "center": center,
            "center_rule": shrink_center
        }
        
        model.validate()
    
    # 保存检查点
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir(mode=0o700)
        
        checkpoint = staging / "checkpoint.npz"
        model.save(checkpoint)
        
        # 生成报告
        report = {
            "format": "contest.cpu-baseline-report.v1",
            "profile": config["profile"],
            "task": config["task"],
            "purpose": "development_reference",
            "official_scoring_ready": False,
            "device": "cpu",
            "cpu_threads_requested": _THREADS,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "stage_id": stage_id,
            "seed": model.meta["seed"],
            "parameter_count": model.meta["parameter_count"],
            "input_features": len(model.meta["paths"]),
            "train_rows": len(train["ids"]),
            "validation_rows": len(validation["ids"]),
            "train_sha256": train["sha256"],
            "validation_sha256": validation["sha256"],
            "profile_sha256": sha256(config_raw),
            "parent_checkpoint_sha256": parent,
            "checkpoint_sha256": checkpoint_digest(checkpoint),
            "initial_steps": initial_steps,
            "initial_epochs": initial_epochs,
            "final_steps": model.meta["steps"],
            "final_epochs": model.meta["epochs"],
            "training_seconds": duration,
            "learning_rate": model.meta["learning_rate"],
            "batch_size": model.meta["batch_size"],
            "validation_before": validation_before,
            "validation_after": records[-1]["validation"],
            "configuration": {
                "hidden": model.meta["hidden"],
                "activation": "tanh",
                "optimizer": "Adam",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "gradient_clip_norm": 5.0,
                "feature_clip": [-10.0, 10.0],
                "normalization": "initial_training_mean_std",
                "learning_rate": model.meta["learning_rate"],
                "batch_size": model.meta["batch_size"],
                "requested_epochs": count,
                "classes": model.meta["classes"]
            },
            "source_sha256": {
                name: sha256((ROOT / name).read_bytes())
                for name in ("model.py", "data.py", "train.py", "predict.py")
            },
            "label_provenance": provenance,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # B 题 RUL 元数据
        if bearing_rul:
            report.update(
                target_unit="minutes",
                target_origin="remaining_time_from_lifecycle",
                target_definition="(last_snapshot - current_snapshot) * 1 minute; last observed snapshot has RUL 0"
            )
        
        # 扩展配置记录
        report["extensions"] = {
            "sample_weight": sample_weight,
            "gauss_aug_sd": gauss_aug,
            "cosine_anneal": cosine_anneal,
            "lr_floor": LR_FLOOR if cosine_anneal else None,
            "target_transform": target_transform,
            "shrink": shrink_meta,
            "epoch_selection": "fixed_budget_final_epoch",
            "note": "all extensions modify only gradient weighting, augmentation, learning-rate schedule "
                    "or target scaling; architecture, feature schema, loader checks and checkpoint format "
                    "are unchanged"
        }
        
        # 写入 training.json
        (staging / "training.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
        
        # 写入 history.jsonl
        with (staging / "history.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps({
                    "stage_id": stage_id,
                    "seed": model.meta["seed"],
                    **record
                }, allow_nan=False) + "\n")
        
        # 原子替换
        os.replace(staging, output)
    
    return report


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    
    # 必需参数
    parser.add_argument("--profile", required=True, help="A, B, C or explicit profile JSON")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    
    # 续训参数
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", help="Previous round checkpoint")
    continuation.add_argument("--from-baseline", help="Official/reference starting checkpoint")
    
    # 训练参数
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage-id", default="baseline")
    parser.add_argument("--label-provenance", help="Required only for historical B classification profiles")
    
    # 扩展参数
    parser.add_argument("--sample-weight", default="none",
                       help="样本加权: none | invfreq | rul-<cap>x<mult>")
    parser.add_argument("--gauss-aug", type=float, default=0.0,
                       help="归一化空间高斯噪声标准差")
    parser.add_argument("--cosine-anneal", action="store_true",
                       help="启用余弦退火 (1e-6 下限)")
    parser.add_argument("--shrink-a", type=float, default=1.0,
                       help="收缩系数 (0, 1], 1.0 表示不收缩")
    parser.add_argument("--shrink-center", default="median", choices=["median", "mean"],
                       help="收缩中心: median | mean")
    parser.add_argument("--target-transform", default="identity", choices=["identity", "log1p"],
                       help="目标变换: identity | log1p")
    
    args = parser.parse_args()
    
    if args.epochs is not None and not 1 <= args.epochs <= 10000:
        parser.error("epochs must be in 1..10000")
    
    # 执行训练
    report = train_extended(
        profile=args.profile,
        train_path=args.train,
        validation_path=args.validation,
        output=args.output,
        resume=args.resume or args.from_baseline,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        stage_id=args.stage_id,
        label_provenance=args.label_provenance,
        sample_weight=args.sample_weight,
        gauss_aug=args.gauss_aug,
        cosine_anneal=args.cosine_anneal,
        shrink_a=args.shrink_a,
        shrink_center=args.shrink_center,
        target_transform=args.target_transform
    )
    
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
