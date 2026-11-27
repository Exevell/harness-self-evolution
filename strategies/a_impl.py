"""A 题（UCI Gas Sensor Drift, 6 类气体识别, macro-F1）—— 单轮"一次性适配"策略。

设计目标（对应 work/CONTEXT.md §3/§7）：
 1. 每个 stage 的公开 train/validation 投影都会换，每阶段只有 1-4 轮：每轮只跑一次、不做跨轮状态积累，
    所有决策只看"本轮 train + 本轮 validation"。
 2. 漂移下"多训"是有害的（实测：plain 续训 120 epoch 在下一 batch 上 macro-F1 从 0.714 掉到 0.60）。
    因此候选一律带正则化（高斯噪声 + 类平衡权重 + 通道 dropout），并且**只在 validation 显著变好时
    才提交候选**，否则返回 selected="parent" 保留父检查点 —— 把单轮下降风险封顶在"不提交"上。
 3. 100% 确定性：不读系统时间做分支决策，随机性全部来自 seed 与 epoch 索引。
 4. 只用 numpy；不改入参、不写 stdout（日志走 log 回调）。

实测依据（本包 public train = batch1 445 行，public validation = batch2 1244 行；父检查点 macro-F1
= 0.714030，本文件复现 0.714029722 逐位一致）：
  * plain 续训：in-sample 最好 0.7264（+0.012），120 epoch 后跌到 0.60；样本外增益仅 +0.0057。
  * σ=0.8 + invfreq + 通道 dropout 0.1，~300 epoch：validation macro-F1 = 0.7475（+0.0335）；
    样本外（一半 val 挑 epoch、另一半评）增益 +0.0273 ± 0.0195，P(gain>0) = 0.95。
  * 6 类中 Acetaldehyde 在 batch2 上完全不可分（p_self 均值 0.020，109 例中 108 例被判成 Ethanol，
    F1 = 0）；这是 batch1→batch2 的类别级漂移，任何只用 batch1 训练的做法都救不回来。
  * hidden 32→128 零填充拓宽：函数保持严格成立（预测最大绝对差 0.0e+00，small_init；copy 模式 3.7e-05），
    参数量 17286 < 100000，合规；但**没有稳定增益**（样本外 +0.009~+0.018，P=0.90），且其表面增益主要
    来自顺带重置 Adam 一/二阶矩（单独重置 Adam 就有 0.7274）。故默认关闭，只在公开 train ≥1500 行
    （容量才有意义）时作为额外候选参与竞争。

用法（每轮）：
    result = train_and_select(Model, parent_ckpt_path, train_rows, val_rows,
                              "classification", CLASSES, "macro_f1", budget_s, seed)
    if result["selected"] == "candidate": 提交 result["checkpoint_path"]
    else:                                直接沿用父检查点

集成用（已解析成矩阵、产物以对象返回、不落盘，见文件末尾 train_and_select_xy）：
    result = train_and_select_xy(Model, parent_ckpt_path, x_train, y_train, x_val, y_val,
                                 "classification", CLASSES, "macro_f1", budget_s, seed)
    if result["selected"] == "candidate": 保存 result["model"]
    else:                                沿用父模型
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

DEFAULT_LR = 0.003
TARGET_STEPS = int(os.environ.get("CONTEST_A_STEPS", "600"))  # 每个候选的梯度步数预算（445 行时 ≈ 150 epoch）
ASSUMED_STEPS_PER_SECOND = 200   # 极保守吞吐假设，仅用于把 budget_s 换算成步数上限
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


# ─────────────────────────────────────────────────────────── 数据适配
def _path_key(path):
    """与官方 json.dumps(path, ensure_ascii=False, separators=(",",":")) 等价的紧凑特征键。"""
    parts = []
    for kind, value in path:
        if kind == "key":
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            parts.append('["key","%s"]' % escaped)
        else:
            parts.append('["index",%d]' % int(value))
    return "[" + ",".join(parts) + "]"


def _flatten(inputs):
    """展平 inputs；与官方 baselines/data.py::flatten 的遍历顺序一致（dict 按键排序）。"""
    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting too deep")
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            paths.append(_path_key(path))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    return paths, values


def _rows_to_xy(rows, classes, paths=None):
    """把多种输入形态统一成 {ids, x, y, paths}：{x,y} / {rows:[...]} / [jsonl_row, ...]。"""
    if isinstance(rows, dict) and "x" in rows and "y" in rows:
        x = np.asarray(rows["x"], dtype=np.float64)
        y = np.asarray(rows["y"])
        if x.ndim != 2:
            raise ValueError("x must be 2-D")
        if y.dtype.kind in "iu":
            labels = y.astype(np.int64)
        elif y.dtype.kind == "f":
            labels = y.astype(np.int64)
        else:
            labels = np.array([classes.index(str(v)) if not isinstance(v, (int, np.integer)) else int(v)
                               for v in y], dtype=np.int64)
        return {"ids": list(rows.get("ids", [])), "x": x, "y": labels,
                "paths": list(rows.get("paths") or (paths if paths is not None else []))}
    if isinstance(rows, dict) and "rows" in rows:
        rows = rows["rows"]
    ids, feats, targets = [], [], []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("row must be a dict")
        current, values = _flatten(row["inputs"])
        if paths is None:
            paths = current
        if current != paths:
            raise ValueError("feature schema changed between rows")
        answer = row["answer"]
        if isinstance(answer, str):
            if answer not in classes:
                raise ValueError("label outside class list")
            targets.append(classes.index(answer))
        else:
            targets.append(int(answer))
        ids.append(str(row.get("id", position)))
        feats.append(values)
    return {"ids": ids, "x": np.asarray(feats, dtype=np.float64),
            "y": np.asarray(targets, dtype=np.int64), "paths": list(paths)}


# ─────────────────────────────────────────────────────────── 指标
def _macro_f1(y, labels):
    """与官方 baselines/train.py 完全同口径：只在 (真实 ∪ 预测) 出现过的类上取平均。"""
    observed = sorted(set(y.tolist()) | set(labels.tolist()))
    f1 = []
    for label in observed:
        tp = np.sum((labels == label) & (y == label))
        denom = np.sum(labels == label) + np.sum(y == label)
        f1.append(2.0 * tp / denom if denom else 0.0)
    return float(np.mean(f1)) if f1 else 0.0


# ─────────────────────────────── 输出层逐类偏置校准（HAR-04.5 / S1，见 work/proxy/a/REPORT.md）
# 机制：漂移批次里稀疏类会整类塌缩（b3 Ethanol 0.08、b6 Toluene 0.00 …），macro-F1 对单类塌缩极敏感。
# 做法：候选定稿后，用**训练集内部留出**（每 4 行留 1 行，纯确定性）对分类头 b2 的 6 个偏置做
#       坐标上升；只改模型内部参数，不改特征 schema、不扩维度、不改 predict 协议。
# 守卫：留出增益 < 0.002 不启用；启用后若公开验证集比未校准更差，立即回退。
A_CALIB_ENABLED = os.environ.get("CONTEST_A_CALIB", "1") == "1"
A_CALIB_MAX_SHIFT = 1.2
A_CALIB_STEP = 0.2
A_CALIB_PASSES = 3
A_CALIB_HOLDOUT_EVERY = 4
A_CALIB_MIN_GAIN = 0.002


def _logits(model, x):
    _, z = model.forward(model.normalized(np.asarray(x, dtype=np.float64)))
    return z


def _calibrate_output_bias(model, x_cal, y_cal):
    """对输出层 b2 做逐类坐标上升，最大化训练集内部留出的 macro-F1（确定性）。"""
    z = _logits(model, x_cal)
    before = _macro_f1(y_cal, np.argmax(z, axis=1))
    best = before
    deltas = np.zeros(z.shape[1], dtype=np.float64)
    for _ in range(A_CALIB_PASSES):
        moved = False
        for c in range(z.shape[1]):
            for delta in (A_CALIB_STEP, -A_CALIB_STEP):
                trial = deltas.copy()
                trial[c] = float(np.clip(trial[c] + delta, -A_CALIB_MAX_SHIFT, A_CALIB_MAX_SHIFT))
                score = _macro_f1(y_cal, np.argmax(z + trial, axis=1))
                if score > best + 1e-12:
                    best, deltas, moved = score, trial, True
        if not moved:
            break
    return {"holdout_macro_f1_before": float(before), "holdout_macro_f1_after": float(best),
            "deltas": [round(float(v), 4) for v in deltas]}


def _apply_calibration(model, x_train, y_train, x_val, y_val, classes, metric, final_met, best, note, tag):
    """把留出校准应用到候选；公开验证变差则回退。返回 (final_met, calib_dict)。"""
    calib = None
    if not (A_CALIB_ENABLED):
        return final_met, calib
    holdout = np.arange(len(y_train)) % A_CALIB_HOLDOUT_EVERY == 0
    if int(holdout.sum()) < 80 or int((~holdout).sum()) < 200:
        return final_met, calib
    stats = _calibrate_output_bias(model, x_train[holdout], y_train[holdout])
    calib = stats
    if stats["holdout_macro_f1_after"] - stats["holdout_macro_f1_before"] < A_CALIB_MIN_GAIN:
        calib["applied"] = False
        calib["reverted_reason"] = "holdout_gain_too_small"
        return final_met, calib
    shift = np.asarray(stats["deltas"], dtype=np.float64)
    model.params["b2"] = model.params["b2"] + shift
    model.validate()
    cal_met, cal_labels = _metrics(model, x_val, y_val, classes, metric)
    if cal_met["macro_f1"] >= final_met["macro_f1"] - 1e-12:
        delta_val = cal_met["macro_f1"] - final_met["macro_f1"]
        best["metrics"], best["labels"] = cal_met, cal_labels
        calib["applied"] = True
        note(f"{tag}: 输出层校准 留出 {stats['holdout_macro_f1_before']:.4f}"
             f"→{stats['holdout_macro_f1_after']:.4f}，公开验证 {delta_val:+.4f}")
        return cal_met, calib
    model.params["b2"] = model.params["b2"] - shift
    calib["applied"] = False
    calib["reverted_reason"] = "public_validation_worse"
    note(f"{tag}: 校准后公开验证变差，回退")
    return final_met, calib


def _per_class(y, labels, classes):
    out = {}
    for index, name in enumerate(classes):
        tp = int(np.sum((labels == index) & (y == index)))
        n_true, n_pred = int(np.sum(y == index)), int(np.sum(labels == index))
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        out[name] = {"support": n_true, "predicted": n_pred, "precision": precision,
                     "recall": recall,
                     "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0}
    return out


def _metrics(model, x, y, classes, metric):
    labels = np.argmax(model.predict(x), axis=1)
    rep = _per_class(y, labels, classes)
    primary = float(metric(y, labels)) if callable(metric) else _macro_f1(y, labels)
    return {"macro_f1": _macro_f1(y, labels), "accuracy": float(np.mean(labels == y)),
            "primary_metric": primary,
            "per_class_f1": {k: v["f1"] for k, v in rep.items()},
            "per_class_recall": {k: v["recall"] for k, v in rep.items()},
            "per_class": rep}, labels


def _paired_bootstrap(y, labels_a, labels_b, n_boot, seed):
    """配对 bootstrap：macro-F1(a) - macro-F1(b) 的分布。1244 例时实测差值 sd ≈ 0.008~0.022。"""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xB007]))
    n = len(y)
    diffs = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        diffs[i] = _macro_f1(y[idx], labels_a[idx]) - _macro_f1(y[idx], labels_b[idx])
    return {"mean": float(diffs.mean()), "p2_5": float(np.percentile(diffs, 2.5)),
            "p97_5": float(np.percentile(diffs, 97.5)), "p_gt0": float(np.mean(diffs > 0))}


# ─────────────────────────────────────────────────────────── 正则化训练
def _class_weights(y, n_classes):
    """invfreq：w_c ∝ N/(K·n_c)，再归一化到均值 1（保持梯度尺度与未加权时可比）。"""
    counts = np.maximum(np.bincount(y, minlength=n_classes).astype(np.float64), 1.0)
    w = counts.sum() / (n_classes * counts)
    sample = w[y]
    return sample * (len(sample) / float(sample.sum()))


def _fit_epochs(model, x, y, epochs, *, lr, sample_weight=None, noise_std=0.0,
                feature_drop=0.0, batch_size=None, rng_seed=0):
    """确定性训练器。语义与官方 Model.fit 一致（同一归一化、同一 Adam、同一梯度裁剪、同一 epoch
    索引打乱），额外支持高斯噪声增强、通道 dropout、类平衡样本权重。

    * 噪声/dropout 作用在 normalized+clip 之后（即网络真实输入尺度：train 上各维 std≈1）。
    * 通道 dropout 把选中特征置为训练均值（归一化后为 0），模拟传感器通道漂移/失效。
    """
    xn = model.normalized(x)
    n = len(xn)
    if n == 0:
        raise ValueError("empty training set")
    bs = int(batch_size or model.meta["batch_size"])
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if w is not None and w.shape != (n,):
        raise ValueError("sample_weight shape mismatch")
    drop_rng = np.random.default_rng(np.random.SeedSequence([int(rng_seed), 0xD809]))
    losses = []
    for _ in range(int(epochs)):
        rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
        order = rng.permutation(n)
        loss_sum = 0.0
        for start in range(0, n, bs):
            indices = order[start:start + bs]
            batch, expected = xn[indices], y[indices]
            if noise_std > 0:
                batch = np.clip(batch + rng.normal(0.0, noise_std, batch.shape), -10.0, 10.0)
            if feature_drop > 0:
                batch = np.where(drop_rng.random(batch.shape) < feature_drop, 0.0, batch)
            hidden, logits = model.forward(batch)
            logits = logits - logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            rows = np.arange(len(indices))
            loss_sum += float(-np.log(np.maximum(prob[rows, expected], 1e-300)).sum())
            delta = prob
            delta[rows, expected] -= 1.0
            if w is not None:
                bw = w[indices][:, None]
                norm_w = float(np.sum(bw))
                if norm_w <= 0:
                    continue
                delta = delta * bw / norm_w
            else:
                delta = delta / len(indices)
            grads = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ model.params["w2"].T) * (1.0 - hidden ** 2)
            grads["w1"] = batch.T @ hidden_delta
            grads["b1"] = hidden_delta.sum(axis=0)
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient")
            model.meta["steps"] += 1
            step = model.meta["steps"]
            bias1, bias2 = 1.0 - 0.9 ** step, 1.0 - 0.999 ** step
            for name, grad in grads.items():
                grad = grad * min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = 0.9 * model.first[name] + 0.1 * grad
                model.second[name] = 0.999 * model.second[name] + 0.001 * grad ** 2
                model.params[name] -= lr * (model.first[name] / bias1) / (np.sqrt(model.second[name] / bias2) + 1e-8)
            if not all(np.all(np.isfinite(p)) for p in model.params.values()):
                raise ValueError("nonfinite model state")
        model.meta["epochs"] += 1
        losses.append(loss_sum / n)
    model.meta["learning_rate"] = float(lr)
    model.validate()
    return losses


# ─────────────────────────────────────────────────────────── 结构性迁移（默认不启用）
def widen(model, new_hidden=128, mode="small_init", seed=0):
    """hidden 拓宽 32→128 的**函数保持**权重迁移（合规迁移，不是重新初始化）。

    迁移合法性（CONTEXT §3 允许"显式兼容的权重迁移"）：
      * 旧单元原样填入前 old_hidden 列/行；新单元的 w2 行恒为 0 ⇒ 拓宽瞬间 f_new(x) ≡ f_old(x)，
        这是同一模型换更宽的参数化，不是"新模型冒充续训"。
      * mode="zero_frozen"：w1 新列也填 0。新单元 pre-activation 恒 0 ⇒ 梯度恒 0 ⇒ 永久冻结，
        是严格 no-op（保留作对照，证明"纯拓宽"本身不产生收益）。
      * mode="small_init"：w1 新列 ~ N(0, (0.05/sqrt(d))²)，w2 新行 = 0。输出仍与父模型逐位相同，
        但 h_new ≠ 0 让新单元第一步就能拿到梯度，从而自举学习（打破对称性）。
      * Adam 一/二阶矩按新形状重建为 0：这是优化器状态重置，属允许的"优化策略"改动，必须与结构
        改动分开计量（实测：单独重置 Adam 即可拿到 0.7274，几乎解释了 widen 的全部表面增益）。
      * 参数量 d·H + H + H·C + C = 128·128+128+128·6+6 = 17286 < 100000，满足 Model.validate()。
    """
    old_hidden = model.meta["hidden"]
    if new_hidden == old_hidden:
        return model, {"widened": False, "hidden": old_hidden}
    if not 1 <= new_hidden <= 128:
        raise ValueError("hidden out of range")
    d = len(model.meta["paths"])
    n_classes = len(model.meta["classes"])
    count = d * new_hidden + new_hidden + new_hidden * n_classes + n_classes
    if not 0 < count < 100000:
        raise ValueError("widened model violates the parameter budget")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x31DE]))
    w1 = np.zeros((d, new_hidden))
    b1 = np.zeros(new_hidden)
    w2 = np.zeros((new_hidden, n_classes))
    for j in range(new_hidden):
        if j < old_hidden:
            w1[:, j] = model.params["w1"][:, j]
            b1[j] = model.params["b1"][j]
            w2[j] = model.params["w2"][j]
        elif mode == "small_init":
            w1[:, j] = rng.normal(0.0, 0.05 / math.sqrt(d), d)
        elif mode != "zero_frozen":
            raise ValueError("unsupported widen mode")
    model.params = {"w1": w1, "b1": b1, "w2": w2, "b2": model.params["b2"].copy()}
    model.first = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.second = {k: np.zeros_like(v) for k, v in model.params.items()}
    model.meta["hidden"] = new_hidden
    model.meta["parameter_count"] = count
    model.validate()
    return model, {"widened": True, "from": old_hidden, "to": new_hidden, "mode": mode,
                   "parameter_count": count,
                   "function_preserving": mode in ("zero_frozen", "small_init"),
                   "adam_state_reset": True}


def widen_prediction_error(old_model, new_model, x, n=128):
    """数值验证函数保持性：拓宽前后预测概率的最大绝对差。"""
    probe = np.asarray(x, dtype=np.float64)[:n]
    return float(np.max(np.abs(old_model.predict(probe) - new_model.predict(probe))))


# ─────────────────────────────────────────────────────────── 内部工具
def _clone(model):
    with tempfile.TemporaryDirectory(prefix="a-clone-") as tmp:
        path = Path(tmp) / "clone.npz"
        model.save(path)
        return type(model).load(path)


def _save(model, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(100000):
        path = directory / f"{stem}-{index}.npz"
        if not path.exists():
            model.save(path)
            return path
    raise RuntimeError("cannot allocate checkpoint path")


def _param_move(model, parent):
    """参数相对父模型的相对 L2 位移（正则化强度的直接指标）。"""
    num = den = 0.0
    for key, value in model.params.items():
        ref = parent.params[key]
        if ref.shape != value.shape:
            continue
        num += float(np.sum((value - ref) ** 2))
        den += float(np.sum(ref ** 2))
    return math.sqrt(num / max(den, 1e-30))


def _digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ─────────────────────────────────────────────────────────── 主入口
def train_and_select(model_cls, parent_ckpt_path, train_rows, val_rows, task, classes,
                     metric, budget_s, seed, log=print, output_dir=None,
                     enable_widen=None, margin=None, min_p_gt0=0.80):
    """从父检查点续训本轮公开 train，按公开 validation 决策是否采纳候选。

    返回 dict（键见 work/CONTEXT.md §7）：checkpoint_path / metrics / parent_metrics /
    experiments / selected("candidate"|"parent") / hypothesis / change_summary / configuration。
    另外附带 decision（增益、配对 bootstrap、耗时）与 parent_meta 供审计。
    """
    started = time.monotonic()
    outdir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="a-round-"))
    outdir.mkdir(parents=True, exist_ok=True)
    note = log if callable(log) else (lambda *_a, **_k: None)

    train = _rows_to_xy(train_rows, classes)
    validation = _rows_to_xy(val_rows, classes, train["paths"])
    if list(train["paths"]) != list(validation["paths"]):
        raise ValueError("train/validation feature schema mismatch")
    if len(train["y"]) == 0 or len(validation["y"]) == 0:
        raise ValueError("empty dataset")
    task = task or "classification"
    n_classes = len(classes)

    # ---- 1) 父模型对照（必须先做） ----
    parent = model_cls.load(parent_ckpt_path)
    parent_meta = {k: parent.meta.get(k) for k in ("hidden", "epochs", "steps", "batch_size", "learning_rate", "seed")
                   if k in parent.meta}
    parent_metrics, parent_labels = _metrics(parent, validation["x"], validation["y"], classes, metric)
    note(f"[a] parent: macro_f1={parent_metrics['macro_f1']:.6f} acc={parent_metrics['accuracy']:.6f}")

    # ---- 2) 预算配比（只依赖 budget_s 与数据规模 ⇒ 确定性） ----
    bs = int(parent.meta.get("batch_size", 128))
    steps_per_epoch = max(1, int(math.ceil(len(train["y"]) / bs)))
    cheap = float(budget_s) < 60.0
    plan_epochs = int(np.clip(int(round(TARGET_STEPS / steps_per_epoch)), 30, 400))
    if cheap:
        plan_epochs = max(30, plan_epochs // 2)
    fractions = (0.5, 1.0) if cheap else CHECKPOINT_FRACTIONS
    lr = float(np.clip(parent_meta.get("learning_rate") or DEFAULT_LR, 1e-3, 5e-3))

    # 去掉恒输的 plain 候选（只省算力，不改变最优候选）
    specs = []
    if task == "classification":
        specs += [("regularized-lite", {"noise": 0.5, "weights": True, "drop": 0.1}),
                  ("regularized-strong", {"noise": 0.8, "weights": True, "drop": 0.1})]
    else:
        specs = [("fit-now-plain", {"noise": 0.0, "weights": False, "drop": 0.0})]
    do_widen = (len(train["y"]) >= 1500) if enable_widen is None else bool(enable_widen)
    if do_widen and task == "classification":
        specs.append(("widened-128", {"noise": 0.5, "weights": True, "drop": 0.1, "widen": 128}))
    # 步数上限只由 budget_s 与数据规模决定（不读墙钟做分支）⇒ 结果仍完全确定。
    # 吞吐按 200 步/s 保守假设（本机实测 >= 2400 步/s，留 12 倍余量），只用预算的 60%。
    budget_steps = int(ASSUMED_STEPS_PER_SECOND * max(float(budget_s), 5.0) * 0.6)
    plan_epochs = min(plan_epochs, max(30, budget_steps // max(len(specs) * steps_per_epoch, 1)))

    note(f"[a] n_train={len(train['y'])} n_val={len(validation['y'])} steps/epoch={steps_per_epoch} "
         f"plan_epochs={plan_epochs} candidates={len(specs)} "
         f"total_steps≈{len(specs) * plan_epochs * steps_per_epoch}")

    # ---- 3) 逐候选续训，按固定分数点在 validation 上打分，保留最优快照 ----
    candidates, experiments = [], []
    for name, spec in specs:
        t0 = time.monotonic()
        model = _clone(parent)
        widen_info = None
        if spec.get("widen"):
            model, widen_info = widen(model, spec["widen"], "small_init", seed=seed)
        weights = _class_weights(train["y"], n_classes) if spec["weights"] else None
        marks = sorted({max(1, int(round(plan_epochs * frac))) for frac in fractions})
        marks[-1] = plan_epochs
        best = None
        done = 0
        for mark in marks:
            _fit_epochs(model, train["x"], train["y"], mark - done, lr=lr, sample_weight=weights,
                        noise_std=spec["noise"], feature_drop=spec["drop"], batch_size=bs,
                        rng_seed=int(seed) + zlib.crc32(name.encode("utf-8")) % 9973)
            done = mark
            met, labels = _metrics(model, validation["x"], validation["y"], classes, metric)
            if best is None or met["macro_f1"] > best["metrics"]["macro_f1"]:
                best = {"epoch": mark, "metrics": met, "labels": labels,
                        "params": {k: v.copy() for k, v in model.params.items()},
                        "meta": {"steps": model.meta["steps"], "epochs": model.meta["epochs"],
                                 "hidden": model.meta["hidden"],
                                 "parameter_count": model.meta["parameter_count"]}}
        # 把最优快照装回模型再落盘：保证"落盘的 checkpoint"与"报告/选型用的指标"严格一致
        model.params = {k: v.copy() for k, v in best["params"].items()}
        model.meta.update(best["meta"])
        model.validate()
        final_met, _ = _metrics(model, validation["x"], validation["y"], classes, metric)
        final_met, calib = _apply_calibration(model, train["x"], train["y"], validation["x"],
                                              validation["y"], classes, metric, final_met, best, note,
                                              f"[a] {name}")
        ckpt = _save(model, outdir, stem=f"cand-{name}")
        seconds = time.monotonic() - t0
        experiments.append({
            "name": name,
            "spec": {"noise": spec["noise"], "class_balanced_weights": spec["weights"],
                     "feature_dropout": spec["drop"], "widen": spec.get("widen")},
            "epochs_planned": plan_epochs, "selected_epoch": best["epoch"], "lr": lr,
            "val_macro_f1_at_selected_epoch": round(best["metrics"]["macro_f1"], 6),
            "val_macro_f1_at_plan_end": round(final_met["macro_f1"], 6),
            "val_accuracy_at_selected_epoch": round(best["metrics"]["accuracy"], 6),
            "per_class_f1": {k: round(v, 4) for k, v in best["metrics"]["per_class_f1"].items()},
            "per_class_recall": {k: round(v, 4) for k, v in best["metrics"]["per_class_recall"].items()},
            "param_relative_l2_from_parent": round(_param_move(model, parent), 4),
            "train_seconds": round(seconds, 3), "widen": widen_info,
            "calibration": calib,
            "checkpoint": str(ckpt), "checkpoint_sha256": _digest(ckpt)})
        candidates.append({"name": name, "score": best["metrics"]["macro_f1"],
                           "labels": best["labels"], "metrics": best["metrics"],
                           "checkpoint": str(ckpt), "index": len(experiments) - 1,
                           "spec": dict(spec)})
        note(f"[a] {name}: val macro_f1={best['metrics']['macro_f1']:.6f}@ep{best['epoch']} "
             f"params_moved={_param_move(model, parent):.3f} ({seconds:.2f}s)")

    # ---- 4) 选型 + 显著性门槛 ----
    if margin is None:
        margin = 0.008 if len(validation["y"]) >= 800 else 0.020
    n_boot = 400 if cheap else 1500
    ranked = sorted(candidates, key=lambda c: (-c["score"], c["name"]))
    chosen = ranked[0]
    boot = _paired_bootstrap(validation["y"], chosen["labels"], parent_labels, n_boot, seed)
    gain = float(chosen["score"] - parent_metrics["macro_f1"])
    accept = bool(gain >= margin and boot["p_gt0"] >= min_p_gt0)
    for candidate in candidates:
        candidate["paired_vs_parent"] = (boot if candidate is chosen else
                                         _paired_bootstrap(validation["y"], candidate["labels"],
                                                           parent_labels, max(200, n_boot // 5), seed))

    if accept:
        checkpoint_path = chosen["checkpoint"]
        selected = "candidate"
        runtime_metrics = chosen["metrics"]
        change_summary = (f"续训 {plan_epochs} epoch（lr={lr:g}，σ={chosen['spec']['noise']}，"
                          f"类平衡 invfreq={'on' if chosen['spec']['weights'] else 'off'}，"
                          f"通道 dropout={chosen['spec']['drop']}，取 ep{experiments[chosen['index']]['selected_epoch']} 快照）。"
                          f"validation macro-F1 {parent_metrics['macro_f1']:.4f} -> {chosen['score']:.4f} "
                          f"({gain:+.4f}，margin={margin:.4f}，配对 bootstrap P(diff>0)={boot['p_gt0']:.2f})；"
                          f"参数相对父模型 L2 位移 {experiments[chosen['index']]['param_relative_l2_from_parent']:.3f}。")
    else:
        checkpoint_path = str(parent_ckpt_path)
        selected = "parent"
        runtime_metrics = parent_metrics
        change_summary = (f"最优候选 {chosen['name']} 仅把 validation macro-F1 从 "
                          f"{parent_metrics['macro_f1']:.4f} 抬到 {chosen['score']:.4f}（{gain:+.4f}），"
                          f"未同时满足 margin≥{margin:.4f} 与 P(diff>0)≥{min_p_gt0:.2f}"
                          f"（实测 {boot['p_gt0']:.2f}），判定不显著，保留父检查点。")

    elapsed = time.monotonic() - started
    result = {
        "checkpoint_path": checkpoint_path,
        "metrics": runtime_metrics,
        "parent_metrics": parent_metrics,
        "experiments": experiments,
        "selected": selected,
        "hypothesis": ("气体传感器跨 batch 存在**类别级**漂移：同一气体在下一 batch 的质心可能更靠近另一种"
                       "气体的旧质心（实测 batch2 Acetaldehyde 质心到 batch1 Ethanol 质心 7.50 < 到自身"
                       "batch1 质心 9.90；batch2 Acetaldehyde 召回 0.000，109 例里 108 例被判成 Ethanol）。"
                       "无约束续训会把决策边界整块推向大类，因此（1）用噪声/通道 dropout/类平衡权重把候选"
                       "限制在父模型附近，（2）只在本轮 validation 上显著变好才采纳，否则保留父检查点。"),
        "change_summary": change_summary,
        "configuration": {"plan_epochs": plan_epochs, "steps_per_epoch": steps_per_epoch,
                          "batch_size": bs, "learning_rate": lr,
                          "candidates": [s[0] for s in specs], "widen_enabled": bool(do_widen),
                          "margin": margin, "min_p_gt0": min_p_gt0, "n_bootstrap": n_boot,
                          "checkpoint_fractions": list(fractions),
                          "selection_metric": metric if isinstance(metric, str) else "custom_callable",
                          "budget_s": budget_s,
                          "regularization": ["gaussian_noise", "feature_dropout", "class_balanced_weights"],
                          "normalization": "parent_mean_std（本轮不重拟合，合规且实测更优）",
                          "seed": int(seed)},
        "decision": {"gain": gain, "margin": float(margin), "paired_bootstrap": boot,
                     "candidate": chosen["name"], "elapsed_seconds": round(elapsed, 3),
                     "candidates_ranked": [(c["name"], round(c["score"], 6)) for c in ranked],
                     "parent_checkpoint_sha256": _digest(parent_ckpt_path)},
        "parent_meta": parent_meta,
        "output_dir": str(outdir),
    }
    note(f"[a] selected={selected} gain={gain:+.4f} margin={margin:.4f} "
         f"P(diff>0)={boot['p_gt0']:.2f} elapsed={elapsed:.2f}s")
    return result


# ═══════════════════════════════════════════════════════════ 集成适配层：numpy 矩阵入参
# 与 train_and_select 同语义、同一套候选池/训练器/门槛，只是入参已经解析成矩阵、产物以对象返回，
# 便利 harness 直接嵌入（调用方负责落盘）。不写任何文件、不写 stdout（日志走 log 回调）。
def _clone_in_memory(model):
    """不落盘的深拷贝：只依赖 checkpoint 契约里的属性名，最后用 validate() 自检。"""
    clone = model.__class__.__new__(model.__class__)
    clone.meta = dict(model.meta)
    clone.params = {k: v.copy() for k, v in model.params.items()}
    clone.first = {k: v.copy() for k, v in model.first.items()}
    clone.second = {k: v.copy() for k, v in model.second.items()}
    clone.mean = model.mean.copy()
    clone.scale = model.scale.copy()
    clone.validate()
    return clone


def _xy_matrices(x, y, expected_width=None, name="x"):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    if expected_width is not None and x.shape[1] != expected_width:
        raise ValueError(f"{name} feature width differs from parent checkpoint")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError(f"{name}/{name.replace('x', 'y')} length mismatch")
    if y.dtype.kind in "iu":
        labels = y.astype(np.int64)
    elif y.dtype.kind == "f":
        labels = y.astype(np.int64)
    else:
        raise ValueError("y must be integer class indices (or strings handled by caller)")
    return x, labels


def train_and_select_xy(model_cls, parent_path, x_train, y_train, x_val, y_val,
                        task, classes, metric, budget_s, seed, log=print,
                        enable_widen=None, margin=None, min_p_gt0=0.80,
                        parent_model=None):
    """与 `train_and_select` 同语义，但输入是已解析好的 numpy 矩阵。

    参数
    ----
    model_cls     : 官方 Model 类（用于 `Model.load(parent_path)`）；传 None 时须给 `parent_model`
    parent_path   : 父检查点路径；若传入的是已有 Model 实例，则直接用它（不读文件）
    x_train/y_train, x_val/y_val : float64 矩阵（**原始单位**，与训练集同 schema）+ int64 类下标
    parent_model  : 可选，直接给已加载的父模型实例（避免重复 load）

    返回
    ----
    dict，字段与 `train_and_select` 一致，另加：
      "model"           : 应被使用的 Model 实例（selected="candidate" 时是候选，否则是父模型的深拷贝）
      "checkpoint_path" : selected="candidate" 时为 None（由调用方落盘）；selected="parent" 时为父路径
    """
    if callable(log):
        note = log
    else:
        note = lambda *_a, **_k: None  # noqa: E731
    started = time.monotonic()
    task = task or "classification"
    n_classes = len(classes)

    # ---- 1) 父模型对照 ----
    if parent_model is not None:
        parent = _clone_in_memory(parent_model)
        parent_origin = "object"
    elif hasattr(parent_path, "predict"):
        parent = _clone_in_memory(parent_path)
        parent_origin = "object"
    else:
        parent = _clone_in_memory(model_cls.load(parent_path))
        parent_origin = "path"
    width = len(parent.meta["paths"])
    x_train, y_train = _xy_matrices(x_train, y_train, width, "x_train")
    x_val, y_val = _xy_matrices(x_val, y_val, width, "x_val")
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("empty dataset")
    if np.any(y_train < 0) or np.any(y_train >= n_classes):
        raise ValueError("y_train has class index out of range")
    if np.any(y_val < 0) or np.any(y_val >= n_classes):
        raise ValueError("y_val has class index out of range")
    parent_meta = {k: parent.meta.get(k) for k in
                   ("hidden", "epochs", "steps", "batch_size", "learning_rate", "seed") if k in parent.meta}
    parent_metrics, parent_labels = _metrics(parent, x_val, y_val, classes, metric)
    note(f"[a-xy] parent: macro_f1={parent_metrics['macro_f1']:.6f} acc={parent_metrics['accuracy']:.6f}")

    # ---- 2) 预算配比（与 train_and_select 完全同一套规则，确定性） ----
    bs = int(parent.meta.get("batch_size", 128))
    steps_per_epoch = max(1, int(math.ceil(len(y_train) / bs)))
    cheap = float(budget_s) < 60.0
    plan_epochs = int(np.clip(int(round(TARGET_STEPS / steps_per_epoch)), 30, 400))
    if cheap:
        plan_epochs = max(30, plan_epochs // 2)
    fractions = (0.5, 1.0) if cheap else CHECKPOINT_FRACTIONS
    lr = float(np.clip(parent_meta.get("learning_rate") or DEFAULT_LR, 1e-3, 5e-3))

    # 去掉恒输的 plain 候选（只省算力，最优候选不变）
    specs = []
    if task == "classification":
        specs += [("regularized-lite", {"noise": 0.5, "weights": True, "drop": 0.1}),
                  ("regularized-strong", {"noise": 0.8, "weights": True, "drop": 0.1})]
    else:
        specs = [("fit-now-plain", {"noise": 0.0, "weights": False, "drop": 0.0})]
    do_widen = (len(y_train) >= 1500) if enable_widen is None else bool(enable_widen)
    if do_widen and task == "classification":
        specs.append(("widened-128", {"noise": 0.5, "weights": True, "drop": 0.1, "widen": 128}))
    budget_steps = int(ASSUMED_STEPS_PER_SECOND * max(float(budget_s), 5.0) * 0.6)
    plan_epochs = min(plan_epochs, max(30, budget_steps // max(len(specs) * steps_per_epoch, 1)))
    note(f"[a-xy] n_train={len(y_train)} n_val={len(y_val)} steps/epoch={steps_per_epoch} "
         f"plan_epochs={plan_epochs} candidates={len(specs)} "
         f"total_steps≈{len(specs) * plan_epochs * steps_per_epoch}")

    # ---- 3) 逐候选续训，固定分数点在 val 上打分，保留最优快照（内存内，不落盘） ----
    candidates, experiments = [], []
    for name, spec in specs:
        t0 = time.monotonic()
        model = _clone_in_memory(parent)
        widen_info = None
        if spec.get("widen"):
            model, widen_info = widen(model, spec["widen"], "small_init", seed=seed)
        weights = _class_weights(y_train, n_classes) if spec["weights"] else None
        marks = sorted({max(1, int(round(plan_epochs * frac))) for frac in fractions})
        marks[-1] = plan_epochs
        best, done = None, 0
        for mark in marks:
            _fit_epochs(model, x_train, y_train, mark - done, lr=lr, sample_weight=weights,
                        noise_std=spec["noise"], feature_drop=spec["drop"], batch_size=bs,
                        rng_seed=int(seed) + zlib.crc32(name.encode("utf-8")) % 9973)
            done = mark
            met, labels = _metrics(model, x_val, y_val, classes, metric)
            if best is None or met["macro_f1"] > best["metrics"]["macro_f1"]:
                best = {"epoch": mark, "metrics": met, "labels": labels,
                        "params": {k: v.copy() for k, v in model.params.items()},
                        "meta": {"steps": model.meta["steps"], "epochs": model.meta["epochs"],
                                 "hidden": model.meta["hidden"],
                                 "parameter_count": model.meta["parameter_count"]}}
        model.params = {k: v.copy() for k, v in best["params"].items()}
        model.meta.update(best["meta"])
        model.validate()
        final_met, _ = _metrics(model, x_val, y_val, classes, metric)
        final_met, calib = _apply_calibration(model, x_train, y_train, x_val, y_val, classes, metric,
                                              final_met, best, note, f"[a-xy] {name}")
        experiments.append({
            "name": name,
            "spec": {"noise": spec["noise"], "class_balanced_weights": spec["weights"],
                     "feature_dropout": spec["drop"], "widen": spec.get("widen")},
            "epochs_planned": plan_epochs, "selected_epoch": best["epoch"], "lr": lr,
            "val_macro_f1_at_selected_epoch": round(best["metrics"]["macro_f1"], 6),
            "val_macro_f1_at_plan_end": round(final_met["macro_f1"], 6),
            "val_accuracy_at_selected_epoch": round(best["metrics"]["accuracy"], 6),
            "per_class_f1": {k: round(v, 4) for k, v in best["metrics"]["per_class_f1"].items()},
            "per_class_recall": {k: round(v, 4) for k, v in best["metrics"]["per_class_recall"].items()},
            "param_relative_l2_from_parent": round(_param_move(model, parent), 4),
            "train_seconds": round(time.monotonic() - t0, 3), "widen": widen_info,
            "calibration": calib,
            "checkpoint": None})
        candidates.append({"name": name, "score": best["metrics"]["macro_f1"],
                           "labels": best["labels"], "metrics": best["metrics"],
                           "model": model, "index": len(experiments) - 1, "spec": dict(spec)})
        note(f"[a-xy] {name}: val macro_f1={best['metrics']['macro_f1']:.6f}@ep{best['epoch']} "
             f"params_moved={_param_move(model, parent):.3f} ({experiments[-1]['train_seconds']:.2f}s)")

    # ---- 4) 选型 + 显著性门槛（与 train_and_select 完全一致） ----
    if margin is None:
        margin = 0.008 if len(y_val) >= 800 else 0.020
    n_boot = 400 if cheap else 1500
    ranked = sorted(candidates, key=lambda c: (-c["score"], c["name"]))
    chosen = ranked[0]
    boot = _paired_bootstrap(y_val, chosen["labels"], parent_labels, n_boot, seed)
    gain = float(chosen["score"] - parent_metrics["macro_f1"])
    accept = bool(gain >= margin and boot["p_gt0"] >= min_p_gt0)
    for candidate in candidates:
        candidate["paired_vs_parent"] = (boot if candidate is chosen else
                                         _paired_bootstrap(y_val, candidate["labels"],
                                                           parent_labels, max(200, n_boot // 5), seed))

    if accept:
        selected = "candidate"
        out_model = chosen["model"]
        runtime_metrics = chosen["metrics"]
        checkpoint_path = None
        change_summary = (f"续训 {plan_epochs} epoch（lr={lr:g}，σ={chosen['spec']['noise']}，"
                          f"类平衡 invfreq={'on' if chosen['spec']['weights'] else 'off'}，"
                          f"通道 dropout={chosen['spec']['drop']}，取 ep{experiments[chosen['index']]['selected_epoch']} 快照）。"
                          f"validation macro-F1 {parent_metrics['macro_f1']:.4f} -> {chosen['score']:.4f} "
                          f"({gain:+.4f}，margin={margin:.4f}，配对 bootstrap P(diff>0)={boot['p_gt0']:.2f})；"
                          f"参数相对父模型 L2 位移 {experiments[chosen['index']]['param_relative_l2_from_parent']:.3f}。")
    else:
        selected = "parent"
        out_model = _clone_in_memory(parent)
        runtime_metrics = parent_metrics
        checkpoint_path = None if parent_origin == "object" else str(parent_path)
        change_summary = (f"最优候选 {chosen['name']} 仅把 validation macro-F1 从 "
                          f"{parent_metrics['macro_f1']:.4f} 抬到 {chosen['score']:.4f}（{gain:+.4f}），"
                          f"未同时满足 margin≥{margin:.4f} 与 P(diff>0)≥{min_p_gt0:.2f}"
                          f"（实测 {boot['p_gt0']:.2f}），判定不显著，返回父模型。")

    elapsed = time.monotonic() - started
    result = {
        "checkpoint_path": checkpoint_path,
        "model": out_model,
        "metrics": runtime_metrics,
        "parent_metrics": parent_metrics,
        "experiments": experiments,
        "selected": selected,
        "hypothesis": ("气体传感器跨 batch 存在**类别级**漂移：同一气体在下一 batch 的质心可能更靠近另一种"
                       "气体的旧质心（实测 batch2 Acetaldehyde 质心到 batch1 Ethanol 质心 7.50 < 到自身"
                       "batch1 质心 9.90；batch2 Acetaldehyde 召回 0.000，109 例里 108 例被判成 Ethanol）。"
                       "无约束续训会把决策边界整块推向大类，因此（1）用噪声/通道 dropout/类平衡权重把候选"
                       "限制在父模型附近，（2）只在本轮 validation 上显著变好才采纳，否则保留父模型。"),
        "change_summary": change_summary,
        "configuration": {"plan_epochs": plan_epochs, "steps_per_epoch": steps_per_epoch,
                          "batch_size": bs, "learning_rate": lr,
                          "candidates": [s[0] for s in specs], "widen_enabled": bool(do_widen),
                          "margin": margin, "min_p_gt0": min_p_gt0, "n_bootstrap": n_boot,
                          "checkpoint_fractions": list(fractions),
                          "selection_metric": metric if isinstance(metric, str) else "custom_callable",
                          "budget_s": budget_s,
                          "regularization": ["gaussian_noise", "feature_dropout", "class_balanced_weights"],
                          "normalization": "parent_mean_std（本轮不重拟合，合规且实测更优）",
                          "seed": int(seed), "interface": "xy_matrices",
                          "returns_model_object": True},
        "decision": {"gain": gain, "margin": float(margin), "paired_bootstrap": boot,
                     "candidate": chosen["name"], "elapsed_seconds": round(elapsed, 3),
                     "candidates_ranked": [(c["name"], round(c["score"], 6)) for c in ranked],
                     "parent_source": parent_origin},
        "parent_meta": parent_meta,
    }
    note(f"[a-xy] selected={selected} gain={gain:+.4f} margin={margin:.4f} "
         f"P(diff>0)={boot['p_gt0']:.2f} elapsed={elapsed:.2f}s")
    return result
