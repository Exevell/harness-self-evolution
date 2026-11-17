"""平台接入层：把 HAR‑01 自进化流程接到 contest.harness.workspace.v2。

一轮 = 一次真实模型调用 + 一组可复现的候选实验 + 按公开验证集指标选择产物。
模型本体沿用 baselines/model.py（NumPy MLP，可续训、无 pickle 检查点）。
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

_BASE = Path(__file__).resolve().parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from baselines import data as D  # noqa: E402
from baselines.model import Model  # noqa: E402

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
TASKS = ("classification", "regression")
TRAIN_CAP = int(os.environ.get("HARNESS_TRAIN_CAP", "12000"))
VAL_CAP = int(os.environ.get("HARNESS_VAL_CAP", "3000"))
MAX_EXPERIMENTS = int(os.environ.get("HARNESS_MAX_EXPERIMENTS", "3"))

_MODELS = {}


def _log(message):
    print("[harness] %s" % message, file=sys.stderr)


def _work():
    return Path(os.environ.get("CONTEST_WORK", "/workspace"))


def _take(iterator, cap):
    rows = []
    for row in iterator:
        rows.append(row)
        if len(rows) >= cap:
            break
    return rows


def _features(inputs, paths):
    """把 JSON 输入对齐到检查点声明的特征顺序，缺字段补 0 并计数。"""
    current, values = D.flatten(inputs)
    if current == paths:
        return values, 0
    lookup = dict(zip(current, values))
    return [lookup.get(key, 0.0) for key in paths], sum(1 for key in paths if key not in lookup)


def _matrix(rows, paths, task, classes):
    x, y, missing = [], [], 0
    for row in rows:
        values, absent = _features(row["inputs"], paths)
        answer = row.get("answer")
        if task == "classification":
            if not isinstance(answer, str) or answer not in classes:
                continue
            target = classes.index(answer)
        else:
            if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer):
                continue
            target = float(answer)
        x.append(values)
        y.append(target)
        missing += absent
    if not x:
        raise ValueError("no usable rows for the declared task")
    kind = np.int64 if task == "classification" else np.float64
    return (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=kind),
            {"rows": len(x), "missing_fields": missing})


def _score(model, x, y, task, classes):
    prediction = model.predict(x)
    if task == "classification":
        labels = prediction.argmax(axis=1)
        scores = []
        for index in range(len(classes)):
            hit = np.sum((labels == index) & (y == index))
            false_positive = np.sum((labels == index) & (y != index))
            false_negative = np.sum((labels != index) & (y == index))
            precision = hit / (hit + false_positive) if hit + false_positive else 0.0
            recall = hit / (hit + false_negative) if hit + false_negative else 0.0
            scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return float(np.mean(scores)), {"macro_f1": round(float(np.mean(scores)), 6),
                                        "accuracy": round(float(np.mean(labels == y)), 6)}
    rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
    mae = float(np.mean(np.abs(prediction - y)))
    return -rmse, {"rmse": round(rmse, 6), "mae": round(mae, 6)}


def _plans(round_index):
    """每一轮的候选网格，越往后学习率越小、轮数越多。"""
    base = [
        [(1, 0.003), (2, 0.002)],
        [(2, 0.002), (2, 0.0015)],
        [(3, 0.0015), (2, 0.001)],
        [(3, 0.001), (4, 0.0008)],
    ]
    return base[(round_index - 1) % len(base)][:MAX_EXPERIMENTS]


def _receipt_json(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _parent_checkpoint(rnd):
    descriptor = rnd.parent.descriptor
    declared = list(descriptor.get("checkpoint_paths") or [])
    name = next((item for item in declared if item.endswith(".npz")), declared[0] if declared else "checkpoint.npz")
    return name, rnd.parent.path(name)


def train(rnd):
    contract = rnd.task or {}
    task = contract.get("task")
    if task not in TASKS:
        raise ValueError("unsupported task contract: %r" % (task,))
    metric = str(contract.get("metric") or ("macro_f1" if task == "classification" else "rmse"))
    dataset = rnd.dataset_key
    round_index = int(rnd.round_index)
    history = rnd.history()

    checkpoint_name, checkpoint_path = _parent_checkpoint(rnd)
    parent = Model.load(checkpoint_path)
    if parent.meta["task"] != task:
        raise ValueError("parent checkpoint task mismatch")
    classes = list(parent.meta["classes"])
    paths = list(parent.meta["paths"])

    train_rows = _take(rnd.data.training_rows(), TRAIN_CAP)
    val_rows = _take(rnd.data.validation_rows(), VAL_CAP)
    x_train, y_train, train_info = _matrix(train_rows, paths, task, classes)
    val_info, parent_metrics, parent_score = {"rows": 0}, {}, None
    try:
        x_val, y_val, val_info = _matrix(val_rows, paths, task, classes)
        parent_score, parent_metrics = _score(parent, x_val, y_val, task, classes)
    except ValueError:
        _log("公开验证集不可用，改用训练集留出评估")
        cut = max(1, int(len(x_train) * 0.8))
        x_val, y_val = x_train[cut:], y_train[cut:]
        x_train, y_train = x_train[:cut], y_train[:cut]
        if len(x_val):
            parent_score, parent_metrics = _score(parent, x_val, y_val, task, classes)
    _log("数据集=%s 轮次=%d 特征=%d 训练=%d 验证=%d 父模型%s=%s"
         % (dataset, round_index, len(paths), len(x_train), val_info.get("rows", len(x_val)),
            metric, parent_metrics.get(metric, "n/a")))

    experiments = []
    best = None  # 只从"真正续训过的候选"里选，父模型仅作基线对照，不作为提交产物
    for epochs, learning_rate in _plans(round_index):
        candidate = Model.load(checkpoint_path)
        candidate.meta["learning_rate"] = float(learning_rate)
        try:
            candidate.fit(x_train, y_train, epochs)
            score, metrics = _score(candidate, x_val, y_val, task, classes)
        except ValueError as error:
            _log("候选 epochs=%d lr=%s 失败：%s" % (epochs, learning_rate, error))
            continue
        experiments.append({"epochs": epochs, "learning_rate": learning_rate,
                            "score": round(score, 6), "metrics": metrics})
        _log("候选 epochs=%d lr=%s -> %s=%s" % (epochs, learning_rate, metric, metrics.get(metric)))
        if best is None or score > best["score"]:
            best = {"score": score, "epochs": epochs, "learning_rate": learning_rate,
                    "metrics": metrics, "model": candidate}

    summary = {
        "dataset_key": dataset, "round_index": round_index, "task": task, "metric": metric,
        "history_rounds": len(history), "parent_metric": parent_metrics,
        "experiments": experiments,
        "train_rows": train_info["rows"], "validation_rows": val_info.get("rows", 0),
    }
    receipt = rnd.model(
        [{"role": "user", "content": (
            "本轮公开数据实验摘要（JSON）：%s\n"
            "请判断：是否采纳当前最优候选，或建议一个 epochs(1-6) 与 learning_rate(0.0005-0.01) 组合继续试。"
            "只输出一个 JSON：{\"decision\":\"keep_best|try_again\",\"epochs\":整数,\"learning_rate\":数,\"reason\":\"一句话\"}。"
            % json.dumps(summary, ensure_ascii=False, sort_keys=True)
        )}],
        max_tokens=min(400, rnd.max_output_tokens),
        temperature=0,
    )
    advice = _receipt_json(receipt.text)
    _log("模型建议：%s" % json.dumps(advice, ensure_ascii=False))

    if advice.get("decision") == "try_again" and len(experiments) < MAX_EXPERIMENTS + 1:
        try:
            epochs = max(1, min(6, int(advice.get("epochs", 0))))
            learning_rate = float(advice.get("learning_rate", 0.0))
        except (TypeError, ValueError):
            epochs, learning_rate = 0, 0.0
        if epochs and 0.0005 <= learning_rate <= 0.01:
            trial = Model.load(checkpoint_path)
            trial.meta["learning_rate"] = learning_rate
            try:
                trial.fit(x_train, y_train, epochs)
                score, metrics = _score(trial, x_val, y_val, task, classes)
                experiments.append({"epochs": epochs, "learning_rate": learning_rate,
                                    "score": round(score, 6), "metrics": metrics, "from_advice": True})
                _log("建议候选 epochs=%d lr=%s -> %s=%s" % (epochs, learning_rate, metric, metrics.get(metric)))
                if best is None or score > best["score"]:
                    best = {"score": score, "epochs": epochs, "learning_rate": learning_rate,
                            "metrics": metrics, "model": trial}
            except ValueError as error:
                _log("建议候选失败：%s" % error)

    work = _work()
    candidate_dir = work / ("candidate-r%02d" % round_index)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    if best is None:
        # 极端兜底：至少完成一次可复现的续训，保证本轮产物一定是"训练过的检查点"，
        # 不会与父检查点字节相同。
        fallback = Model.load(checkpoint_path)
        fallback.meta["learning_rate"] = 0.001
        fallback.fit(x_train, y_train, 1)
        score, metrics = _score(fallback, x_val, y_val, task, classes)
        experiments.append({"epochs": 1, "learning_rate": 0.001, "score": round(score, 6),
                            "metrics": metrics, "fallback": True})
        best = {"score": score, "epochs": 1, "learning_rate": 0.001, "metrics": metrics,
                "model": fallback}
    best["model"].save(candidate_dir / checkpoint_name)

    hypothesis = ("第 %d 轮：父模型 %s=%s，用 epochs/lr 网格在公开训练集续训并只在公开验证集上选择，"
                  "避免把隐藏集信息带回训练。" % (round_index, metric, parent_metrics.get(metric, "n/a")))
    change_summary = ("相对父模型：epochs+%d、learning_rate=%s，%s 由 %s 变为 %s。"
                      % (best["epochs"], best["learning_rate"], metric,
                         parent_metrics.get(metric, "n/a"), best["metrics"].get(metric, "n/a")))

    flat = {
        "dataset_key": dataset,
        "task": task,
        "metric": metric,
        "epochs": int(best["epochs"]),
        "learning_rate": float(best["learning_rate"]),
        "parent_metric_value": float(parent_metrics.get(metric, 0.0) or 0.0),
        "candidate_metric_value": float(best["metrics"].get(metric, 0.0) or 0.0),
        "experiments": len(experiments),
        "train_rows": int(train_info["rows"]),
        "validation_rows": int(val_info.get("rows", 0)),
    }
    if best["metrics"].get(metric) is None:
        raise ValueError("candidate metric missing for %s" % metric)

    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=hypothesis,
        change_summary=change_summary,
        configuration=flat,
        seed=int(parent.meta["seed"]) + round_index,
        checkpoint_paths=[checkpoint_name],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )

    state_dir = work / ("state-r%02d" % round_index)
    state = rnd.state_workspace(str(state_dir))
    (state / "note.json").write_text(json.dumps({"round": round_index, "dataset_key": dataset,
                                                 "metric": metric, "selected": best["metrics"],
                                                 "experiments": len(experiments)},
                                                ensure_ascii=False, sort_keys=True), encoding="utf-8")
    rnd.save_state(state)

    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=hypothesis,
        change_summary=change_summary,
        reason=("依据本轮 %d 组候选在公开验证集上的实测 %s，以及模型 `%s` 的建议（call_id=%s）：%s。"
                % (len(experiments), metric, rnd.model_name, receipt.call_id,
                   str(advice.get("reason") or "采纳实测最优，未采纳建议" ))),
        configuration=flat,
        seed=int(parent.meta["seed"]) + round_index,
    )


def _checkpoint_for(model_dir):
    override = os.environ.get("CONTEST_ARTIFACT_PATH")
    if override and Path(override).is_file():
        return override
    found = sorted(Path(model_dir).rglob("*.npz"))
    if not found:
        raise ValueError("no checkpoint found under %s" % model_dir)
    return str(found[0])


def predict(sample, model_dir):
    checkpoint = _checkpoint_for(model_dir)
    model = _MODELS.get(checkpoint)
    if model is None:
        model = Model.load(checkpoint)
        _MODELS[checkpoint] = model
    values, missing = _features(sample.inputs, model.meta["paths"])
    if missing:
        _log("预测输入缺少 %d 个声明特征，已补 0" % missing)
    x = np.asarray([values], dtype=np.float64)
    if sample.task == "regression":
        value = float(model.predict(x)[0])
        if not math.isfinite(value):
            value = 0.0
        return {"value": max(0.0, value)}
    probabilities = model.predict(x)[0]
    scores = []
    for value in probabilities:
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        scores.append(min(1.0, max(0.0, value)))
    labels = list(model.meta["classes"])
    if len(scores) != len(labels):
        raise ValueError("class count mismatch")
    rounded = [round(score, 6) for score in scores]
    top = int(np.argmax(rounded))
    if sum(1 for value in rounded if value == rounded[top]) > 1:
        # 顶层并列必须消掉：优先抬高最优项，若已到 1.0 就把其余并列项压低。
        if rounded[top] < 1.0:
            rounded[top] = round(rounded[top] + 1e-6, 6)
        else:
            for index, value in enumerate(rounded):
                if index != top and value == rounded[top]:
                    rounded[index] = max(0.0, round(value - 1e-6, 6))
    return [{"label": label, "score": score} for label, score in zip(labels, rounded)]
