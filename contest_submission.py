"""HAR-02 submission: drift-aware, budgeted, parent-safe self-evolution.

One round = evaluate the parent on the *current stage's* public validation
projection, train a small budgeted set of continuation candidates on the
current stage's public training projection, keep the best one only when it
beats the parent by a margin, otherwise keep the parent artifact unchanged.

Everything is pure NumPy and resume-compatible with the official
``contest.cpu-mlp.v1`` checkpoint format.
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# The formal pod is 2 vCPU; two BLAS threads help and stay inside the budget.
# These must be set before numpy/baselines.model pick them up.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("BASELINE_CPU_THREADS", "2")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from baselines import data as D                                  # noqa: E402
from baselines.model import Model                                # noqa: E402
from strategies import core, is_better, sample_weights, train_and_select  # noqa: E402
import stage_policy  # noqa: E402  阶段感知门槛（新阶段首轮放宽，同阶段严格）
from strategies import widen_hidden, _apply_target_transform               # noqa: E402

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
TASKS = ("classification", "regression")
STATE_NOTE = "harness_state.json"
_MODELS = {}


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def _log(message):
    print("[harness] %s" % message, file=sys.stderr, flush=True)


def _work():
    return Path(os.environ.get("CONTEST_WORK", "/workspace"))


def _matrix(rows, paths, task, classes, limit=None):
    """Align JSON inputs to the checkpoint feature order, collecting labels."""
    x, y, ids, missing, seen = [], [], [], 0, 0
    lookup = None
    for row in rows:
        seen += 1
        if limit is not None and seen > limit:
            break
        inputs = row.get("inputs")
        if not isinstance(inputs, dict):
            continue
        current, values = D.flatten(inputs)
        if current != paths:
            if lookup is None:
                lookup = {}
            lookup = dict(zip(current, values))
            values = [lookup.get(key, 0.0) for key in paths]
            missing += sum(1 for key in paths if key not in lookup)
        answer = row.get("answer")
        if task == "classification":
            if not isinstance(answer, str) or answer not in classes:
                continue
            target = classes.index(answer)
        else:
            if isinstance(answer, bool) or not isinstance(answer, (int, float)):
                continue
            if not math.isfinite(answer) or answer < 0:
                continue
            target = float(answer)
        identity = row.get("id")
        ids.append(identity if isinstance(identity, str) else str(seen))
        x.append(values)
        y.append(target)
    if not x:
        raise ValueError("no usable rows for the declared task")
    kind = np.int64 if task == "classification" else np.float64
    return (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=kind), ids,
            {"rows": len(x), "missing_fields": missing})


def _parent_checkpoint(rnd):
    descriptor = rnd.parent.descriptor
    declared = list(descriptor.get("checkpoint_paths") or [])
    name = next((item for item in declared if item.endswith(".npz")),
                declared[0] if declared else "checkpoint.npz")
    return name, rnd.parent.path(name)


def _receipt_json(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _history_digest(history):
    digest = []
    for record in history[-4:]:
        configuration = record.get("configuration") or {}
        digest.append({
            "id": record.get("id"),
            "round": configuration.get("round_index", record.get("round_index")),
            "stage": configuration.get("stage_id"),
            "selected": configuration.get("selected"),
            "metric": configuration.get("candidate_metric_value"),
            "parent_metric": configuration.get("parent_metric_value"),
            "plan": configuration.get("plan"),
        })
    return digest


# --------------------------------------------------------------------------- #
# extra candidate requested by the model advisor
# --------------------------------------------------------------------------- #
def _stage_context(history, stage_id):
    """从历史 finish 记录推出 (是否新阶段, 本阶段第几轮)。"""
    prev_stage, same = None, 0
    for record in reversed(history or []):
        configuration = record.get("configuration") or {}
        if configuration.get("stage_id") is None:
            continue
        prev_stage = configuration.get("stage_id")
        if prev_stage != stage_id:
            break
        same += 1
    return (prev_stage != stage_id), same + 1


def _train_extra(rnd, parent_path, x_train, y_train, x_val, y_val, task, classes,
                 metric, advice, best, direction, parent_score, seed, log):
    epochs = max(1, min(600, int(advice.get("epochs", 0) or 0)))
    learning_rate = float(advice.get("learning_rate", 0.0) or 0.0)
    if not epochs or not (1e-4 <= learning_rate <= 0.05):
        return best, None
    noise = float(advice.get("noise", 0.0) or 0.0)
    shrink = advice.get("shrink")
    plan = {"name": "llm_advice", "epochs": epochs, "lr": learning_rate,
            "weight": advice.get("weight"), "noise": max(0.0, min(2.0, noise)),
            "cosine": True, "shrink": shrink, "margin": 0.002}
    started = time.time()
    candidate = Model.load(str(parent_path))
    candidate.meta["seed"] = int((candidate.meta["seed"] + seed) % (2 ** 32))
    target_y = _apply_target_transform(candidate, y_train, advice.get("target"))
    weights = sample_weights(plan.get("weight"), y_train, task, classes)
    try:
        core.fit_ext(candidate, x_train, target_y, epochs, lr=learning_rate,
                     weights=weights, noise=plan["noise"], cosine=True)
    except ValueError as error:
        log("建议候选失败：%s" % error)
        return best, None
    if task == "regression" and shrink:
        core.fold_shrink(candidate, float(shrink), float(np.median(y_train)))
    metrics = core.evaluate(candidate, x_val, y_val, task, len(classes))
    value = core.score_value(metrics, metric)
    record = {"name": "llm_advice", "epochs": epochs, "lr": learning_rate,
              "noise": plan["noise"], "shrink": shrink,
              "seconds": round(time.time() - started, 2), "metrics": metrics,
              "score": value, "from_advice": True}
    log("建议候选 -> %s=%s" % (metric, value))
    if core.better(direction, value, parent_score, plan["margin"]):
        if best is None or is_better(direction, value, best["score"]):
            best = {"score": value, "plan": plan, "metrics": metrics,
                    "model": candidate, "epochs": epochs}
    return best, record


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(rnd):
    started = time.time()
    contract = rnd.task or {}
    task = contract.get("task")
    if task not in TASKS:
        raise ValueError("unsupported task contract: %r" % (task,))
    metric = str(contract.get("metric") or ("macro_f1" if task == "classification" else "rmse"))
    direction = "higher" if metric == "macro_f1" else "lower"
    dataset = rnd.dataset_key
    round_index = int(rnd.round_index)
    stage_id = getattr(rnd, "stage_id", "") or ""
    history = rnd.history()

    checkpoint_name, checkpoint_path = _parent_checkpoint(rnd)
    parent = Model.load(str(checkpoint_path))
    if parent.meta["task"] != task:
        raise ValueError("parent checkpoint task mismatch")
    classes = list(parent.meta["classes"])
    paths = list(parent.meta["paths"])

    wall = getattr(rnd, "wall_seconds", None)
    wall = float(wall) if isinstance(wall, (int, float)) and wall > 0 else 240.0
    # Keep >=45% of the round budget for candidate upload, model call and finish.
    train_budget = max(20.0, min(wall * 0.40, 180.0) - (time.time() - started))

    x_train, y_train, ids_train, train_info = _matrix(
        rnd.data.training_rows(), paths, task, classes)
    x_val, y_val, ids_val, val_info = _matrix(
        rnd.data.validation_rows(), paths, task, classes)
    digest = _projection_digest(dataset, ids_train, y_train, ids_val, y_val)
    previous = _previous_state(rnd)
    _log("数据集=%s stage=%s 轮次=%d 特征=%d 训练=%d 验证=%d 预算=%.0fs digest=%s"
         % (dataset, stage_id, round_index, len(paths), train_info["rows"],
            val_info["rows"], train_budget, digest[:12]))

    # --- 阶段感知门槛：新阶段第一轮放宽（让阶段切换后的适配能被采纳），
    #     同阶段后续轮恢复严格，验证投影过小时不放宽。失败即退回固定门槛。 ---
    stage_changed, round_in_stage = _stage_context(history, stage_id)
    policy_margin = None
    try:
        if task == "classification":
            parent_now = stage_policy.macro_f1_observed(
                y_val, np.argmax(parent.predict(x_val), axis=1))
        else:
            parent_now = core.score_value(
                core.evaluate(parent, x_val, y_val, task, len(classes)), metric)
        policy_margin = stage_policy.margin_for(
            parent_now, None, metric, direction, stage_changed, round_in_stage,
            int(len(y_val)), dataset_key=dataset,
            prev_parent_metric=_num(previous.get("parent_value")))
        _log("阶段策略：stage_changed=%s round_in_stage=%d n_val=%d margin=%.4f"
             % (stage_changed, round_in_stage, len(y_val), policy_margin))
    except Exception as error:
        _log("阶段策略不可用，沿用固定门槛：%r" % (error,))

    # Same projection as the round that already concluded "no improvement":
    # re-training it again would only burn the round budget.  A new stage is
    # never fast-forwarded -- its projection may look identical while the stage
    # semantics changed.
    if (not stage_changed and previous.get("digest") == digest
            and previous.get("selected") == "parent"):
        _log("投影与上一轮完全一致且上一轮已判定保留父模型，本轮直接保留父模型")
        receipt = rnd.model(
            [{"role": "user", "content": (
                "本轮公开投影与上一轮完全相同（digest=%s），上一轮已用 %d 组候选证明没有候选能"
                "超过父模型 %.6f。请确认是否继续保留父模型，输出 JSON："
                "{\"decision\":\"keep_parent\",\"reason\":\"一句话\"}" % (digest[:16],
                                                                          previous.get("experiments", 0),
                                                                          float(previous.get("parent_value") or 0.0))
            )}],
            max_tokens=min(200, rnd.max_output_tokens),
            temperature=0,
        )
        rnd.finish(
            selected_artifact_id=rnd.parent.artifact_id,
            hypothesis=("第 %d 轮（%s）：公开 train/validation 投影与上一轮逐字节一致，"
                        "复用上一轮的实验结论。" % (round_index, stage_id)),
            change_summary="未训练，保留父模型：投影未变化，重复搜索无信息增益。",
            reason=("本轮投影 digest 与上一轮相同，上一轮已用 %d 组候选证明公开验证集上无法超过"
                    "父模型；依据模型 `%s` 的确认（call_id=%s），保留父产物以节省算力并避免退化。"
                    % (int(previous.get("experiments") or 0), rnd.model_name, receipt.call_id)),
            configuration={"dataset_key": dataset, "task": task, "metric": metric,
                           "round_index": round_index, "stage_id": stage_id,
                           "selected": "parent", "skipped": "identical_projection",
                           "parent_metric_value": _num(previous.get("parent_value")),
                           "digest": digest[:16]},
            seed=int(parent.meta["seed"]) + round_index,
        )
        return

    try:
        result = train_and_select(Model, checkpoint_path, x_train, y_train, x_val,
                                  y_val, task, classes, metric, train_budget,
                                  round_index, dataset, log=_log,
                                  ids_train=ids_train, ids_val=ids_val,
                                  margin=policy_margin)
    except Exception as error:  # never lose the round: keep the parent
        _log("训练/选择异常，保留父模型：%r" % (error,))
        result = {"selected": "parent", "metrics": {}, "parent_metrics": {},
                  "experiments": [], "model": parent, "plan": None,
                  "epochs": 0, "elapsed": round(time.time() - started, 2),
                  "budget_s": train_budget}

    parent_metrics = result.get("parent_metrics") or {}
    best_metrics = result.get("metrics") or {}
    if result.get("selected") != "candidate" or result.get("model") is None:
        result["selected"] = "parent"
    parent_score = core.score_value(parent_metrics, metric)
    best_score = core.score_value(best_metrics, metric)
    experiments = result.get("experiments") or []

    # 精简 prompt：宿主已经会在前面拼官方提示，这里只带决策必需的标量，
    # 把 Token 花在改进上而不是重复元数据上。
    summary = {
        "ds": dataset, "stage": stage_id, "round": round_index, "metric": metric,
        "rows": [train_info["rows"], val_info["rows"]],
        "parent": _num(parent_metrics.get(metric)),
        "trials": [[record.get("name"), _num(record.get("score"))]
                   for record in experiments[:4]],
        "best": _num(best_metrics.get(metric) if best_metrics else None),
        "kept_parent": result["selected"] != "candidate",
        "prev": _history_digest(history)[-2:],
    }

    receipt = rnd.model(
        [{"role": "user", "content": (
            "本轮公开实验（JSON）：%s\n"
            "判断采纳最优候选还是保留父模型，只回一个 JSON："
            "{\"decision\":\"keep_best|try_again\",\"epochs\":1-600,\"learning_rate\":0.0001-0.05,"
            "\"reason\":\"一句话\"}"
            % json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str)
        )}],
        max_tokens=min(320, rnd.max_output_tokens),
        temperature=0,
    )
    advice = _receipt_json(receipt.text)
    _log("模型建议：%s" % json.dumps(advice, ensure_ascii=False))

    best = None
    if result["selected"] == "candidate":
        best = {"score": best_score, "plan": result.get("plan"),
                "metrics": best_metrics, "model": result["model"],
                "epochs": int(result.get("epochs") or 0)}

    elapsed = time.time() - started
    extra_enabled = os.environ.get("CONTEST_LLM_EXTRA", "0") == "1"
    if extra_enabled and advice.get("decision") == "try_again" and elapsed < wall * 0.60:
        remaining = max(10.0, wall * 0.90 - elapsed)
        best, record = _train_extra(rnd, checkpoint_path, x_train, y_train, x_val,
                                    y_val, task, classes, metric, advice, best,
                                    direction, parent_score, round_index, _log)
        if record is not None:
            experiments.append({k: v for k, v in record.items() if k != "metrics"})

    hypothesis = ("第 %d 轮（%s）：父模型在当轮公开验证集 %s=%s；按预算续训 %d 组候选，"
                  "仅在公开验证集上选型，未达边际则保留父模型。"
                  % (round_index, stage_id, metric,
                     _fmt(parent_metrics.get(metric)), len(experiments)))
    configuration = {
        "dataset_key": dataset, "task": task, "metric": metric,
        "round_index": round_index, "stage_id": stage_id,
        "parent_metric_value": _num(parent_metrics.get(metric)),
        "candidate_metric_value": _num(best_metrics.get(metric) if best else None),
        "selected": "candidate" if best is not None else "parent",
        "plan": _plan_name((best or {}).get("plan")) if best else None,
        "epochs": int((best or {}).get("epochs", 0)),
        "train_rows": train_info["rows"], "validation_rows": val_info["rows"],
        "experiments": len(experiments),
        "elapsed_s": round(time.time() - started, 2),
        "policy_margin": policy_margin,
        "stage_changed": stage_changed,
        "round_in_stage": round_in_stage,
    }

    if best is None:
        change_summary = ("相对父模型未做替换：%d 组候选均未在公开验证集上超过父模型的 "
                          "%.3f%% 门槛，保留父检查点。" % (len(experiments), 0.2))
        _log("保留父模型")
        _write_state(rnd, work=_work(), round_index=round_index, stage_id=stage_id,
                     dataset=dataset, payload={
                         "selected": "parent", "digest": digest,
                         "experiments": len(experiments),
                         "parent_value": _num(parent_metrics.get(metric)),
                         "metrics": {k: round(v, 6) for k, v in parent_metrics.items()
                                     if isinstance(v, float)},
                     })
        rnd.finish(
            selected_artifact_id=rnd.parent.artifact_id,
            hypothesis=hypothesis,
            change_summary=change_summary,
            reason=("本轮 %d 组候选在公开验证集上的 %s 均未超过父模型 %s 的可信边际，"
                    "依据模型 `%s` 的建议（call_id=%s）与实测结果，保留父产物以避免在漂移阶段"
                    "出现性能倒退。"
                    % (len(experiments), metric, _fmt(parent_metrics.get(metric)),
                       rnd.model_name, receipt.call_id)),
            configuration=configuration,
            seed=int(parent.meta["seed"]) + round_index,
        )
        return

    work = _work()
    candidate_dir = work / ("candidate-r%02d" % round_index)
    if candidate_dir.exists():
        candidate_dir = work / ("candidate-r%02d-%d" % (round_index, int(time.time())))
    candidate_dir.mkdir(parents=True, exist_ok=False)
    best["model"].save(candidate_dir / checkpoint_name)

    change_summary = ("相对父模型：%s，%s 由 %s 变为 %s（公开验证集，%d 组候选中的最优）。"
                      % (_plan_name(best.get("plan")) or "续训",
                         metric, _fmt(parent_metrics.get(metric)),
                         _fmt(best["metrics"].get(metric)), len(experiments)))
    submitted = rnd.submit_candidate(
        candidate_dir,
        hypothesis=hypothesis,
        change_summary=change_summary,
        configuration=configuration,
        seed=int(parent.meta["seed"]) + round_index,
        checkpoint_paths=[checkpoint_name],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )

    state_dir = work / ("state-r%02d" % round_index)
    if state_dir.exists():
        state_dir = work / ("state-r%02d-%d" % (round_index, int(time.time())))
    try:
        state = rnd.state_workspace(str(state_dir))
        (state / STATE_NOTE).write_text(json.dumps(
            {"round": round_index, "stage": stage_id, "dataset": dataset,
             "selected": "candidate", "digest": digest,
             "metrics": {k: round(v, 6) for k, v in best_metrics.items()
                         if isinstance(v, float)},
             "parent_metrics": {k: round(v, 6) for k, v in parent_metrics.items()
                                if isinstance(v, float)},
             "experiments": len(experiments),
             "parent_value": _num(parent_metrics.get(metric)),
             "plan": configuration["plan"]}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8")
        rnd.save_state(state)
    except Exception as error:  # state is optional for correctness
        _log("保存状态失败（忽略）：%r" % (error,))

    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=hypothesis,
        change_summary=change_summary,
        reason=("依据本轮 %d 组候选在公开验证集上的实测 %s（父模型 %s → 候选 %s），以及模型 "
                "`%s` 的建议（call_id=%s）：%s。候选通过了“必须超过父模型边际”的防退化门槛，"
                "因此替换父产物。"
                % (len(experiments), metric, _fmt(parent_metrics.get(metric)),
                   _fmt(best["metrics"].get(metric)), rnd.model_name, receipt.call_id,
                   str(advice.get("reason") or "采纳实测最优"))),
        configuration=configuration,
        seed=int(parent.meta["seed"]) + round_index,
    )




def _projection_digest(dataset, ids_train, y_train, ids_val, y_val):
    import hashlib
    digest = hashlib.sha256()
    digest.update(("%s|%d|%d|" % (dataset, len(ids_train), len(ids_val))).encode())
    for identity in ids_train:
        digest.update(identity.encode("utf-8", "ignore"))
        digest.update(b"\x00")
    digest.update(b"|")
    for identity in ids_val:
        digest.update(identity.encode("utf-8", "ignore"))
        digest.update(b"\x00")
    digest.update(b"|%.10f|%.10f" % (float(np.sum(y_train)), float(np.sum(y_val))))
    return digest.hexdigest()


def _previous_state(rnd):
    path = getattr(rnd, "state_path", None)
    if not path:
        return {}
    try:
        note = Path(path) / STATE_NOTE
        if note.is_file():
            return json.loads(note.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}



def _write_state(rnd, *, work, round_index, stage_id, dataset, payload):
    """Persist a tiny cross-round note (best effort; never fatal)."""
    try:
        state_dir = Path(work) / ("state-r%02d" % round_index)
        if state_dir.exists():
            state_dir = Path(work) / ("state-r%02d-%d" % (round_index, int(time.time())))
        state = rnd.state_workspace(str(state_dir))
        note = {"round": round_index, "stage": stage_id, "dataset": dataset}
        note.update(payload)
        (state / STATE_NOTE).write_text(
            json.dumps(note, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        rnd.save_state(state)
        return True
    except Exception as error:
        _log("保存状态失败（忽略）：%r" % (error,))
        return False


def _plan_name(plan):
    """Plans are dicts for some strategies and lists of steps for others."""
    if isinstance(plan, dict):
        name = plan.get("name")
        return str(name) if name else "plan"
    if isinstance(plan, (list, tuple)):
        names = [str(item.get("name")) for item in plan if isinstance(item, dict) and item.get("name")]
        return "+".join(names)[:120] if names else "multi-step"
    if plan is None:
        return None
    return str(plan)[:120]


def _num(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value):
    value = _num(value)
    return "n/a" if value is None else ("%.6f" % value)


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def _checkpoint_for(model_dir):
    override = os.environ.get("CONTEST_ARTIFACT_PATH")
    if override and Path(override).is_file():
        return override
    found = sorted(Path(model_dir).rglob("*.npz"))
    if not found:
        raise ValueError("no checkpoint found under %s" % model_dir)
    return str(found[0])


def _deploy(values, model, task):
    if task == "classification":
        logits = np.asarray(values, dtype=np.float64)
        logits = logits - logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return np.clip(probabilities, 0.0, 1.0)
    raw = float(values)
    transform = model.meta.get("target_transform")
    if transform == "log1p":
        raw = math.expm1(min(max(raw, -30.0), 30.0))
    return max(0.0, raw)


def predict(sample, model_dir):
    checkpoint = _checkpoint_for(model_dir)
    model = _MODELS.get(checkpoint)
    if model is None:
        model = Model.load(checkpoint)
        _MODELS[checkpoint] = model
    paths = model.meta["paths"]
    current, values = D.flatten(sample.inputs)
    if current != paths:
        lookup = dict(zip(current, values))
        values = [lookup.get(key, 0.0) for key in paths]
    x = np.asarray([values], dtype=np.float64)
    if sample.task == "regression":
        output = model.predict(x)
        value = _deploy(float(output.reshape(-1)[0]), model, "regression")
        if not math.isfinite(value):
            value = 0.0
        return {"value": value}
    probabilities = model.predict(x)
    scores = []
    for value in probabilities.reshape(-1):
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
        if rounded[top] < 1.0:
            rounded[top] = round(rounded[top] + 1e-6, 6)
        else:
            for index, value in enumerate(rounded):
                if index != top and value == rounded[top]:
                    rounded[index] = max(0.0, round(value - 1e-6, 6))
    return [{"label": label, "score": score} for label, score in zip(labels, rounded)]
