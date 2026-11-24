"""Per-dataset candidate plans and the shared train/select driver.

The three datasets share one driver; only ``PLANS`` differs.  Every plan is a
plain dict so the harness can log it verbatim into ``configuration``.
"""
import math
import time

import numpy as np

from .core import (better, evaluate, fit_ext, fold_shrink, inverse_frequency,
                   score_value)

# --------------------------------------------------------------------------- #
# default plans (measured / to be replaced by dataset specialists' recipes)
# --------------------------------------------------------------------------- #
PLANS = {
    "A": [
        {"name": "balanced_gauss05_cos", "epochs": 160, "lr": 4e-3,
         "weight": "invfreq", "noise": 0.5, "cosine": True, "margin": 0.002},
        {"name": "balanced_gauss03_cos", "epochs": 250, "lr": 3e-3,
         "weight": "invfreq", "noise": 0.3, "cosine": True, "margin": 0.002},
    ],
    "B": [
        {"name": "shrink045_cos", "epochs": 120, "lr": 1e-2,
         "weight": None, "noise": 0.0, "cosine": True, "shrink": 0.45,
         "margin": 0.006},
        {"name": "shrink030_cos", "epochs": 120, "lr": 3e-3,
         "weight": None, "noise": 0.0, "cosine": True, "shrink": 0.30,
         "margin": 0.006},
    ],
    "C": [
        {"name": "plain_cos300", "epochs": 300, "lr": 1e-3,
         "weight": None, "noise": 0.0, "cosine": True, "margin": 0.008},
        {"name": "plain_cos150_w150x2", "epochs": 150, "lr": 2e-3,
         "weight": "rul150x2", "noise": 0.0, "cosine": True, "margin": 0.008},
    ],
}


def plans_for(dataset_key):
    try:
        from . import tuned  # optional overrides produced by the specialists
        override = tuned.PLANS.get(dataset_key)
        if override:
            return override
    except Exception:
        pass
    return PLANS[dataset_key]


# --------------------------------------------------------------------------- #
def sample_weights(kind, y, task, classes):
    if not kind:
        return None
    if kind == "invfreq" and task == "classification":
        return inverse_frequency(y, len(classes))
    if kind.startswith("rul"):
        threshold, factor = kind[3:].split("x")
        base = np.ones(len(y), dtype=np.float64)
        base[y > float(threshold)] = float(factor)
        return base
    return None


def widen_hidden(model, width):
    """Function-preserving widening of the single hidden layer."""
    current = int(model.meta["hidden"])
    if width <= current:
        return False
    pad = width - current
    model.params["w1"] = np.concatenate(
        [model.params["w1"], np.zeros((model.params["w1"].shape[0], pad))], axis=1)
    model.params["b1"] = np.concatenate([model.params["b1"], np.zeros(pad)])
    model.params["w2"] = np.concatenate(
        [model.params["w2"], np.zeros((pad, model.params["w2"].shape[1]))], axis=0)
    for group in (model.first, model.second):
        group["w1"] = np.concatenate([group["w1"], np.zeros((group["w1"].shape[0], pad))], axis=1)
        group["b1"] = np.concatenate([group["b1"], np.zeros(pad)])
        group["w2"] = np.concatenate([group["w2"], np.zeros((pad, group["w2"].shape[1]))], axis=0)
    model.meta["hidden"] = width
    model.meta["parameter_count"] = int(sum(v.size for v in model.params.values()))
    model.validate()
    return True


def _apply_target_transform(model, y, transform):
    if transform == "log1p":
        model.meta["target_transform"] = "log1p"
        return np.log1p(np.clip(y, 0.0, None))
    model.meta.pop("target_transform", None)
    return y


_IMPL_CACHE = {}


def _impl_module(dataset_key):
    """Optional per-dataset specialist module (``a_impl``/``b_impl``/``c_impl``)."""
    if dataset_key in _IMPL_CACHE:
        return _IMPL_CACHE[dataset_key]
    module = None
    try:
        if dataset_key == "A":
            from . import a_impl as module
        elif dataset_key == "B":
            from . import b_impl as module
        elif dataset_key == "C":
            from . import c_impl as module
    except Exception:
        module = None
    _IMPL_CACHE[dataset_key] = module
    return module


def train_and_select(model_cls, parent_path, x_train, y_train, x_val, y_val,
                     task, classes, metric, budget_s, seed, dataset_key,
                     log=print, monotone=False, ids_train=None, ids_val=None):
    """Train a handful of candidates and pick the best deployable one.

    Returns a dict with the winning model; ``selected`` is ``"parent"`` when
    nothing beat the parent model by the required margin.
    """
    import inspect

    module = _impl_module(dataset_key)
    if module is not None and hasattr(module, "train_and_select_xy"):
        function = module.train_and_select_xy
        parameters = inspect.signature(function).parameters
        kwargs = {}
        if "ids_train" in parameters:
            kwargs["ids_train"] = ids_train
            kwargs["ids_val"] = ids_val
        try:
            out = function(model_cls, parent_path, x_train, y_train, x_val,
                           y_val, task, classes, metric, budget_s, seed,
                           log=log, **kwargs)
        except TypeError:
            out = function(model_cls, parent_path, x_train, y_train, x_val,
                           y_val, task, classes, metric, budget_s, seed,
                           **kwargs)
        out.setdefault("experiments", [])
        out.setdefault("plan", None)
        out.setdefault("epochs", 0)
        out.setdefault("budget_s", float(budget_s))
        out.setdefault("metrics", {})
        out.setdefault("parent_metrics", {})
        out["source"] = getattr(module, "__name__", "impl")
        return out

    started = time.time()
    deadline = started + max(20.0, float(budget_s))
    direction = "higher" if metric == "macro_f1" else "lower"

    parent = model_cls.load(str(parent_path))
    parent_metrics = evaluate(parent, x_val, y_val, task, len(classes))
    parent_score = score_value(parent_metrics, metric)
    log("父模型 %s=%s" % (metric, parent_metrics.get(metric)))

    plans = plans_for(dataset_key)
    experiments, best = [], None
    for index, plan in enumerate(plans):
        remaining = deadline - time.time()
        if remaining <= 5:
            log("预算不足，跳过剩余候选")
            break
        share = remaining / float(len(plans) - index)
        candidate = model_cls.load(str(parent_path))
        candidate.meta["seed"] = int((parent.meta["seed"] + seed) % (2 ** 32))
        if plan.get("widen"):
            widen_hidden(candidate, int(plan["widen"]))
        target_y = _apply_target_transform(candidate, y_train, plan.get("target"))
        weights = sample_weights(plan.get("weight"), y_train, task, classes)
        plan_started = time.time()
        done_epochs = 0
        chunk = 10
        try:
            while done_epochs < int(plan["epochs"]):
                left = min(chunk, int(plan["epochs"]) - done_epochs)
                if time.time() + 0.5 > plan_started + share:
                    break
                fit_ext(candidate, x_train, target_y, left, lr=plan["lr"],
                        weights=weights, noise=plan.get("noise") or 0.0,
                        cosine=plan.get("cosine", True), huber=plan.get("huber"))
                done_epochs += left
        except ValueError as error:  # pragma: no cover - defensive
            log("候选 %s 失败：%s" % (plan["name"], error))
            continue
        if done_epochs == 0:
            log("候选 %s 未获得训练时间" % plan["name"])
            continue
        if task == "regression" and plan.get("shrink"):
            center = float(np.median(y_train))
            fold_shrink(candidate, float(plan["shrink"]), center)
        metrics = evaluate(candidate, x_val, y_val, task, len(classes))
        value = score_value(metrics, metric)
        record = {"name": plan["name"], "epochs": done_epochs, "lr": plan["lr"],
                  "weight": plan.get("weight"), "noise": plan.get("noise"),
                  "shrink": plan.get("shrink"), "target": plan.get("target"),
                  "widen": plan.get("widen"), "seconds": round(time.time() - plan_started, 2),
                  "metrics": {k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in metrics.items() if k != "per_class_f1"}}
        record["score"] = value
        experiments.append(record)
        log("候选 %s -> %s=%s (%.1fs, %d ep)" % (plan["name"], metric, value,
                                                 record["seconds"], done_epochs))
        margin = float(plan.get("margin", 0.002))
        if better(direction, value, parent_score, margin):
            if best is None or is_better(direction, value, best["score"]):
                best = {"score": value, "plan": plan, "metrics": metrics,
                        "model": candidate, "epochs": done_epochs}

    selected = "parent"
    chosen = parent, parent_metrics
    if best is not None:
        selected, chosen = "candidate", (best["model"], best["metrics"])
    return {
        "checkpoint_path": None,
        "selected": selected,
        "metrics": chosen[1],
        "parent_metrics": parent_metrics,
        "experiments": experiments,
        "model": chosen[0],
        "plan": best["plan"] if best else None,
        "epochs": best["epochs"] if best else 0,
        "elapsed": round(time.time() - started, 2),
        "budget_s": float(budget_s),
        "source": "core",
    }


def is_better(direction, value, reference):
    return value > reference if direction == "higher" else value < reference
