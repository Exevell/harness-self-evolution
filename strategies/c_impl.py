"""C-task (turbofan RUL) one-round continue-training strategy.

Deterministic, numpy-only, designed for the per-round budget (~150 s).
Resumes from the round's parent checkpoint (never re-initialises weights),
trains on the *full* current public train projection, evaluates the parent and
the candidate on the current public validation projection, and only returns the
candidate when it clearly beats the parent.

Public API (see work/CONTEXT.md section 7 and the harness request):

    train_and_select_xy(model_cls, parent_path, x_train, y_train, x_val, y_val,
                        task, classes, metric, budget_s, seed, log=print) -> dict
        Array interface: returns {"model": <trained Model>, "selected": ...,
        "parent_metrics": ..., "metrics": ..., "experiments": [...], "plan": {...},
        "epochs": int, ...}; the caller persists the model.  No file parsing.

    train_and_select(model_cls, parent_ckpt_path, train_rows, val_rows,
                     task, classes, metric, budget_s, seed, log=print) -> dict
        Row interface: accepts JSONL paths / contest row dicts / {"x","y"} dicts and
        writes the candidate checkpoint itself (temp dir) -> "checkpoint_path".

Both share one round rule, driven by the platform evidence that every previous C
round under-performed the *parent* on the same val projection:
  1. evaluate the parent on THIS round's val projection first;
  2. continue-train the candidate on this round's full train projection;
  3. submit only if candidate RMSE beats the parent by >= rmse_margin (1 %) AND
     does not regress the NASA asymmetric score; otherwise selected="parent".
`meta` is NEVER read into the model: only `inputs`/`answer` are used.
"""
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

PARAMETERS = ("w1", "b1", "w2", "b2")

# ---------------------------------------------------------------- tuned plan
# Measured on the C package public train/validation projection (see report.md):
# batch 512, cosine 3e-3 -> 1e-6, 600 epochs = "one round" and the strongest
# deployable setting inside 150 s (local wall clock ~55 s).
PLAN = {
    "epochs": 600,
    "batch_size": 256,
    "learning_rate": 3e-3,
    "lr_min": 1e-6,
    "loss": "mse",
    "weight_threshold": 125.0,     # long-life emphasis hook: w=1 below, w=factor above
    "weight_factor": 1.0,          # 1.0 = off (RMSE-optimal); 2.0 = NASA-leaning variant
    "widen_to": 0,                 # 0 = keep parent width; 128 = zero-fill widen
    "rmse_margin": 0.01,           # candidate must beat parent RMSE by >= 1 % (strict gate)
    "nasa_guard": 1.00,            # candidate NASA sum must not exceed parent's
    "time_fraction": 0.85,         # never spend more than this fraction of budget
    "chunk_epochs": 25,
}


# ---------------------------------------------------------------- metric
def _metrics(pred, y):
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    nasa = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "nasa_sum": float(np.sum(nasa)), "nasa_mean": float(np.mean(nasa)),
            "bias": float(np.mean(err)), "n": int(len(err))}


def _predict(model, x):
    z = model.normalized(x)
    hidden = np.tanh(z @ model.params["w1"] + model.params["b1"])
    logit = (hidden @ model.params["w2"] + model.params["b2"])[:, 0]
    return np.maximum(0.0, logit * model.meta["target_scale"] + model.meta["target_mean"])


# ---------------------------------------------------------------- row handling
def _parse_paths(paths):
    out = []
    for path in paths:
        decoded = json.loads(path) if isinstance(path, str) else path
        out.append([(step[0], step[1]) for step in decoded])
    return out


def _walk(value, steps):
    for kind, key in steps:
        value = value[key]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError("non-numeric input")


def _read_jsonl(path, model):
    parsed = _parse_paths(model.meta["paths"])
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            xs.append([_walk(row["inputs"], steps) for steps in parsed])
            ys.append(float(row["answer"]))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _to_xy(rows, model):
    n_features = len(model.meta["paths"])
    if isinstance(rows, (str, bytes, os.PathLike)):
        return _read_jsonl(Path(rows), model)
    if isinstance(rows, dict):
        if "x" in rows and "y" in rows:
            return np.asarray(rows["x"], dtype=np.float64), np.asarray(rows["y"], dtype=np.float64)
        if "rows" in rows:
            return _to_xy(rows["rows"], model)
        raise ValueError("row dict needs x/y")
    rows = list(rows)
    if not rows:
        raise ValueError("empty rows")
    first = rows[0]
    if isinstance(first, dict):
        if "x" in first and "y" in first:
            return (np.asarray([r["x"] for r in rows], dtype=np.float64),
                    np.asarray([r["y"] for r in rows], dtype=np.float64))
        parsed = _parse_paths(model.meta["paths"])
        xs = [[_walk(r["inputs"], steps) for steps in parsed] for r in rows]
        ys = [float(r["answer"]) for r in rows]
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == n_features:
        raise ValueError("feature matrix passed without targets; pass dict(x=..., y=...)")
    raise ValueError("unsupported row container")


# ---------------------------------------------------------------- weights
def _sample_weights(y, threshold, factor):
    if not threshold or factor is None or float(factor) == 1.0:
        return None
    w = np.ones(len(y), dtype=np.float64)
    w[y > float(threshold)] = float(factor)
    return w


# ---------------------------------------------------------------- training
def _copy_state(model):
    return {"params": {k: v.copy() for k, v in model.params.items()},
            "first": {k: v.copy() for k, v in model.first.items()},
            "second": {k: v.copy() for k, v in model.second.items()},
            "meta": dict(model.meta)}


def _restore_state(model, snap):
    model.params = {k: v.copy() for k, v in snap["params"].items()}
    model.first = {k: v.copy() for k, v in snap["first"].items()}
    model.second = {k: v.copy() for k, v in snap["second"].items()}
    model.meta = dict(snap["meta"])
    model.validate()


def _widen(model, new_hidden, seed):
    """Function-preserving 32->128 migration; keeps the parent's prediction."""
    old = model.params["w1"].shape[1]
    if new_hidden <= old:
        return False
    if new_hidden > 128:
        raise ValueError("hidden width above the 128 cap")
    in_dim = model.params["w1"].shape[0]
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    for group in (model.params, model.first, model.second):
        w1 = np.zeros((in_dim, new_hidden), dtype=np.float64)
        b1 = np.zeros(new_hidden, dtype=np.float64)
        w2 = np.zeros((new_hidden, 1), dtype=np.float64)
        w1[:, :old] = group["w1"]
        b1[:old] = group["b1"]
        w2[:old, :] = group["w2"]
        group["w1"], group["b1"], group["w2"] = w1, b1, w2
    # random fan-in for the new units, output weights 1e-2 of their init scale:
    # |prediction change| < 1 cycle, but the new columns are not frozen at zero.
    model.params["w1"][:, old:] = rng.normal(0.0, math.sqrt(1.0 / in_dim), (in_dim, new_hidden - old))
    model.params["w2"][old:, :] = rng.normal(0.0, 1e-2 / math.sqrt(new_hidden), (new_hidden - old, 1))
    model.meta["hidden"] = int(new_hidden)
    model.meta["parameter_count"] = int(in_dim * new_hidden + new_hidden + new_hidden + 1)
    model.validate()
    return True


def _train(model, x, y, *, epochs, batch, learning_rate, lr_min, weights,
           deadline, chunk_epochs, on_epoch):
    """Cosine-annealed Adam continuation; stops early only if the wall clock
    budget would otherwise be exceeded (the planned epoch count always fits on
    the reference machine, so behaviour there is fully deterministic)."""
    zn = model.normalized(x)
    target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    w = np.ones(len(y), dtype=np.float64) if weights is None else weights
    params = model.params
    trained = 0
    for epoch in range(epochs):
        frac = 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs - 1)))
        model.meta["learning_rate"] = float(max(lr_min, lr_min + (learning_rate - lr_min) * frac))
        cur_lr = model.meta["learning_rate"]
        rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
        order = rng.permutation(len(zn))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            zb, tb, wb = zn[idx], target[idx], w[idx]
            hidden = np.tanh(zb @ params["w1"] + params["b1"])
            logit = (hidden @ params["w2"] + params["b2"])[:, 0]
            residual = logit - tb
            wsum = float(wb.sum())
            if wsum <= 0:
                wsum = float(len(idx))
            delta = (2.0 * wb * residual / wsum)[:, None]
            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ params["w2"].T) * (1.0 - hidden ** 2)
            gradients["w1"] = zb.T @ hidden_delta
            gradients["b1"] = hidden_delta.sum(axis=0)
            model.meta["steps"] += 1
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient")
            for name, gradient in gradients.items():
                gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = 0.9 * model.first[name] + 0.1 * gradient
                model.second[name] = 0.999 * model.second[name] + 0.001 * gradient ** 2
                first = model.first[name] / (1 - 0.9 ** model.meta["steps"])
                second = model.second[name] / (1 - 0.999 ** model.meta["steps"])
                params[name] -= cur_lr * first / (np.sqrt(second) + 1e-8)
        model.meta["epochs"] += 1
        trained += 1
        on_epoch(model, epoch)
        if (trained % chunk_epochs == 0) and time.monotonic() > deadline:
            break
    model.validate()
    return trained


# ---------------------------------------------------------------- core round
def _fit_select(model_cls, parent_path, x_tr, y_tr, x_va, y_va, budget_s, seed, log, plan):
    """One round: evaluate parent, continue-train a candidate on the *current*
    projection, pick the best non-NASA-regressing epoch, decide candidate/parent.
    Shares every use of randomness with Model.fit (seed, cumulative epoch)."""
    started = time.monotonic()
    budget = float(budget_s)
    deadline = started + budget * float(plan["time_fraction"])
    experiments = []
    parent = model_cls.load(str(parent_path))
    x_tr = np.asarray(x_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64).ravel()
    x_va = np.asarray(x_va, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64).ravel()
    if x_tr.shape[1] != len(parent.meta["paths"]) or x_va.shape[1] != x_tr.shape[1]:
        raise ValueError("feature width does not match the parent checkpoint")
    if len(x_tr) == 0 or len(x_va) == 0 or len(y_tr) != len(x_tr) or len(y_va) != len(x_va):
        raise ValueError("empty or inconsistent projection")

    parent_metrics = _metrics(_predict(parent, x_va), y_va)
    log(f"[c] parent: rmse={parent_metrics['rmse']:.4f} mae={parent_metrics['mae']:.4f} "
        f"nasa_sum={parent_metrics['nasa_sum']:.3e} (n_train={len(x_tr)}, n_val={len(x_va)})")

    # ---- candidate: resume the parent (never re-initialise); widen only if asked
    model = model_cls.load(str(parent_path))
    widened = False
    if plan["widen_to"]:
        target_hidden = 128 if plan["widen_to"] == 128 else int(plan["widen_to"])
        widened = _widen(model, target_hidden, seed)
        if widened:
            pre = _metrics(_predict(model, x_va), y_va)
            log(f"[c] widened {parent.meta['hidden']}->{model.meta['hidden']} hidden "
                f"(function-preserving check: rmse {pre['rmse']:.4f})")

    weights = _sample_weights(y_tr, plan["weight_threshold"], plan["weight_factor"])
    model.meta["batch_size"] = int(plan["batch_size"])
    epochs = int(plan["epochs"])

    best = {"state": _copy_state(model), "metrics": None, "epoch": 0}
    guard_floor = parent_metrics["nasa_sum"] * float(plan["nasa_guard"])

    def on_epoch(m, epoch_index):
        m_metrics = _metrics(_predict(m, x_va), y_va)
        rel = m.meta["epochs"] - parent.meta["epochs"]
        m_metrics["epoch"] = rel
        if rel % 25 == 0 or rel == 1 or rel == epochs:
            experiments.append({"epoch": rel, "rmse": m_metrics["rmse"], "mae": m_metrics["mae"],
                                "nasa_sum": m_metrics["nasa_sum"]})
        # best RMSE epoch that does not regress the asymmetric score
        if m_metrics["nasa_sum"] <= guard_floor and (
                best["metrics"] is None or m_metrics["rmse"] < best["metrics"]["rmse"]):
            best.update(state=_copy_state(m), metrics=m_metrics, epoch=rel)

    trained = _train(model, x_tr, y_tr, epochs=epochs, batch=int(plan["batch_size"]),
                     learning_rate=float(plan["learning_rate"]), lr_min=float(plan["lr_min"]),
                     weights=weights, deadline=deadline,
                     chunk_epochs=int(plan["chunk_epochs"]), on_epoch=on_epoch)
    elapsed = time.monotonic() - started

    if best["metrics"] is None:
        best = {"state": _copy_state(model), "metrics": _metrics(_predict(model, x_va), y_va),
                "epoch": trained}
    _restore_state(model, best["state"])
    candidate_metrics = best["metrics"] if best["metrics"] is not None else \
        _metrics(_predict(model, x_va), y_va)
    improved = candidate_metrics["rmse"] <= parent_metrics["rmse"] * (1.0 - float(plan["rmse_margin"]))
    nasa_ok = candidate_metrics["nasa_sum"] <= guard_floor
    selected = "candidate" if (improved and nasa_ok) else "parent"
    reason = ("candidate beats parent RMSE by >= %.2f%% without NASA regression"
              % (plan["rmse_margin"] * 100)) if selected == "candidate" else (
              "RMSE gain below %.2f%% margin" % (plan["rmse_margin"] * 100) if not improved
              else "NASA score would regress")
    log(f"[c] trained {trained} epochs in {elapsed:.1f}s "
        f"(cand rmse={candidate_metrics['rmse']:.4f} nasa={candidate_metrics['nasa_sum']:.3e}; "
        f"best epoch={best['epoch']}) -> selected={selected} [{reason}]")

    configuration = {
        "epochs": trained, "planned_epochs": epochs, "batch_size": int(plan["batch_size"]),
        "learning_rate": float(plan["learning_rate"]), "lr_min": float(plan["lr_min"]),
        "loss": plan["loss"], "weight_threshold": plan["weight_threshold"],
        "weight_factor": plan["weight_factor"], "hidden": model.meta["hidden"],
        "widened": widened, "elapsed_s": elapsed, "budget_s": budget,
        "rmse_margin": plan["rmse_margin"], "nasa_guard": plan["nasa_guard"],
        "selection": "best_epoch_by_val_rmse_with_nasa_guard",
        "deterministic": trained == epochs,
    }
    return {
        "model": model,
        "selected": selected,
        "parent_metrics": parent_metrics,
        "metrics": candidate_metrics,
        "experiments": experiments,
        "plan": configuration,
        "epochs": trained,
        "reason": reason,
        "configuration": configuration,
        "hypothesis": ("Continuing the parent on the full current public train projection with a "
                       "cosine-annealed Adam schedule reduces the compressed long-life bias, which "
                       "drives both RMSE and the asymmetric NASA penalty; best-epoch selection under "
                       "a NASA guard protects against drift-induced overfitting."),
        "change_summary": (f"full-projection continuation: epochs={trained}/{epochs}, "
                           f"batch={plan['batch_size']}, cosine lr {plan['learning_rate']:g}"
                           f"->{plan['lr_min']:g}, hidden={model.meta['hidden']}"
                           f"{' (widened)' if widened else ''}, best-epoch-by-val selection with "
                           f"NASA guard"
                           + (f", weight(y>{plan['weight_threshold']:g})={plan['weight_factor']:g}"
                              if float(plan["weight_factor"]) != 1.0 else ", no sample weighting")),
    }


# ---------------------------------------------------------------- public API
def train_and_select_xy(model_cls, parent_path, x_train, y_train, x_val, y_val,
                        task, classes, metric, budget_s, seed, log=print, **overrides):
    """Array interface (preferred by the harness): caller supplies matrices and
    persists the returned ``model`` itself.  numpy-only, deterministic, no I/O.

    If ``selected == "parent"`` the returned ``model`` is the *parent* checkpoint
    (functionally identical), so a caller that persists ``model`` unconditionally
    can never upload a candidate that is worse than the parent.
    """
    if str(task).lower() != "regression":
        raise ValueError("C strategy is regression-only")
    plan = dict(PLAN)
    plan.update({k: v for k, v in overrides.items() if k in plan})
    result = _fit_select(model_cls, parent_path, x_train, y_train, x_val, y_val,
                         budget_s, seed, log, plan)
    if result["selected"] == "parent":
        result["model"] = model_cls.load(str(parent_path))
    return result


def train_and_select(model_cls, parent_ckpt_path, train_rows, val_rows,
                     task, classes, metric, budget_s, seed, log=print, **overrides):
    """Row interface from work/CONTEXT.md section 7: accepts JSONL paths, contest
    row dicts or {"x","y"} dicts; persists the candidate itself to a temp dir."""
    if str(task).lower() != "regression":
        raise ValueError("C strategy is regression-only")
    plan = dict(PLAN)
    plan.update({k: v for k, v in overrides.items() if k in plan})
    parent = model_cls.load(str(parent_ckpt_path))
    x_tr, y_tr = _to_xy(train_rows, parent)
    x_va, y_va = _to_xy(val_rows, parent)
    result = _fit_select(model_cls, parent_ckpt_path, x_tr, y_tr, x_va, y_va,
                         budget_s, seed, log, plan)

    out_dir = tempfile.mkdtemp(prefix="c-strategy-")
    checkpoint_path = str(Path(out_dir) / "checkpoint.npz")
    if result["selected"] == "candidate":
        result["model"].save(checkpoint_path)
    else:
        # keep the parent authoritative, but leave a valid checkpoint for callers
        # that upload unconditionally
        model_cls.load(str(parent_ckpt_path)).save(checkpoint_path)
    # the row interface returns a JSON-serialisable dict (CONTEXT section 7): drop
    # the live model object, the caller uses checkpoint_path instead
    result.pop("model", None)
    result["checkpoint_path"] = checkpoint_path
    return result
