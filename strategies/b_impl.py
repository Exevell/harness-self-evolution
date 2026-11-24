"""B (rolling-bearing RUL, RMSE in minutes) - drift-stream per-round training/selection strategy.

CORE FINDING
------------
The target is  RUL_minutes = D_b - i  (D_b = whole life of bearing b in minutes, i = snapshot
index).  Absolute minutes are NOT identifiable across bearings: the 9 public bearings have
D_b in [123, 2538] and nothing in a single 34-feature snapshot reveals D_b (verified: per-bearing
predicted "level" is uncorrelated with D_b, see report.md section 2).  A model trained on absolute
minutes therefore only memorises the training bearings' levels; on an unseen bearing it is off by
hundreds of minutes and ends up *worse than a constant* (public-validation RMSE 681.16 vs 465.55
for the validation-mean constant, and 3 of the 10 rounds of the previous submission regressed).

So we regress the **life fraction**  p = y / D_b  and restore minutes with ONE scalar life-scale
prior c:  y_hat = c * p_hat.  This is exactly the affine target transform
    fit_target = c * y / D_b
which folds into the contract-fixed prediction path
    y_hat = max(0, logit * target_scale + target_mean)
so no change to model.py / predict.py is required (fully deployable).  Training is invariant to c
because standardisation cancels, so c is a pure post-hoc decision.

Deployability audit (what CANNOT be used, because Model.predict is frozen):
  * log1p / sqrt / power transforms of the *output* -> need a non-affine inverse at predict time;
  * log / robust *feature* transforms -> only affine (x-mean)/scale is available;
  * per-bearing or history features (snapshot index, total snapshots, id/meta) as model INPUTS.

Deployment must also be a real resume, so the recipe continues Adam from the parent weights on
the new (fraction) objective.  Safety first: the parent is always evaluated on the current public
validation projection, and the parent is kept unless the candidate is clearly better there.
"""
import json
import math
import os
import re
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------- hyper-parameters
C_LIFE = 1000.0          # life-scale prior in minutes (report.md section 4: plateau 900-1100)
EPOCHS = 250             # chosen by LOBO; 600/1200 epochs were worse on held-out bearings
N_SEEDS = 3
LR = 3e-3
BATCH = 128
SCHEDULE = "const"       # LOBO winner; "cosine" looks better on public val but is LOBO-worse
MIN_REL_GAIN = 0.05      # keep the parent unless the candidate is >=5% better on public val
MIN_IQR_RATIO = 0.35     # anti-collapse guard: predicted IQR / true IQR
USE_SNAPSHOT_INDEX = True  # only to recover a per-bearing life scale for TRUNCATED fragments


# ---------------------------------------------------------------- data adapters
def _flatten(value, path=None, out=None):
    if out is None:
        out = ([], [])
    path = path or []
    if isinstance(value, dict):
        for k in sorted(value):
            _flatten(value[k], path + [("key", k)], out)
    elif isinstance(value, list):
        for i, ch in enumerate(value):
            _flatten(ch, path + [("index", i)], out)
    else:
        out[0].append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
        out[1].append(float(value))
    return out[0], out[1]


def _as_xy(rows):
    """Accept a dict-of-arrays (data.read_dataset output) or a list of raw jsonl rows."""
    if isinstance(rows, dict):
        return (np.asarray(rows["x"], dtype=np.float64),
                np.asarray(rows["y"], dtype=np.float64),
                list(rows.get("ids") or ["row%d" % i for i in range(len(rows["y"]))]),
                list(rows["paths"]))
    xs, ys, ids, paths = [], [], [], None
    for r in rows:
        p, val = _flatten(r["inputs"])
        if paths is None:
            paths = p
        xs.append(val)
        ys.append(float(r["answer"]))
        ids.append(str(r.get("id", "row%d" % len(ids))))
    return np.asarray(xs, float), np.asarray(ys, float), ids, paths


def _snapshot_of(identity):
    m = re.match(r"^(.*)_s(\d+)$", identity)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def _groups(ids, y):
    """Split rows into bearings -> (group key per row, D_b per row = whole life in minutes).

    Grouping: (1) id prefix before '_s<digits>' (public convention); (2) contiguous runs where
    consecutive y differ by exactly -1 (inside one bearing y = D_b - i); (3) one group.

    Life scale D_b, in order of preference:
      a. the group contains a RUL==0 row  -> D_b = max(y)+1  (exact: last snapshot is RUL 0);
      b. the id carries the snapshot index -> D_b = y + i     (exact even for a truncated
         fragment; the index is used ONLY to recover the label scale, never as a model input);
      c. otherwise -> D_b = max(y)+1 (assumes a head fragment / complete bearing).
    """
    ok_ids = bool(ids) and all(re.match(r"^(.*)_s(\d+)$", i) for i in ids)
    if ok_ids:
        keys = [re.match(r"^(.*)_s(\d+)$", i).group(1) for i in ids]
        snaps = [int(re.match(r"^(.*)_s(\d+)$", i).group(2)) for i in ids]
    else:
        keys, snaps = None, None
    if keys is None or len(set(keys)) < 2:
        runs, cur = [], 0
        for i in range(1, len(y)):
            if abs((y[i - 1] - y[i]) - 1.0) > 1e-9:
                runs.append((cur, i))
                cur = i
        runs.append((cur, len(y)))
        keys = [None] * len(y)
        for gi, (a, b) in enumerate(runs):
            for i in range(a, b):
                keys[i] = "grp%d" % gi
        if len(runs) < 2:
            keys = ["grp0"] * len(y)
        if snaps is None:
            snaps = [None] * len(y)      # keep parsed snapshot indices when available
    keys = np.asarray(keys)
    y = np.asarray(y, dtype=np.float64)
    D = np.zeros(len(y), dtype=np.float64)
    for k in sorted(set(keys.tolist())):
        sel = keys == k
        yy = y[sel]
        if yy.min() <= 1e-9:                                   # complete bearing
            D[sel] = float(yy.max()) + 1.0
        elif USE_SNAPSHOT_INDEX and snaps is not None and all(s is not None for s in np.asarray(snaps)[sel]):
            cand = np.asarray(snaps, dtype=np.float64)[sel] + yy   # y + i  == D_b, constant per group
            D[sel] = float(np.median(cand))
        else:
            D[sel] = float(yy.max()) + 1.0
    D = np.maximum(D, 1.0)
    return keys, D


# ---------------------------------------------------------------- model helpers
def _normalize_stats(x):
    s = x.std(axis=0)
    return x.mean(axis=0), np.where(s < 1e-12, 1.0, s)


def _make_model(model_cls, paths, x, hidden, seed, target_mean, target_scale, init_params=None):
    m = model_cls("regression", list(paths), [], np.asarray(x, float), np.zeros(len(x)),
                  hidden=int(hidden), seed=int(seed))
    m.mean, m.scale = _normalize_stats(x)
    m.meta["target_mean"] = float(target_mean)
    m.meta["target_scale"] = float(target_scale)
    if init_params is not None:
        m.params = {k: np.asarray(v, dtype=np.float64).copy() for k, v in init_params.items()}
    m.validate()
    return m


def _train(m, x, fit_y, epochs, lr, batch_size, seed, schedule="const"):
    """Adam continuation; identical maths to baselines Model.fit, optional cosine decay."""
    X = np.clip((x - m.mean) / m.scale, -10.0, 10.0)
    t = (np.asarray(fit_y, float) - m.meta["target_mean"]) / m.meta["target_scale"]
    n = len(X)
    p, first, second = m.params, m.first, m.second
    steps = int(m.meta["steps"])
    epochs = int(epochs)
    for ep in range(epochs):
        if schedule == "cosine":
            cur_lr = 0.5 * lr * (1.0 + math.cos(math.pi * ep / max(epochs - 1, 1)))
        else:
            cur_lr = lr
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(m.meta["epochs"])]))
        order = rng.permutation(n)
        for st in range(0, n, batch_size):
            idx = order[st:st + batch_size]
            xb, tb = X[idx], t[idx]
            h = np.tanh(xb @ p["w1"] + p["b1"])
            err = (h @ p["w2"] + p["b2"])[:, 0] - tb
            g = 2.0 * err / len(idx)
            delta = g[:, None]
            grads = {"w2": h.T @ delta, "b2": delta.sum(axis=0)}
            hd = (delta @ p["w2"].T) * (1 - h ** 2)
            grads["w1"] = xb.T @ hd
            grads["b1"] = hd.sum(axis=0)
            gnorm = math.sqrt(sum(float(np.sum(v * v)) for v in grads.values()))
            if not math.isfinite(gnorm):
                raise ValueError("nonfinite gradient")
            steps += 1
            b1c, b2c = 1 - .9 ** steps, 1 - .999 ** steps
            for name in ("w1", "b1", "w2", "b2"):
                gr = grads[name] * min(1.0, 5.0 / max(gnorm, 1e-12))
                first[name] = .9 * first[name] + .1 * gr
                second[name] = .999 * second[name] + .001 * gr ** 2
                p[name] = p[name] - cur_lr * (first[name] / b1c) / (np.sqrt(second[name] / b2c) + 1e-8)
        m.meta["epochs"] += 1
        m.validate()
    m.meta["steps"] = steps
    return m


# ---------------------------------------------------------------- metrics / health
def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _metric(name, y, pred):
    return _mae(y, pred) if str(name).lower() in ("mae", "mean_absolute_error") else _rmse(y, pred)


def _health(y, pred, groups):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    yiqr = float(np.subtract(*np.percentile(y, [75, 25])))
    piqr = float(np.subtract(*np.percentile(pred, [75, 25])))
    biases = {}
    for g in sorted(set(np.asarray(groups).tolist())):
        k = np.asarray(groups) == g
        biases[str(g)] = float(np.mean(pred[k] - y[k]))
    return {"iqr_ratio": (piqr / yiqr if yiqr > 1e-9 else 1.0),
            "per_bearing_bias": biases,
            "const_equiv_rmse": _rmse(y, np.full(len(y), y.mean())),
            "pred_iqr": piqr, "true_iqr": yiqr}


# ---------------------------------------------------------------- core
def _core(model_cls, parent_path, x_tr, y_tr, x_va, y_va, ids_tr, ids_va,
          metric, budget_s, seed, log):
    t_start = time.time()
    x_tr = np.asarray(x_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    x_va = np.asarray(x_va, dtype=np.float64)
    y_va = np.asarray(y_va, dtype=np.float64)

    parent = model_cls.load(parent_path)
    paths = list(parent.meta["paths"])
    par_pred = np.maximum(0.0, parent.predict(x_va))
    par_val = _metric(metric, y_va, par_pred)
    keys_va, _ = _groups(ids_va, y_va)
    log("[b] parent public-val %s = %.3f" % (metric, par_val))

    # ---- per-bearing life scale of the training projection (pure target redefinition)
    keys_tr, D_tr = _groups(ids_tr, y_tr)
    n_groups = len(set(keys_tr.tolist()))
    n_frag = 0
    if len(D_tr):
        by = {}
        for k, d in zip(keys_tr.tolist(), D_tr.tolist()):
            by.setdefault(k, []).append(d)
        n_frag = sum(1 for k, ds in by.items() if len(set(np.round(ds, 6))) > 1)
    fit_y = C_LIFE * (y_tr / D_tr)              # = C_LIFE * p, still in minutes-space
    tm = float(fit_y.mean())
    ts = max(float(fit_y.std()), 1e-6)

    # keep the number of optimisation steps roughly constant for larger projections
    epochs = int(min(EPOCHS, max(150, round(EPOCHS * 7032.0 / max(len(y_tr), 1)))))
    seeds = N_SEEDS if len(y_tr) <= 14000 else 2
    if budget_s and budget_s < 45:
        epochs = min(epochs, 250)
        seeds = min(seeds, 2)

    models = []
    for si in range(seeds):
        s = int(seed) + 101 * si
        m = _make_model(model_cls, paths, x_tr, parent.meta["hidden"], s, tm, ts,
                        init_params=parent.params)
        _train(m, x_tr, fit_y, epochs, LR, BATCH, s, SCHEDULE)
        v = _metric(metric, y_va, np.maximum(0.0, m.predict(x_va)))
        models.append((v, s, m))
        log("[b] seed %d: val %s = %.3f" % (s, metric, v))
    models.sort(key=lambda t: t[0])
    best_val, best_seed, best = models[0]
    ens_val = _metric(metric, y_va,
                      np.mean([np.maximum(0.0, m.predict(x_va)) for _, _, m in models], axis=0))
    best_pred = np.maximum(0.0, best.predict(x_va))
    cand_health = _health(y_va, best_pred, keys_va)
    log("[b] best seed %d: %s=%.3f (ensemble %.3f); iqr_ratio=%.2f"
        % (best_seed, metric, best_val, ens_val, cand_health["iqr_ratio"]))

    gain = (1.0 - best_val / par_val) if par_val > 0 else 0.0
    healthy = cand_health["iqr_ratio"] >= MIN_IQR_RATIO
    use_cand = bool((gain >= MIN_REL_GAIN) and healthy)
    log("[b] gain %.1f%%, healthy=%s -> %s" % (100 * gain, healthy,
                                              "candidate" if use_cand else "parent"))

    plan = [
        {"step": "evaluate parent on current public validation", "value": par_val},
        {"step": "derive per-bearing life scale from training labels",
         "groups": n_groups, "inconsistent_scale_groups": n_frag},
        {"step": "continue Adam from parent on affine target c*y/D_b",
         "epochs": epochs, "seeds": seeds, "lr": LR, "c_life": C_LIFE, "schedule": SCHEDULE},
        {"step": "select best seed on public validation", "value": best_val,
         "iqr_ratio": cand_health["iqr_ratio"]},
        {"step": "keep parent unless candidate is >=%.0f%% better and not collapsed"
                 % (100 * MIN_REL_GAIN), "decision": "candidate" if use_cand else "parent"},
    ]
    return {
        "model": best if use_cand else None,
        "selected": "candidate" if use_cand else "parent",
        "metrics": {str(metric): best_val if use_cand else par_val,
                    "mae": _mae(y_va, best_pred) if use_cand else _mae(y_va, par_pred),
                    "ensemble_val": ens_val},
        "parent_metrics": {str(metric): par_val, "mae": _mae(y_va, par_pred)},
        "experiments": [{
            "name": "life_fraction_target+resume",
            "value": best_val,
            "parent_value": par_val,
            "params": {"c_life": C_LIFE, "epochs": epochs, "lr": LR, "batch_size": BATCH,
                       "schedule": SCHEDULE, "seeds": seeds,
                       "hidden": int(parent.meta["hidden"]),
                       "target_mean": tm, "target_scale": ts, "train_groups": n_groups},
            "seed_values": [v for v, _, _ in models],
            "health": cand_health,
        }],
        "plan": plan,
        "epochs": epochs,
        "hypothesis": ("Absolute-minute targets are dominated by an unidentifiable per-bearing life "
                       "scale; regressing the life fraction p=y/D_b and restoring minutes with a "
                       "single prior c removes the cross-bearing level error. Because c cancels in "
                       "standardisation, training is c-invariant and c is one post-hoc constant."),
        "change_summary": ("Resume the parent checkpoint and continue Adam for %d epochs x %d seeds "
                           "on the affine target c*y/D_b (c=%.0f); keep the parent unless public-val "
                           "%s improves by >=%.0f%% and the prediction IQR is not collapsed."
                           % (epochs, seeds, C_LIFE, metric, 100 * MIN_REL_GAIN)),
        "configuration": {"c_life": C_LIFE, "epochs": epochs, "n_seeds": seeds, "lr": LR,
                          "batch_size": BATCH, "schedule": SCHEDULE,
                          "hidden": int(parent.meta["hidden"]),
                          "target_mean": tm, "target_scale": ts, "train_groups": n_groups,
                          "val_groups": int(len(set(keys_va.tolist()))),
                          "inconsistent_scale_groups": n_frag,
                          "min_rel_gain": MIN_REL_GAIN, "min_iqr_ratio": MIN_IQR_RATIO,
                          "deployable": True, "seconds": time.time() - t_start},
    }


# ---------------------------------------------------------------- public entries
def train_and_select_xy(model_cls, parent_path, x_train, y_train, x_val, y_val,
                        task, classes, metric, budget_s, seed, log=print,
                        ids_train=None, ids_val=None):
    """Array interface: matrices in, trained Model object out (caller saves the checkpoint).

    Returns a dict with `model` (None when the parent is kept), `selected`,
    `parent_metrics`, `metrics`, `experiments`, `plan`, `epochs`.
    """
    out = _core(model_cls, parent_path, x_train, y_train, x_val, y_val,
                ids_train, ids_val, metric, budget_s, seed, log)
    out["configuration"]["interface"] = "xy"
    return out


def train_and_select(model_cls, parent_ckpt_path, train_rows, val_rows,
                     task, classes, metric, budget_s, seed, log=print):
    """Row interface (CONTEXT section 7): rows in, checkpoint path out."""
    x_tr, y_tr, ids_tr, paths = _as_xy(train_rows)
    x_va, y_va, ids_va, paths_va = _as_xy(val_rows)
    if list(paths_va) != list(paths):
        raise ValueError("feature schema mismatch between train and validation")
    out = _core(model_cls, parent_ckpt_path, x_tr, y_tr, x_va, y_va,
                ids_tr, ids_va, metric, budget_s, seed, log)
    if out["selected"] == "candidate":
        out_dir = tempfile.mkdtemp(prefix="b_cand_")
        ckpt_path = os.path.join(out_dir, "candidate.npz")
        out["model"].save(ckpt_path)          # xb mode: the file must not pre-exist
    else:
        ckpt_path = str(parent_ckpt_path)
    out["checkpoint_path"] = ckpt_path
    out["configuration"]["interface"] = "rows"
    return out
