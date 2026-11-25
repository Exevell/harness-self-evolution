"""Stage-aware adoption-margin policy for the drift-stream harness (HAR-03, task-7).

Why
---
The platform replaces the public train/validation projection at every *stage*.
The previous submission used one fixed gate per dataset:

    A  macro_f1 : absolute margin 0.008 (0.020 when the val projection is small)
    B  rmse     : relative margin 0.05  (candidate must be >=5 % better)
    C  rmse     : relative margin 0.01  (>=1 % better) + NASA guard

At a stage switch the new data increment is small (B: +16..+108 rows, C:
+5 k rows), so a candidate trained on it almost never clears a 5 % B gate
(observed: the parent was kept in 9/10 B rounds and 9/10 C rounds).  The stage
therefore produces *no* gain and "worst stage" stays near zero.

Policy
------
Pure function of the *stage context*:

1. **first round of a new stage** (`stage_changed=True`) and the projection is
   big enough to trust (`n_val >= SMALL_FLOOR`; A reuses the lead's own 800-row
   threshold): relax the gate A 0.008 -> 0.004, B 0.05 -> 0.010, C 0.01 -> 0.003;
2. **drift detected** (the same lineage scores materially worse on the new
   projection than on the previous round's projection): halve the relaxed
   margin once more -- a real shift deserves a lower bar;
3. **small projection** (`n_val < SMALL_FLOOR`): do **not** relax.  On a few
   hundred validation rows a ~1 % "gain" is noise (measured: A's 155-row stage
   projections made 0.8-1.0 % public-val gains *worse* on the held-out 7/8);
   keep the fixed margin, relaxing only halfway if a drift is detected;
4. **later rounds inside the same stage**: restore the strict fixed margin --
   the stage already had its adaptation round, further adoption mostly chases
   validation noise.

Units are identical to the strategy modules, so integration is a one-line
margin override:

    A / macro_f1 : ABSOLUTE  (candidate_f1 - parent_f1 >= margin)
    B, C / rmse  : RELATIVE  (candidate_rmse <= parent_rmse * (1 - margin))

Pure, deterministic, stdlib only, no I/O, no global state.
"""

# --------------------------------------------------------------------------- #
# tuned constants (see work/stage/report.md for the measurements behind them)
# --------------------------------------------------------------------------- #
BASE_MARGIN = {"A": 0.008, "B": 0.05, "C": 0.01}     # current lead gate
SMALL_VAL_BASE = {"A": 0.020}                        # lead's rule for n_val < 800
FIRST_MARGIN = {"A": 0.004, "B": 0.010, "C": 0.003}  # new-stage first round
SMALL_FLOOR = {"A": 800, "B": 800, "C": 3000}        # below this: no relaxation
DRIFT_REL = {"A": 0.03, "B": 0.10, "C": 0.03}        # drop that counts as drift
DRIFT_RELAX = 0.5                                    # extra factor when drifting
SMALL_VAL_N = 800                                    # lead's A threshold

# Dataset resolution when the caller only knows the metric name.
_METRIC_TO_DATASET = {"macro_f1": "A"}


def resolve_dataset(dataset_key, metric, n_val):
    """Best-effort dataset key.  Callers that know the key should pass it."""
    if dataset_key in BASE_MARGIN:
        return str(dataset_key)
    key = str(metric or "").lower()
    if key in _METRIC_TO_DATASET:
        return _METRIC_TO_DATASET[key]
    if key in ("rmse", "mae", "mean_absolute_error"):
        # B and C share the metric name; C's stage projections are an order of
        # magnitude larger, so row count disambiguates a metric-only caller.
        return "C" if int(n_val or 0) >= 3000 else "B"
    raise ValueError("unknown metric/dataset: %r/%r" % (dataset_key, metric))


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def base_margin(dataset_key, n_val):
    """The fixed gate the previous submission used (one threshold for all rounds)."""
    key = resolve_dataset(dataset_key,
                          "macro_f1" if dataset_key == "A" else "rmse", n_val)
    if key == "A":
        return SMALL_VAL_BASE["A"] if int(n_val or 0) < SMALL_VAL_N else BASE_MARGIN["A"]
    return BASE_MARGIN[key]


def small_projection(dataset_key, n_val):
    """True when the stage's validation projection is too small to trust a relax."""
    key = resolve_dataset(dataset_key,
                          "macro_f1" if dataset_key == "A" else "rmse", n_val)
    return int(n_val or 0) < SMALL_FLOOR[key]


def drift_drop(prev_parent_metric, parent_metric, direction):
    """Relative score drop of the same lineage across two stage projections.

    Positive value == the parent got *worse* on the new projection (drift).
    """
    if prev_parent_metric is None or parent_metric is None:
        return 0.0
    try:
        prev = float(prev_parent_metric)
        cur = float(parent_metric)
    except (TypeError, ValueError):
        return 0.0
    scale = abs(prev)
    if scale < 1e-12:
        return 0.0
    if str(direction) == "lower":          # rmse: larger == worse
        return (cur - prev) / scale
    return (prev - cur) / scale            # macro_f1: smaller == worse


def margin_for(parent_metric, candidate_metric=None, metric="macro_f1",
               direction="higher", stage_changed=False, round_in_stage=1,
               n_val=0, dataset_key=None, prev_parent_metric=None):
    """Return the adoption margin (dataset-specific units) for this round."""
    key = resolve_dataset(dataset_key, metric, n_val)
    base = base_margin(key, n_val)
    if not stage_changed:
        return float(base)
    drift = drift_drop(prev_parent_metric, parent_metric, direction) >= DRIFT_REL[key]
    if small_projection(key, n_val):
        # Not enough rows to justify a looser gate; a detected drift is still a
        # strong signal, so allow a half-step.
        return float(base * (DRIFT_RELAX if drift else 1.0))
    margin = FIRST_MARGIN[key]
    if drift:
        margin *= DRIFT_RELAX
    return float(margin)


def adopt_by_margin(parent_metric, candidate_metric, direction, margin):
    """Margin-only comparison (same math as the three strategy modules)."""
    if direction == "higher":
        accept = float(candidate_metric) >= float(parent_metric) + float(margin)
        gain = float(candidate_metric) - float(parent_metric)
    else:
        accept = float(candidate_metric) <= float(parent_metric) * (1.0 - float(margin))
        gain = (float(parent_metric) - float(candidate_metric)) / max(abs(float(parent_metric)), 1e-12)
    return bool(accept), float(gain)


def should_adopt(parent_metric, candidate_metric, metric="macro_f1",
                 direction="higher", stage_changed=False, round_in_stage=1,
                 n_val=0, dataset_key=None, prev_parent_metric=None):
    """Decision + explanation for one round.

    Returns ``(accept: bool, info: dict)``.  ``info`` is JSON-serialisable and is
    meant to be stored verbatim in ``finish.configuration`` for auditability.
    """
    key = resolve_dataset(dataset_key, metric, n_val)
    margin = margin_for(parent_metric, candidate_metric, metric, direction,
                        stage_changed, round_in_stage, n_val, key,
                        prev_parent_metric)
    accept, gain = adopt_by_margin(parent_metric, candidate_metric, direction, margin)
    drop = drift_drop(prev_parent_metric, parent_metric, direction)
    info = {
        "dataset_key": key,
        "metric": metric,
        "direction": direction,
        "stage_changed": bool(stage_changed),
        "round_in_stage": int(round_in_stage),
        "n_val": int(n_val or 0),
        "parent_metric": _num(parent_metric),
        "candidate_metric": _num(candidate_metric),
        "gain": _num(gain),
        "margin": float(margin),
        "base_margin": float(base_margin(key, n_val)),
        "small_projection": bool(small_projection(key, n_val)),
        "drift_drop": _num(drop),
        "drift_detected": bool(stage_changed and drop >= DRIFT_REL[key]),
        "mode": "new_stage_first_round" if stage_changed else "within_stage_strict",
        "accept": bool(accept),
    }
    return bool(accept), info


def fixed_margin(dataset_key, metric=None, direction=None, n_val=0):
    """The previous submission's gate, reproduced exactly (baseline comparator)."""
    return float(base_margin(dataset_key, n_val))


def macro_f1_observed(y_true, y_pred):
    """Official A-metric convention (`baselines/train.py`): mean F1 over
    (truth u prediction) classes.  Used so that the ``prev_parent_metric``
    stored by the strategies and the ``parent_metric`` passed to
    :func:`margin_for` are on the same scale -- otherwise a stage projection
    that happens to miss one class would look like a 1/6 macro-F1 "drift"."""
    truth = list(y_true)
    pred = list(y_pred)
    observed = sorted(set(truth) | set(pred))
    scores = []
    for label in observed:
        tp = sum(1 for a, b in zip(truth, pred) if a == label and b == label)
        denom = sum(1 for b in pred if b == label) + sum(1 for a in truth if a == label)
        scores.append(2.0 * tp / denom if denom else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _num(value):
    try:
        import math
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


if __name__ == "__main__":  # tiny self-check: deterministic and stage-monotone
    cases = [
        ("A", 0.687, 0.705, "macro_f1", "higher", 155),
        ("A", 0.687, 0.705, "macro_f1", "higher", 1244),
        ("B", 400.0, 392.0, "rmse", "lower", 840),
        ("B", 400.0, 392.0, "rmse", "lower", 42),
        ("C", 40.0, 39.0, "rmse", "lower", 12254),
        ("C", 40.0, 39.0, "rmse", "lower", 1724),
    ]
    for key, parent, cand, metric, direction, n_val in cases:
        first, info = should_adopt(parent, cand, metric, direction, True, 1,
                                   n_val, dataset_key=key)
        later, info2 = should_adopt(parent, cand, metric, direction, False, 2,
                                    n_val, dataset_key=key)
        print("%s n_val=%-6d first margin=%.4f accept=%-5s | later margin=%.4f accept=%-5s"
              % (key, n_val, info["margin"], first, info2["margin"], later))
