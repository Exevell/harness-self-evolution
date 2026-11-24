"""Core training/selection machinery for the drift-stream harness.

Pure NumPy.  Everything here mirrors ``baselines/model.py`` math so that a
checkpoint produced here is a *genuine* continuation of the parent model
(same architecture, same Adam state semantics, same normalisation).
"""
import math
import time
import numpy as np

MODEL_FORMAT = "contest.cpu-mlp.v1"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def macro_f1(labels, y, n_classes):
    scores = []
    for index in range(n_classes):
        hit = np.sum((labels == index) & (y == index))
        fp = np.sum((labels == index) & (y != index))
        fn = np.sum((labels != index) & (y == index))
        precision = hit / (hit + fp) if hit + fp else 0.0
        recall = hit / (hit + fn) if hit + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(scores)), scores


def nasa_score(pred, y):
    """NASA asymmetric score (lower better).  Under-estimation is punished harder."""
    error = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    penalty = np.where(error < 0, np.expm1(-error / 13.0), np.expm1(error / 10.0))
    return float(np.sum(penalty))


def evaluate(model, x, y, task, n_classes):
    prediction = model.predict(x)
    if task == "classification":
        labels = prediction.argmax(axis=1)
        f1, per_class = macro_f1(labels, y, n_classes)
        return {"macro_f1": f1, "accuracy": float(np.mean(labels == y)),
                "per_class_f1": [round(v, 6) for v in per_class]}
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - y
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "nasa_score": nasa_score(prediction, y),
            "pred_iqr": float(np.subtract(*np.percentile(prediction, [75, 25]))),
            "true_iqr": float(np.subtract(*np.percentile(y, [75, 25])))}


def score_value(metrics, metric):
    return float(metrics.get(metric, float("nan")))


# --------------------------------------------------------------------------- #
# extended fit: sample weights, gaussian augmentation, cosine decay, huber
# --------------------------------------------------------------------------- #
def fit_ext(model, x, y, epochs, *, lr=None, weights=None, noise=0.0,
            cosine=True, lr_floor=1e-6, huber=None, log=None):
    """Continue training *model* for ``epochs`` epochs.

    Identical gradient math to ``Model.fit`` but additionally supports
    per-sample weights, gaussian input augmentation (in normalised space),
    cosine learning-rate decay and a huber regression loss.
    """
    if type(epochs) is not int or not 1 <= epochs <= 10000:
        raise ValueError("epochs must be between 1 and 10000")
    xn = model.normalized(x)
    n = len(xn)
    if n == 0:
        raise ValueError("empty training matrix")
    if y.shape != (n,):
        raise ValueError("target shape mismatch")
    task = model.meta["task"]
    n_classes = len(model.meta["classes"]) if task == "classification" else 1
    if task == "regression":
        target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    else:
        if np.any(y < 0) or np.any(y >= n_classes):
            raise ValueError("unknown class")
        target = y

    if weights is None:
        weight = np.ones(n, dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64).copy()
        if weight.shape != (n,):
            raise ValueError("weight shape mismatch")
        weight = np.clip(weight, 1e-6, None)
        weight /= weight.mean()

    lr_start = float(model.meta["learning_rate"] if lr is None else lr)
    lr_start = max(1e-6, min(lr_start, 1.0))
    batch_size = int(model.meta["batch_size"])
    losses = []
    for epoch in range(epochs):
        if cosine and epochs > 1:
            ratio = epoch / float(epochs - 1)
            current_lr = lr_floor + 0.5 * (lr_start - lr_floor) * (1 + math.cos(math.pi * ratio))
        else:
            current_lr = lr_start
        model.meta["learning_rate"] = max(float(current_lr), lr_floor)

        rng = np.random.default_rng(
            np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
        order = rng.permutation(n)
        noise_field = None
        if noise > 0:
            noise_rng = np.random.default_rng(
                np.random.SeedSequence([model.meta["seed"], model.meta["epochs"], 7919]))
            noise_field = noise * noise_rng.standard_normal(xn.shape)
        loss_sum = 0.0
        for start in range(0, n, batch_size):
            indices = order[start:start + batch_size]
            batch = xn[indices]
            if noise_field is not None:
                batch = np.clip(batch + noise_field[indices], -10.0, 10.0)
            expected = target[indices]
            sample_weight = weight[indices]
            hidden, logits = model.forward(batch)
            if task == "classification":
                logits = logits - logits.max(axis=1, keepdims=True)
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                loss_sum -= float(np.log(np.maximum(
                    probabilities[np.arange(len(indices)), expected], 1e-300)).sum())
                delta = probabilities
                delta[np.arange(len(indices)), expected] -= 1
                delta *= (sample_weight / len(indices))[:, None]
            else:
                error = logits[:, 0] - expected
                if huber:
                    loss_sum += float(np.sum(sample_weight * np.where(
                        np.abs(error) <= huber,
                        0.5 * error ** 2,
                        huber * (np.abs(error) - 0.5 * huber))))
                    clipped = np.clip(error, -huber, huber)
                    delta = (sample_weight * clipped / len(indices))[:, None]
                else:
                    loss_sum += float(np.sum(sample_weight * error ** 2))
                    delta = (2 * sample_weight * error / len(indices))[:, None]
            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
            if not math.isfinite(norm):
                raise ValueError("nonfinite gradient")
            model.meta["steps"] += 1
            for name, gradient in gradients.items():
                gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                model.first[name] = .9 * model.first[name] + .1 * gradient
                model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
                first = model.first[name] / (1 - .9 ** model.meta["steps"])
                second = model.second[name] / (1 - .999 ** model.meta["steps"])
                model.params[name] -= model.meta["learning_rate"] * first / (
                    np.sqrt(second) + 1e-8)
            if not all(np.all(np.isfinite(p)) for p in model.params.values()):
                raise ValueError("nonfinite model state")
        model.meta["epochs"] += 1
        losses.append(loss_sum / n)
    model.validate()
    return losses


# --------------------------------------------------------------------------- #
# helpers used by the per-dataset strategies
# --------------------------------------------------------------------------- #
def inverse_frequency(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    return weights[y]


def fold_shrink(model, alpha, center):
    """Fold ``p = alpha*raw + (1-alpha)*center`` into target_scale/mean."""
    original_scale = float(model.meta["target_scale"])
    original_mean = float(model.meta["target_mean"])
    model.meta["target_scale"] = alpha * original_scale
    model.meta["target_mean"] = alpha * original_mean + (1.0 - alpha) * center
    model.validate()


def estimate_epoch_seconds(model, x, y, probe_epochs=1, **kwargs):
    started = time.time()
    fit_ext(model, x, y, probe_epochs, **kwargs)
    return (time.time() - started) / probe_epochs


def clone_from(model_cls, path):
    return model_cls.load(path)


def better(direction, candidate, reference, margin):
    """``margin`` is an absolute tolerance; for 'lower' it is a relative fraction."""
    if direction == "higher":
        return candidate > reference + margin
    return candidate < reference * (1.0 - margin)
