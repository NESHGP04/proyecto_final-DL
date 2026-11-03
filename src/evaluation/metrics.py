"""Métricas técnicas con intervalos de confianza por bootstrap. Dueña: Cami (CP3); la usa Nes."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)
        },
    }


def _macro_f1(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def bootstrap_ci(y_true, y_pred, labels, n_boot=1000, seed=42, alpha=0.05):
    """IC percentil del F1 macro remuestreando casos. Devuelve (bajo, alto)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(_macro_f1(y_true[idx], y_pred[idx], labels))
    return float(np.percentile(scores, 100 * alpha / 2)), float(np.percentile(scores, 100 * (1 - alpha / 2)))


def paired_bootstrap_delta(y_true, pred_a, pred_b, labels, n_boot=1000, seed=42, alpha=0.05):
    """Delta de F1 macro (b - a) sobre los mismos casos, con IC. Sirve para M2 -> M3 y M3 -> M4.

    Devuelve (delta, bajo, alto). Si el IC no cruza 0, el cambio es distinguible del azar de muestreo.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(_macro_f1(y_true[idx], pred_b[idx], labels) - _macro_f1(y_true[idx], pred_a[idx], labels))
    delta = _macro_f1(y_true, pred_b, labels) - _macro_f1(y_true, pred_a, labels)
    return (float(delta), float(np.percentile(deltas, 100 * alpha / 2)),
            float(np.percentile(deltas, 100 * (1 - alpha / 2))))
