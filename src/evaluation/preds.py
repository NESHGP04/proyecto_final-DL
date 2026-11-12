"""Formato común de predicciones (docs/CONTEXTO.md 7.2).

experiments/results/<id_modelo>_<split>_preds.csv con: case_id, y_true, y_pred, conf, prob_<area>...
"""
import numpy as np
import pandas as pd


def preds_frame(case_ids, y_true, probs, labels):
    """DataFrame en el formato 7.2. probs: array (n, len(labels)) en el orden de `labels`."""
    probs = np.asarray(probs, dtype=float)
    pred_idx = probs.argmax(axis=1)
    df = pd.DataFrame({
        "case_id": list(case_ids),
        "y_true": list(y_true),
        "y_pred": [labels[i] for i in pred_idx],
        "conf": probs.max(axis=1),
    })
    for j, label in enumerate(labels):
        df[f"prob_{label}"] = probs[:, j]
    return df


def write_preds(path, case_ids, y_true, probs, labels):
    df = preds_frame(case_ids, y_true, probs, labels)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def read_preds(path):
    return pd.read_csv(path, encoding="utf-8")
