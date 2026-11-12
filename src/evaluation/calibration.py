"""Calibración por escalado de temperatura.

Un modelo bien calibrado acierta el 80 % de las veces que dice tener 0,8 de confianza. M3 sale
sobreconfiado: eso desordena el umbral τ (0,7 no significa 70 % de acierto). El escalado por
temperatura divide los logits por un solo número T > 0 y no cambia la clase predicha, así que el F1
macro no se mueve; solo cambian las probabilidades.

T se ajusta SOLO con val. El informe debe decir que τ y T salen de la misma partición (optimista):
para medirlo se reporta también un ajuste cruzado (T de una mitad de val, ECE en la otra).

Las predicciones guardan probabilidades, no logits; softmax(log p / T) es idéntico a softmax(z / T).

Uso: python -m src.evaluation.calibration --model m3      # escribe artifacts/m3_temperature.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.config import load_config, path_of
from src.evaluation.preds import preds_frame, read_preds


def probs_and_labels(preds, labels):
    """(probabilidades n×k, índice de la clase real) a partir de un DataFrame en formato 7.2."""
    probs = preds[[f"prob_{lab}" for lab in labels]].to_numpy(dtype=float)
    y = preds["y_true"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()
    return probs, y


def apply_temperature(probs, temperature):
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(probs, y, bounds=(0.05, 20.0)):
    """T que minimiza la log-verosimilitud negativa."""
    res = minimize_scalar(lambda t: nll(apply_temperature(probs, t), y), bounds=bounds, method="bounded")
    return float(res.x)


def expected_calibration_error(probs, y, bins=10):
    """ECE sobre la clase predicha: promedio ponderado de |exactitud − confianza| por rango."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def crossfit_ece(probs, y, folds=2, seed=42):
    """ECE con T ajustada en otras particiones de val (estimación menos optimista)."""
    order = np.random.default_rng(seed).permutation(len(y))
    out = np.empty_like(probs)
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        out[test] = apply_temperature(probs[test], fit_temperature(probs[train], y[train]))
    return expected_calibration_error(out, y)


def calibrated_frame(preds, labels, temperature):
    """Predicciones (formato 7.2) con las probabilidades escaladas por `temperature`."""
    probs, _ = probs_and_labels(preds, labels)
    return preds_frame(preds["case_id"], preds["y_true"], apply_temperature(probs, temperature), labels)


def report(preds, labels, folds=2):
    probs, y = probs_and_labels(preds, labels)
    t = fit_temperature(probs, y)
    cal = apply_temperature(probs, t)
    return {"temperature": t, "n": int(len(y)),
            "ece_antes": expected_calibration_error(probs, y), "ece_despues": expected_calibration_error(cal, y),
            "ece_despues_ajuste_cruzado": crossfit_ece(probs, y, folds),
            "nll_antes": nll(probs, y), "nll_despues": nll(cal, y),
            "conf_media_antes": float(probs.max(1).mean()), "conf_media_despues": float(cal.max(1).mean()),
            "exactitud": float((probs.argmax(1) == y).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m3")
    ap.add_argument("--seed-id", default=None, help="id de la corrida (por defecto la principal, p. ej. m3)")
    args = ap.parse_args()
    cfg = load_config()
    labels = list(cfg["areas"])
    mid = args.seed_id or args.model
    results = path_of(cfg, "results")
    preds = read_preds(results / f"{mid}_val_preds.csv")
    rep = report(preds, labels)
    rep.update({"modelo": mid, "ajustado_con": "val"})
    art = path_of(cfg, "artifacts")
    art.mkdir(parents=True, exist_ok=True)
    (art / f"{args.model}_temperature.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    calibrated_frame(preds, labels, rep["temperature"]).to_csv(results / f"{mid}_cal_val_preds.csv", index=False,
                                                              encoding="utf-8")
    pd.DataFrame([rep]).to_csv(results / f"{mid}_calibracion_temperatura.csv", index=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
