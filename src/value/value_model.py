"""Modelo de valor (docs/CONTEXTO.md sección 5).

Valor(τ) = N·κ(τ)·(t_m/60)·c_h − N·κ(τ)·[(1 − p(τ)) − e_0]·c_err − C_op

κ(τ): fracción de casos con confianza ≥ τ (se asignan automáticamente; el resto va a revisión humana).
p(τ): precisión de esos casos asignados automáticamente.
Los parámetros (N, t_m, c_h, e_0, c_err, C_op) vienen de configs/value_params.yaml.
"""
import numpy as np
import pandas as pd


def coverage_precision(preds, tau):
    """κ(τ) y p(τ) a partir de un DataFrame de predicciones (formato 7.2)."""
    auto = preds[preds["conf"] >= tau]
    kappa = len(auto) / len(preds)
    p = float((auto["y_pred"] == auto["y_true"]).mean()) if len(auto) else float("nan")
    return kappa, p


def value(preds, tau, params):
    """Valor mensual con umbral tau. `params`: dict plano {N, t_m, c_h, e_0, c_err, C_op}."""
    kappa, p = coverage_precision(preds, tau)
    if kappa == 0:
        return -params["C_op"]
    saved = params["N"] * kappa * (params["t_m"] / 60.0) * params["c_h"]
    error_delta = params["N"] * kappa * ((1.0 - p) - params["e_0"]) * params["c_err"]
    return saved - error_delta - params["C_op"]


def value_curve(preds, params, taus=None):
    """Tabla con tau, kappa, p y valor para una malla de umbrales."""
    taus = np.linspace(0.0, 1.0, 101) if taus is None else taus
    rows = []
    for tau in taus:
        kappa, p = coverage_precision(preds, tau)
        rows.append({"tau": float(tau), "kappa": kappa, "p": p, "value": value(preds, tau, params)})
    return pd.DataFrame(rows)


def optimal_tau(curve):
    """Umbral que maximiza el valor (primer máximo si hay empates)."""
    best = curve.loc[curve["value"].idxmax()]
    return float(best["tau"]), float(best["value"])


def sensitivity(preds, params, tau, factors=(0.5, 0.8, 1.0, 1.2, 1.5), names=None):
    """Valor al multiplicar cada parámetro por cada factor (uno a la vez), con tau fijo.

    Con factors 0.5/0.8/1.2/1.5 cubre errores de ±20 % y ±50 % en los supuestos.
    """
    names = names or list(params)
    rows = []
    for name in names:
        for f in factors:
            changed = dict(params)
            changed[name] = params[name] * f
            rows.append({"param": name, "factor": f, "value": value(preds, tau, changed)})
    return pd.DataFrame(rows)
