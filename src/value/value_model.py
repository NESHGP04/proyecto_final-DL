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


def workflow_components(preds, tau, params):
    """Extensión explícita para organización con confirmación humana.

    t_m: tiempo manual; t_a: tiempo asistido de los aceptados;
    t_r: sobrecosto frente al flujo manual en los enviados a revisión.
    No cambia value(): con t_a=t_r=0 se recupera su fórmula.
    c_err penaliza sugerencias erróneas según preds, no errores jurídicos.
    """
    if preds.empty:
        raise ValueError("Se necesitan predicciones no vacías")
    names = ("N", "t_m", "t_a", "t_r", "c_h", "e_0", "c_err", "C_op")
    pms = {name: float(params[name]) for name in names}
    if not all(np.isfinite(v) for v in pms.values()):
        raise ValueError("Los parámetros deben ser finitos")
    if any(pms[name] < 0 for name in names) or pms["e_0"] > 1:
        raise ValueError("Parámetros negativos o tasa fuera de [0,1]")
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("Umbral fuera de [0,1]")
    kappa, precision = coverage_precision(preds, tau)
    minutes = pms["N"] * (kappa * (pms["t_m"] - pms["t_a"])
                           - (1-kappa) * pms["t_r"])
    exposure = pms["N"] * kappa * ((1-precision) - pms["e_0"]) if kappa else 0.0
    budget = minutes / 60 * pms["c_h"] - exposure * pms["c_err"]
    return dict(tau=float(tau), kappa=kappa, p=precision,
                net_minutes=minutes, net_error_exposure=exposure,
                operation_budget=budget, value=budget-pms["C_op"])


def workflow_curve(preds, params, taus=None):
    """Curva de escenarios: las salidas usan las unidades de c_h/c_err suministradas."""
    taus = np.linspace(0, 1, 101) if taus is None else taus
    return pd.DataFrame([workflow_components(preds, tau, params) for tau in taus])
