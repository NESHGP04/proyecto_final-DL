"""Análisis provisional de CP4; lee solo predicciones de validación.

Uso: python -m src.value.analyze_validation
No entrena modelos, no abre test y no cambia parámetros ni resultados previos.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, load_value_params, param_values, path_of
from src.value.value_model import optimal_tau, sensitivity, value, value_curve


def analyze(results, out, params_metadata):
    results, out = Path(results), Path(out)
    # Una carpeta nueva evita sobreescribir una corrida anterior.
    out.mkdir(parents=True, exist_ok=False)
    params = param_values(params_metadata)
    summaries, curves, sensitivities, differences, checks = [], [], [], [], []
    reference = None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in ("m0", "m1", "m2", "m3", "m4"):
        preds = pd.read_csv(results / f"{model}_val_preds.csv")
        required = {"case_id", "y_true", "y_pred", "conf"}
        if not required <= set(preds) or preds.empty:
            raise ValueError(f"{model}: formato incompleto o vacío")
        if preds[list(required)].isna().any().any() or preds.case_id.duplicated().any():
            raise ValueError(f"{model}: nulos o identificadores duplicados")
        if not np.isfinite(preds.conf).all() or not preds.conf.between(0, 1).all():
            raise ValueError(f"{model}: confianza inválida")
        truth = preds.set_index("case_id").y_true.sort_index()
        if reference is None:
            reference = truth
        elif not truth.equals(reference):
            raise ValueError(f"{model}: casos o etiquetas diferentes")
        prob_cols = [c for c in preds if c.startswith("prob_")]
        probs = preds[prob_cols].to_numpy()
        if not prob_cols or not np.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
            raise ValueError(f"{model}: probabilidades inválidas")
        if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"{model}: probabilidades no suman uno")
        labels = np.array([c[5:] for c in prob_cols])
        if not np.array_equal(labels[probs.argmax(axis=1)], preds.y_pred.to_numpy()):
            raise ValueError(f"{model}: clase no coincide con argmax")
        if not np.allclose(probs.max(axis=1), preds.conf, atol=1e-5):
            raise ValueError(f"{model}: confianza no coincide con probabilidad máxima")
        curve = value_curve(preds, params)
        tau, best = optimal_tau(curve)
        chosen = curve.loc[curve.tau == tau].iloc[0]
        auto = preds[preds.conf >= tau]
        correct = int((auto.y_true == auto.y_pred).sum())
        saved = params["N"] * chosen.kappa * params["t_m"] / 60 * params["c_h"]
        errors = params["N"] * chosen.kappa * ((1 - chosen.p) - params["e_0"]) * params["c_err"]
        assert np.isclose(best, saved - errors - params["C_op"])
        summaries.append(dict(model=model, n_val=len(preds), tau=tau,
                              coverage=chosen.kappa, precision_auto=chosen.p,
                              n_auto_val=len(auto), correct_auto_val=correct,
                              monthly_auto_cases=params["N"] * chosen.kappa,
                              time_saving_value=saved, incremental_error_cost=errors,
                              operation_cost=params["C_op"], monthly_value=best))
        curves.append(curve.assign(model=model))
        sensitivities.append(sensitivity(preds, params, tau).assign(model=model))
        checks.append(dict(model=model, n=len(preds), checks="ok"))
        ax.plot(curve.tau, curve.value, label=model.upper(), linewidth=2)
        ax.scatter([tau], [best], s=25)
        # C_op=0 no cambia al multiplicarlo: explorar sus extremos YAML en forma aditiva.
        for cost in sorted({float(params_metadata["C_op"][k]) for k in ("min", "value", "max")}):
            changed = dict(params, C_op=cost)
            differences.append(dict(model=model, tau_fixed=tau, C_op=cost,
                                    monthly_value=value(preds, tau, changed)))
    summary = pd.DataFrame(summaries)
    summary["delta_value_vs_m0"] = summary.monthly_value - summary.iloc[0].monthly_value
    summary["delta_value_vs_previous"] = summary.monthly_value.diff()
    summary.to_csv(out / "resumen.csv", index=False)
    pd.concat(curves).to_csv(out / "curvas_valor.csv", index=False)
    sens = pd.concat(sensitivities)
    sens.to_csv(out / "sensibilidad.csv", index=False)
    pd.DataFrame(differences).to_csv(out / "escenarios_costo_operacion.csv", index=False)
    (out / "parametros_y_verificacion.json").write_text(json.dumps(
        dict(params=params_metadata, checks=checks, split="val", calibrated=False,
             selection="Malla 0.00 a 1.00, paso 0.01; primer máximo; semilla 42"),
        ensure_ascii=False, indent=2), encoding="utf-8")
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set(xlabel="Umbral de confianza τ", ylabel="Valor mensual (unidades monetarias)",
           title="Escenario provisional sobre SAIJ · parámetros supuestos")
    ax.legend(ncol=5)
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(out / "curvas_valor.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in summary.model:
        s = sens[(sens.model == model) & (sens.factor != 1)]
        b = float(summary.loc[summary.model == model, "monthly_value"].iloc[0])
        ranges = s.groupby("param").value.agg(["min", "max"])
        ax.plot(ranges.index, ranges["max"] - ranges["min"], marker="o", label=model.upper())
    ax.set(xlabel="Parámetro variado individualmente", ylabel="Amplitud del valor mensual (u. m.)",
           title="Sensibilidad: factores 0,5 a 1,5 · τ fijo por modelo")
    ax.text(.01, -.22, "C_op parte de cero: ver escenarios aditivos de 0 y 200 en el CSV.",
            transform=ax.transAxes, fontsize=9)
    ax.legend(ncol=5)
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(out / "sensibilidad.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=path_of(cfg, "results"))
    parser.add_argument("--out", type=Path, default=path_of(cfg, "results") / "cp4_cami")
    args = parser.parse_args()
    analyze(args.results, args.out, load_value_params())
