"""CP4: escenarios GTQ con evidencia reportada y predicciones calibradas VAL.

python -m src.value.analyze_bufete --results experiments/results --out experiments/results/cp4_bufete
No entrena, no lee test ni cambia calibración/umbral de la API.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.config import ROOT, load_config
from src.value.analyze_workflow_scenarios import read_predictions
from src.value.value_model import workflow_curve


def analyze(results, out, config_file):
    config = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    labels = load_config()["areas"]
    data, reference = {}, None
    for model in ("m0", "m1", "m2", "m3", "m4"):
        data[model] = read_predictions(Path(results)/f"{model}_cal_val_preds.csv", labels, reference)
        reference = data[model].y_true
    out = Path(out)
    if out.exists():
        raise ValueError("La carpeta de salida ya existe. Usa otra para conservar resultados previos.")
    out.mkdir(parents=True)
    curves, selections, ceilings = [], [], []
    penalty_minutes = config["error_deteccion_minutos"] + config["error_correccion_minutos"]
    for scenario, key in (("menor_volumen_valor_hora", "min"), ("central", "referencia"),
                          ("mayor_volumen_valor_hora", "max")):
        n, hourly = config["documentos_mes"][key], config["c_h"][key]
        ceiling = n*(config["t_m"]-config["t_a"])/60*hourly
        ceilings.append(dict(scenario=scenario, documentos_mes=n, c_h=hourly,
                             valor_tiempo_gtq=ceiling, mantenimiento_gtq=config["mantenimiento_gtq_mes"],
                             saldo_antes_errores_otros_costos_gtq=ceiling-config["mantenimiento_gtq_mes"]))
        for overhead in config["sobrecosto_revision_minutos"]:
            params = dict(N=n, t_m=config["t_m"], t_a=config["t_a"], t_r=overhead, c_h=hourly,
                          e_0=config["e_0_escenario"], c_err=penalty_minutes/60*hourly,
                          C_op=config["mantenimiento_gtq_mes"])
            for model, preds in data.items():
                curve = workflow_curve(preds, params)
                curve["time_value_gtq"] = curve.net_minutes/60*hourly
                curve["error_penalty_gtq"] = curve.net_error_exposure*params["c_err"]
                curve["maintenance_gtq"] = params["C_op"]
                curve["documentos_mes"] = n
                curve["c_h"] = hourly
                curve["scenario"] = scenario
                curve["review_overhead_minutes"] = overhead
                curve["model"] = model
                curves.append(curve)
                fixed = curve.loc[(curve.tau-config["umbral_referencia"]).abs().idxmin()]
                best = curve.loc[curve.value.idxmax()]
                for policy, row in (("referencia_0.70", fixed), ("maximo_analitico_val", best)):
                    selections.append(dict(policy=policy, **row.to_dict()))
    pd.concat(curves, ignore_index=True).to_csv(out/"curvas_valor_gtq.csv", index=False)
    selected = pd.DataFrame(selections)
    selected.to_csv(out/"resumen_gtq.csv", index=False)
    pd.DataFrame(ceilings).to_csv(out/"referencia_sin_errores_cobertura_total.csv", index=False)
    limitations = [
        "Todos los resultados son escenarios exploratorios, no ahorro comprobado ni flujo de efectivo.",
        "N=50/60/70 documentos: diez casos por cinco/seis/siete documentos; puntos centrales derivados.",
        "e_0 es desconocido. Se fija cero solo para no acreditar mejoras frente al error manual.",
        "Precisión y cobertura provienen de SAIJ-val calibrado; su transferencia al bufete no está medida.",
        "No se verificó aquí cómo se ajustó la temperatura. La selección de umbral usa esta misma val.",
        "t_a procede de M3 con dos usuarios familiarizados. Aplicarlo a M0/M1/M2/M4 es hipótesis comparativa.",
        "Los tiempos no incluyen la nueva carga PDF/OCR; no se conoce su sobrecosto.",
        "Se penalizan 15 minutos por sugerencia incorrecta aceptada; no es tasa de errores finales tras revisión.",
        "Se supone que detectar/corregir lo hace el abogado y que esos 15 minutos no están incluidos en t_a.",
        "Mantenimiento cotizado Q400/mes por cuatro revisiones; 20 horas mensuales, no Q100 por hora.",
        "Sin pagos de cómputo reportados; electricidad, equipo y esfuerzo de reentrenamiento no valorados.",
        "Saldo después de mantenimiento y penalización modelada; antes de otros costos no medidos.",
        "Máximos analíticos no actualizan modelo, API, temperatura ni umbral activo.",
    ]
    (out/"metodo_y_fuentes.json").write_text(json.dumps(dict(split="val", n_val=len(reference),
        calibrated_files=True, config=config, limitations=limitations), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(selected[(selected.scenario=="central") & (selected.review_overhead_minutes==0)][
        ["model", "policy", "tau", "kappa", "p", "value"]].to_string(index=False))
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"experiments/results")
    parser.add_argument("--out", type=Path, default=ROOT/"experiments/results/cp4_bufete")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/value_bufete_confirmado.yaml")
    args = parser.parse_args()
    analyze(args.results, args.out, args.config)
