"""Escenarios CP3/CP4 en horas equivalentes, exclusivamente con predicciones val.

python -m src.value.analyze_workflow_scenarios
No estima retorno real, no entrena, no abre test y no cambia políticas activas.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.config import ROOT, load_config, path_of
from src.value.value_model import workflow_components, workflow_curve

MODELS = ("m0", "m1", "m2", "m3", "m4")
COLORS = dict(zip(MODELS, ("#28658b", "#d17b24", "#398568", "#8058a6", "#596875")))


def read_predictions(path, labels, reference=None):
    df = pd.read_csv(path)
    required = {"case_id", "y_true", "y_pred", "conf"}
    if df.empty or not required <= set(df) or df[list(required)].isna().any().any() or df.case_id.duplicated().any():
        raise ValueError("Predicciones incompletas o IDs repetidos")
    if not df.y_true.isin(labels).all() or not df.y_pred.isin(labels).all():
        raise ValueError("Clases incompatibles")
    probs = df[[f"prob_{label}" for label in labels]].to_numpy()
    if not np.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
        raise ValueError("Probabilidades inválidas")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilidades no normalizadas")
    if not np.isfinite(df.conf).all() or not np.allclose(df.conf, probs.max(axis=1), atol=1e-5):
        raise ValueError("Confianza incompatible")
    if not np.array_equal(np.array(labels)[probs.argmax(axis=1)], df.y_pred.to_numpy()):
        raise ValueError("Clase incompatible con probabilidades")
    df = df.set_index("case_id").sort_index()
    if reference is not None and not df.y_true.equals(reference):
        raise ValueError("IDs o etiquetas distintos entre predicciones")
    return df


def base_params(config, scenario):
    return dict(N=float(config["N"]), t_m=float(config["t_m"]),
                t_a=float(scenario["t_a"]), t_r=float(scenario["t_r"]),
                e_0=float(config["e_0"]), c_h=1., c_err=0., C_op=0.)


def budgets(curve, penalty_minutes):
    # c_h=1 y c_err=penalty_minutes/60: horas equivalentes antes de operación.
    result = curve.copy()
    result["operation_budget"] = result.net_minutes/60 - result.net_error_exposure*penalty_minutes/60
    result["value"] = result.operation_budget
    return result


def best_row(curve):
    return curve.loc[curve.operation_budget.idxmax()].to_dict()


def analyze(results, out, scenario_file):
    results, out = Path(results), Path(out)
    config = yaml.safe_load(Path(scenario_file).read_text(encoding="utf-8"))
    cfg = load_config()
    if config["c_h"] != 1 or config["C_op"] != 0 or config["e_0"] != 0:
        raise ValueError("Este informe exige normalización c_h=1, C_op=0 y escenario conservador e_0=0")
    data, reference = {}, None
    for model in MODELS:
        for variant, name in (("original", f"{model}_val_preds.csv"),
                              ("temperatura", f"{model}_cal_val_preds.csv")):
            frame = read_predictions(results/name, cfg["areas"], reference)
            if reference is None:
                reference = frame.y_true
            data[model, variant] = frame
        if not data[model,"original"].y_pred.equals(data[model,"temperatura"].y_pred):
            raise ValueError("La variante por temperatura cambió clases; revisar procedencia")
    if out.exists():
        raise ValueError("La carpeta de salida ya existe; se conserva. Indicar --out con otra carpeta.")
    out.mkdir(parents=True, exist_ok=False)
    curves, selected, sensitivity, seed_rows = [], [], [], []
    cache = {}
    for scenario_name, scenario in config["escenarios"].items():
        params = base_params(config, scenario)
        for (model, variant), preds in data.items():
            curve = workflow_curve(preds, params)
            cache[scenario_name,model,variant] = curve
            curves.append(curve.assign(scenario=scenario_name, model=model, variant=variant))
            for penalty in config["error_minutes_grid"]:
                best = best_row(budgets(curve, penalty))
                selected.append(dict(scenario=scenario_name, model=model, variant=variant,
                                     error_penalty_minutes=penalty, **best))
            ref_penalty = config["error_minutes_reference"]
            tau = best_row(budgets(curve, ref_penalty))["tau"]
            reference_params = dict(params, c_err=ref_penalty/60)
            # Un factor a la vez; el umbral analítico se mantiene fijo.
            for name in ("N", "t_m", "t_a", "t_r", "c_h", "c_err"):
                for factor in config["sensitivity_factors"]:
                    changed = dict(reference_params)
                    changed[name] *= factor
                    row = workflow_components(preds, tau, changed)
                    sensitivity.append(dict(scenario=scenario_name, model=model, variant=variant,
                                            parameter=name, factor=factor, reference_tau=tau,
                                            value_equivalent_hours=row["value"]))
            if variant == "original":
                seed_rows.extend(dict(scenario=scenario_name, model=model, seed=42,
                                      error_penalty_minutes=p, **best_row(budgets(curve,p)))
                                 for p in config["error_minutes_grid"])
        for model in MODELS[1:]:
            for seed in (43,44):
                preds = read_predictions(results/f"{model}_s{seed}_val_preds.csv", cfg["areas"], reference)
                curve = workflow_curve(preds, params)
                seed_rows.extend(dict(scenario=scenario_name, model=model, seed=seed,
                                      error_penalty_minutes=p, **best_row(budgets(curve,p)))
                                 for p in config["error_minutes_grid"])
    curves = pd.concat(curves, ignore_index=True)
    selected = pd.DataFrame(selected)
    sensitivity = pd.DataFrame(sensitivity)
    seeds = pd.DataFrame(seed_rows)
    curves.to_csv(out/"curvas_componentes.csv", index=False)
    selected.to_csv(out/"presupuestos_equilibrio.csv", index=False)
    sensitivity.to_csv(out/"sensibilidad_tau_fijo.csv", index=False)
    seeds.to_csv(out/"variacion_semillas.csv", index=False)
    deltas = []
    for (scenario, variant, penalty), group in selected.groupby(["scenario","variant","error_penalty_minutes"]):
        for previous, current in (("m2","m3"),("m3","m4"),("m0","m3"),("m0","m4")):
            before = float(group.loc[group.model==previous,"operation_budget"].iloc[0])
            after = float(group.loc[group.model==current,"operation_budget"].iloc[0])
            deltas.append(dict(scenario=scenario,variant=variant,error_penalty_minutes=penalty,
                               comparison=f"{previous}->{current}",delta_budget_equivalent_hours=after-before,
                               delta_budget_equivalent_minutes=60*(after-before)))
    pd.DataFrame(deltas).to_csv(out/"deltas_presupuesto.csv", index=False)

    # Costos: coeficientes de tiempo activo; no suman servidor en espera ni mantenimiento.
    costs=[]
    for model in MODELS:
        if model=="m0":
            source=results/"m0_busqueda_val_corrida1/resumen_busqueda.json"
            run=json.loads(source.read_text())["baseline"]
            device="equipo local de Cami; hardware no registrado"
        else:
            source=results/f"{model}_val_run.json"
            run=json.loads(source.read_text())
            device=run.get("device","no registrado")
        ms=float(run["inference_ms_per_text"])
        costs.append(dict(model=model,source=source.relative_to(results).as_posix(),device=device,
                          inference_ms_per_text=ms,active_inference_hours_per_scenario_month=config["N"]*ms/3600000,
                          train_seconds=run.get("train_seconds"),total_train_seconds=run.get("total_train_seconds"),
                          head_train_seconds=run.get("head_train_seconds"),embed_seconds=run.get("embed_seconds")))
    pd.DataFrame(costs).to_csv(out/"coeficientes_costo_computo.csv",index=False)
    metadata=dict(prediction_split="val",n_validation=len(reference),main_seed=42,
                  variants=["original","temperatura"],seed_check=[42,43,44],
                  units="Horas equivalentes del costo/hora base; no GTQ ni retorno real",
                  assumptions=config,
                  policy="Umbrales analíticos por escenario elegidos en la misma validación, no políticas adoptadas.",
                  limitations=["Confianza y error de SAIJ; transferencia real no medida.",
                               "Temperatura ajustada sobre val; resultados in-sample.",
                               "Tiempo asistido procede de demo con dos participantes.",
                               "N=10 supone una unidad elegible por caso judicial.",
                               "Penalizar sugerencias erróneas no mide errores finales tras confirmación humana.",
                               "Precios de cómputo, costo por hora, costo de error y mantenimiento sin medir."])
    (out/"metodo_y_supuestos.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")

    fig, axes=plt.subplots(1,3,figsize=(16,4.8),sharey=True)
    for ax,(scenario, settings) in zip(axes,config["escenarios"].items()):
        for model in MODELS:
            curve=cache[scenario,model,"original"]
            ax.plot(curve.tau,curve.net_minutes,label=model.upper(),color=COLORS[model],linewidth=2)
        ax.axhline(0,color="#555",linewidth=.7)
        ax.set(title=scenario.replace("_"," "),xlabel="Umbral de confianza τ")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Minutos netos por mes hipotético de 10 entradas")
    axes[-1].legend()
    fig.suptitle("Capacidad de tiempo condicional · referencia demo, no medición de M0–M4")
    fig.tight_layout();fig.savefig(out/"capacidad_tiempo.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,variant in zip(axes,("original","temperatura")):
        part=selected[(selected.scenario=="referencia_demo") & (selected.variant==variant)]
        for model in MODELS:
            s=part[part.model==model].sort_values("error_penalty_minutes")
            ax.plot(s.error_penalty_minutes,s.operation_budget*60,label=model.upper(),color=COLORS[model],marker="o")
        ax.set(title=variant,xlabel="Costo hipotético por sugerencia errónea\n(minutos equivalentes de trabajo)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Máximo para operación mensual\n(minutos equivalentes; convertir con costo/hora real)")
    axes[-1].legend()
    fig.suptitle("Punto de equilibrio exploratorio · referencia de tiempos demo")
    fig.tight_layout();fig.savefig(out/"punto_equilibrio.png",dpi=160);plt.close(fig)

    subset=sensitivity[(sensitivity.scenario=="demo_revision_extra") & (sensitivity.model=="m3") & (sensitivity.variant=="original")]
    ranges=subset.groupby("parameter").value_equivalent_hours.agg(["min","max"]).sort_values("max")*60
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(ranges.index,ranges["max"]-ranges["min"],left=ranges["min"],color="#39728d")
    ax.axvline(0,color="#555",linewidth=.7)
    ax.set(xlabel="Presupuesto equivalente para operación (minutos/mes)",
           title="Sensibilidad M3 · ±20 % y ±50 % · τ analítico fijo\nDemo con revisión extra; error valorado en 15 minutos hipotéticos")
    ax.grid(axis="x",alpha=.15)
    fig.tight_layout();fig.savefig(out/"sensibilidad_m3.png",dpi=160);plt.close(fig)
    print("Escenarios calculados sin entrenar ni abrir test. Unidades: horas equivalentes, no quetzales.")
    print(selected[(selected.scenario=="referencia_demo") & (selected.variant=="original")
                   & (selected.error_penalty_minutes==config["error_minutes_reference"])][
                       ["model","tau","kappa","p","net_minutes","net_error_exposure","operation_budget"]].to_string(index=False))
    return selected


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=path_of(load_config(),"results"))
    parser.add_argument("--out",type=Path,default=path_of(load_config(),"results")/"cp3_cp4_escenarios")
    parser.add_argument("--scenarios",type=Path,default=ROOT/"configs/value_scenarios.yaml")
    args=parser.parse_args()
    analyze(args.results,args.out,args.scenarios)
