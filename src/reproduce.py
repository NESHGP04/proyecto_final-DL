"""Reproduce las métricas del informe: entrena y evalúa los modelos y escribe una tabla resumen.

Uso:
  python -m src.reproduce --stage val [--models m0 m1 m2 m3] [--seeds 42 43 44]
  python -m src.reproduce --stage final --confirm-frozen [--models m0 m1 m2 m3]

`val` entrena cada modelo con train y evalúa en val. `final` evalúa en test UNA sola vez, con la
configuración ya congelada: el umbral τ se elige con las predicciones de val de cada modelo (nunca
con las de test) y se aplica a test. Si ya existen predicciones de test, `final` se niega a correr
salvo `--again`, que hay que declarar en el informe.

Salida: experiments/results/summary_<stage>.md (F1 macro con IC, exactitud, recall por clase, valor
mensual, deltas entre versiones y costos). El modelo de valor usa configs/value_params.yaml.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.config import load_config, load_value_params, param_values, path_of
from src.evaluation.compare import compare_models, cost_row
from src.evaluation.metrics import bootstrap_ci, compute_metrics
from src.evaluation.preds import read_preds
from src.value.value_model import optimal_tau, value, value_curve

ALL_MODELS = ["m0", "m1", "m2", "m3", "m4"]
PAIRS = [("m2", "m3"), ("m3", "m4"), ("m1", "m3"), ("m0", "m1")]  # (anterior, siguiente)


def _runner(name):
    # import tardío: m0 no necesita torch
    if name == "m0":
        from src.models.m0_tfidf_svm import run
    elif name == "m1":
        from src.models.m1_cnn_scratch import run
    elif name == "m2":
        from src.models.m2_bert_frozen import run
    elif name == "m3":
        from src.models.m3_bert_finetune import run
    elif name == "m4":
        from src.models.m4_bert_dapt import run
    else:
        raise ValueError(name)
    return run


def _id(model, seed, cfg):
    return model if model == "m0" or seed == cfg["seed"] else f"{model}_s{seed}"


def _train(cfg, model, split, seed, final, data_dir, out_dir):
    run = _runner(model)
    kw = {"data_dir": data_dir, "out_dir": out_dir}
    if model != "m0":
        kw["seed"] = seed
    return run(cfg, split, final=final, **kw)


def _fmt_ci(f1, ci):
    return f"{f1:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def reproduce(cfg, stage="val", models=None, seeds=None, data_dir=None, out_dir=None,
              confirm_frozen=False, again=False, n_boot=1000):
    models = models or [m for m in ALL_MODELS if m != "m4"]
    seeds = list(seeds) if seeds else [cfg["seed"]]
    out = Path(out_dir or path_of(cfg, "results"))
    out.mkdir(parents=True, exist_ok=True)
    final = stage == "final"
    split = "test" if final else "val"
    labels = list(cfg["areas"])
    params = param_values(load_value_params())

    if final:
        if not confirm_frozen:
            raise SystemExit("final usa el test: confirmar que la configuración está congelada con --confirm-frozen")
        if list(out.glob("*_test_preds.csv")) and not again:
            raise SystemExit("ya hay predicciones de test: el test se usa una sola vez (--again lo permite; declararlo en el informe)")

    primary = {}
    spread = {}
    for m in models:
        f1s = []
        for seed in (seeds if m != "m0" else seeds[:1]):
            path = _train(cfg, m, split, seed, final, data_dir, out)
            p = read_preds(path)
            f1s.append(compute_metrics(p["y_true"], p["y_pred"], labels)["macro_f1"])
            if seed == seeds[0]:
                primary[m] = (p, _id(m, seed, cfg))
        spread[m] = f1s

    rows, recalls, values, costs = [], {}, {}, {}
    for m, (p, mid) in primary.items():
        met = compute_metrics(p["y_true"], p["y_pred"], labels)
        ci = bootstrap_ci(p["y_true"], p["y_pred"], labels, n_boot=n_boot, seed=cfg["seed"])
        f1s = spread[m]
        rows.append((m, _fmt_ci(met["macro_f1"], ci), f"{met['accuracy']:.3f}",
                     f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f} (n={len(f1s)})" if len(f1s) > 1 else "una semilla"))
        recalls[m] = {lab: met["per_class"][lab]["recall"] for lab in labels}
        if final:  # τ de val aplicado a test
            val_path = out / f"{mid}_val_preds.csv"
            if not val_path.exists():
                raise SystemExit(f"falta {val_path.name}: correr primero --stage val para elegir τ")
            tau, _ = optimal_tau(value_curve(read_preds(val_path), params))
        else:
            tau, _ = optimal_tau(value_curve(p, params))
        values[m] = (tau, value(p, tau, params))
        run_json = out / f"{mid}_{split}_run.json"
        costs[m] = cost_row(run_json) if run_json.exists() else {}

    lines = [f"# Resumen {'final (test)' if final else 'de validación'}", "",
             f"Modelos: {', '.join(models)}. Semillas: {seeds}. Casos evaluados: {len(next(iter(primary.values()))[0])}.", "",
             "| Modelo | F1 macro [IC 95 %] | Exactitud | F1 macro entre semillas |", "|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows]
    lines += ["", "Recall por clase", "", "| Modelo | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    lines += [f"| {m} | " + " | ".join(f"{recalls[m][lab]:.2f}" for lab in labels) + " |" for m in recalls]
    lines += ["", f"Valor mensual con los parámetros de value_params.yaml (τ elegido con {'val' if final else 'esta misma partición'}).",
              "", "| Modelo | τ | Valor mensual |", "|---|---|---|"]
    lines += [f"| {m} | {values[m][0]:.2f} | {values[m][1]:,.1f} |" for m in values]
    lines += ["", "Deltas de F1 macro (siguiente − anterior), IC pareado 95 %", "",
              "| Comparación | Delta | IC | ¿Cruza 0? |", "|---|---|---|---|"]
    for a, b in PAIRS:
        if a in primary and b in primary:
            r = compare_models(primary[a][0], primary[b][0], labels, a, b, n_boot=n_boot, seed=cfg["seed"])["delta_macro_f1"]
            lines.append(f"| {a} → {b} | {r['delta']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                         f"{'no' if r['significant'] else 'sí'} |")
    keys = ["trainable_params", "train_seconds", "head_train_seconds", "embed_seconds", "inference_ms_per_text", "peak_memory_mb"]
    lines += ["", "Costos", "", "| Modelo | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    lines += [f"| {m} | " + " | ".join(str(costs[m].get(k, "")) for k in keys) + " |" for m in costs]
    lines += ["", "Nota: el valor usa un solo C_op para todos los modelos; ajustarlo por versión en value_params.yaml si se quiere comparar costo de cómputo."]
    path = out / f"summary_{stage}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "final"], required=True)
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--confirm-frozen", action="store_true")
    ap.add_argument("--again", action="store_true")
    a = ap.parse_args()
    print(reproduce(load_config(), a.stage, a.models, a.seeds, confirm_frozen=a.confirm_frozen, again=a.again))
