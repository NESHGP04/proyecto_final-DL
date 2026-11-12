"""Comparación entre dos versiones de modelo: delta de F1 macro con IC, recall por clase y costos.

Sirve para documentar M2 -> M3 (y M3 -> M4): es lo que pide el catedrático para fine-tuning y
transfer learning. Las dos predicciones deben ser del mismo split y los mismos casos.
"""
import json
from pathlib import Path

from src.evaluation.metrics import bootstrap_ci, compute_metrics, paired_bootstrap_delta
from src.evaluation.preds import read_preds


def compare_models(preds_a, preds_b, labels, name_a="A", name_b="B", n_boot=1000, seed=42):
    """preds_a/preds_b: DataFrames en el formato 7.2. Devuelve un dict con los números de la tabla."""
    a = preds_a.set_index("case_id").sort_index()
    b = preds_b.set_index("case_id").sort_index()
    if list(a.index) != list(b.index) or not (a["y_true"] == b["y_true"]).all():
        raise ValueError("las predicciones no son sobre los mismos casos")
    y = a["y_true"].to_numpy()
    ma, mb = compute_metrics(y, a["y_pred"], labels), compute_metrics(y, b["y_pred"], labels)
    delta, low, high = paired_bootstrap_delta(y, a["y_pred"].to_numpy(), b["y_pred"].to_numpy(),
                                              labels, n_boot=n_boot, seed=seed)
    return {
        "a": name_a, "b": name_b, "n": len(y),
        "macro_f1": {name_a: ma["macro_f1"], name_b: mb["macro_f1"],
                     "ci_" + name_a: bootstrap_ci(y, a["y_pred"].to_numpy(), labels, n_boot, seed),
                     "ci_" + name_b: bootstrap_ci(y, b["y_pred"].to_numpy(), labels, n_boot, seed)},
        "delta_macro_f1": {"delta": delta, "ci_low": low, "ci_high": high, "significant": low > 0 or high < 0},
        "recall_por_clase": {lab: {name_a: ma["per_class"][lab]["recall"], name_b: mb["per_class"][lab]["recall"],
                                   "delta": mb["per_class"][lab]["recall"] - ma["per_class"][lab]["recall"],
                                   "support": ma["per_class"][lab]["support"]} for lab in labels},
    }


def cost_row(run_json_path):
    """Costos de un modelo a partir de su <id>_<split>_run.json."""
    info = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
    keys = ("trainable_params", "total_params", "train_seconds", "head_train_seconds", "embed_seconds",
            "inference_ms_per_text", "peak_memory_mb", "embeddings_from_cache", "device", "best_epoch",
            "total_train_seconds")
    return {k: info[k] for k in keys if k in info}


def to_markdown(result, cost_a=None, cost_b=None):
    a, b = result["a"], result["b"]
    f = result["macro_f1"]
    d = result["delta_macro_f1"]
    lines = [
        f"Comparación {a} → {b} sobre {result['n']} casos",
        "",
        "| Métrica | " + a + " | " + b + " |",
        "|---|---|---|",
        f"| F1 macro | {f[a]:.3f} [{f['ci_' + a][0]:.3f}, {f['ci_' + a][1]:.3f}] | "
        f"{f[b]:.3f} [{f['ci_' + b][0]:.3f}, {f['ci_' + b][1]:.3f}] |",
        "",
        f"Delta F1 macro ({b} − {a}): {d['delta']:+.3f}, IC 95 % [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] "
        + ("(el IC no cruza 0)" if d["significant"] else "(el IC cruza 0: no distinguible del azar de muestreo)"),
        "",
        f"| Clase | n | Recall {a} | Recall {b} | Delta |",
        "|---|---|---|---|---|",
    ]
    for lab, r in result["recall_por_clase"].items():
        lines.append(f"| {lab} | {r['support']} | {r[a]:.3f} | {r[b]:.3f} | {r['delta']:+.3f} |")
    if cost_a and cost_b:
        lines += ["", f"| Costo | {a} | {b} |", "|---|---|---|"]
        for k in cost_a.keys() | cost_b.keys():
            lines.append(f"| {k} | {cost_a.get(k, '')} | {cost_b.get(k, '')} |")
    return "\n".join(lines)


def compare_files(path_a, path_b, labels, name_a, name_b, **kw):
    """Atajo: lee los CSV de predicciones y los JSON de costos hermanos (mismo prefijo)."""
    res = compare_models(read_preds(path_a), read_preds(path_b), labels, name_a, name_b, **kw)
    runs = [Path(str(p).replace("_preds.csv", "_run.json")) for p in (path_a, path_b)]
    costs = [cost_row(r) if r.exists() else None for r in runs]
    return res, to_markdown(res, *costs)
