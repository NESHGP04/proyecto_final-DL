"""Análisis de errores a partir de predicciones (formato 7.2) y metadatos de los casos.

No usa texto de los casos: solo clases, confianzas, canal y fuente del texto. Las salidas son
agregados y se pueden versionar. Sirve para ver qué clases se confunden, cómo se comportan los
casos de baja confianza y si el modelo funciona peor en algún canal o con texto de OCR.
"""
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


def confusion_matrix_df(preds, labels):
    """Filas = clase real, columnas = clase predicha."""
    cm = pd.crosstab(preds["y_true"], preds["y_pred"]).reindex(index=labels, columns=labels, fill_value=0)
    cm.index.name, cm.columns.name = "real", "predicha"
    return cm


def top_confusions(cm, k=5):
    """Pares (real, predicha) con más errores, de mayor a menor."""
    rows = [(r, p, int(cm.loc[r, p])) for r in cm.index for p in cm.columns if r != p and cm.loc[r, p] > 0]
    return sorted(rows, key=lambda x: -x[2])[:k]


def group_metrics(preds, meta, labels, by, min_n=1):
    """F1 macro, exactitud y recall por grupo de `by` (p. ej. canal o fuente_texto).

    meta: DataFrame con case_id y la columna `by`. Los grupos con menos de min_n casos se marcan
    `pocos_casos`: con tan pocos casos las diferencias entre grupos no son interpretables.
    """
    df = preds.merge(meta[["case_id", by]], on="case_id", how="left")
    rows = []
    for group, g in df.groupby(by):
        m = compute_metrics(g["y_true"], g["y_pred"], labels)
        rows.append({by: group, "n": len(g), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "conf_media": float(g["conf"].mean()), "pocos_casos": len(g) < min_n})
    return pd.DataFrame(rows)


def low_confidence_report(preds, threshold):
    """Cuántos casos caen bajo el umbral, y qué tan acertados serían (los que irían a revisión humana)."""
    low = preds[preds["conf"] < threshold]
    high = preds[preds["conf"] >= threshold]
    acc = lambda d: float((d["y_pred"] == d["y_true"]).mean()) if len(d) else float("nan")  # noqa: E731
    return {"threshold": threshold, "n_low": len(low), "share_low": len(low) / len(preds),
            "accuracy_low": acc(low), "accuracy_high": acc(high), "n_high": len(high)}


def calibration_table(preds, bins=5):
    """Confianza media vs. exactitud por rango de confianza (un modelo calibrado las iguala)."""
    edges = np.linspace(0, 1, bins + 1)
    b = pd.cut(preds["conf"], edges, include_lowest=True)
    rows = []
    for interval, g in preds.groupby(b, observed=True):
        rows.append({"rango": str(interval), "n": len(g), "conf_media": float(g["conf"].mean()),
                     "exactitud": float((g["y_pred"] == g["y_true"]).mean())})
    return pd.DataFrame(rows)


def plot_confusion(cm, path, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set(xlabel="predicha", ylabel="real", title=title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center",
                    color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_report(preds, meta, labels, threshold=0.7, out_dir=None, model_id="model"):
    """Informe completo; si out_dir, guarda CSV y la matriz de confusión."""
    cm = confusion_matrix_df(preds, labels)
    report = {
        "confusion": cm,
        "top_confusions": top_confusions(cm),
        "por_canal": group_metrics(preds, meta, labels, "canal") if "canal" in meta else None,
        "por_fuente": group_metrics(preds, meta, labels, "fuente_texto") if "fuente_texto" in meta else None,
        "baja_confianza": low_confidence_report(preds, threshold),
        "calibracion": calibration_table(preds),
    }
    if out_dir is not None:
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cm.to_csv(out / f"{model_id}_confusion.csv", encoding="utf-8")
        plot_confusion(cm, out / f"{model_id}_confusion.png", model_id)
        for key in ("por_canal", "por_fuente"):
            if report[key] is not None:
                report[key].to_csv(out / f"{model_id}_{key}.csv", index=False, encoding="utf-8")
        report["calibracion"].to_csv(out / f"{model_id}_calibracion.csv", index=False, encoding="utf-8")
    return report
