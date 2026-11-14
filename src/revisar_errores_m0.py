"""Revisión agregada de M0. Ejecutar pasando la raíz del repo; no lee textos ni test."""
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


def main(root):
    results = root / "experiments/results"
    out = results / "cp4_cami/errores_m0"
    out.mkdir(parents=True, exist_ok=False)
    data = {}
    for model in ("m0", "m3", "m4"):
        frame = pd.read_csv(results / f"{model}_val_preds.csv").set_index("case_id").sort_index()
        if frame.index.has_duplicates or frame[["y_true", "y_pred", "conf"]].isna().any().any():
            raise ValueError("Predicciones con duplicados o nulos")
        if not np.isfinite(frame.conf).all() or not frame.conf.between(0, 1).all():
            raise ValueError("Confianza inválida")
        data[model] = frame
    m0 = data["m0"]
    for frame in data.values():
        if not frame.y_true.equals(m0.y_true):
            raise ValueError("Las etiquetas o IDs de validación no coinciden")
    labels = sorted(set(m0.y_true) | set(m0.y_pred))
    confusion = pd.crosstab(m0.y_true, m0.y_pred).reindex(index=labels, columns=labels, fill_value=0)
    wrong = m0.y_true.ne(m0.y_pred)
    errors = m0[wrong]
    by_class = []
    for area in labels:
        support = int(m0.y_true.eq(area).sum())
        tp = int((m0.y_true.eq(area) & m0.y_pred.eq(area)).sum())
        predicted = int(m0.y_pred.eq(area).sum())
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        by_class.append(dict(area=area, n_reales=support, n_predichos=predicted,
                             correctos=tp, errores_salida=support-tp,
                             errores_entrada=predicted-tp, precision=precision, recall=recall,
                             f1=2*precision*recall/(precision+recall) if precision+recall else 0))
    by_class = pd.DataFrame(by_class)
    pairs = errors.groupby(["y_true", "y_pred"]).size().rename("casos").reset_index()
    pairs["fraccion_de_errores"] = pairs.casos / len(errors)
    pairs = pairs.sort_values(["casos", "y_true", "y_pred"], ascending=[False, True, True])
    thresholds = []
    for tau in (0, .7, .8, .9, .95):
        selected = m0[m0.conf >= tau]
        n = len(selected)
        incorrect = int(selected.y_true.ne(selected.y_pred).sum())
        thresholds.append(dict(umbral=tau, aceptados=n, cobertura=n/len(m0),
                               errores_aceptados=incorrect,
                               precision_aceptados=1-incorrect/n if n else float("nan"),
                               revision_humana=len(m0)-n))
    thresholds = pd.DataFrame(thresholds)
    comparison = []
    correct_m0 = ~wrong
    for model in ("m3", "m4"):
        correct = data[model].y_true.eq(data[model].y_pred)
        fixed = int((~correct_m0 & correct).sum())
        regressions = int((correct_m0 & ~correct).sum())
        assert fixed - regressions == int(correct.sum()) - int(correct_m0.sum())
        comparison.append(dict(modelo=model, errores_m0_corregidos=fixed,
                               aciertos_m0_que_se_vuelven_error=regressions,
                               mejora_neta_aciertos=fixed-regressions,
                               ambos_correctos=int((correct_m0 & correct).sum()),
                               ambos_incorrectos=int((~correct_m0 & ~correct).sum())))
    confusion.to_csv(out / "matriz_confusion.csv")
    by_class.to_csv(out / "metricas_por_area.csv", index=False)
    pairs.to_csv(out / "confusiones_ordenadas.csv", index=False)
    thresholds.to_csv(out / "cobertura_precision.csv", index=False)
    pd.DataFrame(comparison).to_csv(out / "cambios_frente_m0.csv", index=False)
    summary = dict(split="val", semilla=42, probabilidades="sin calibracion adicional por temperatura",
                   casos=len(m0), correctos=int((~wrong).sum()), errores=int(wrong.sum()),
                   exactitud=float((~wrong).mean()), f1_macro=float(by_class.f1.mean()),
                   ids_y_etiquetas_coincidentes=True,
                   umbrales="ilustrativos, no seleccionados como politica operativa ni optimo economico")
    (out / "resumen_verificado.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    assert confusion.to_numpy().sum()==len(m0)
    assert by_class.errores_salida.sum()==wrong.sum()==pairs.casos.sum()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(comparison).to_string(index=False))
    print(by_class.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv)>1 else Path.cwd())
