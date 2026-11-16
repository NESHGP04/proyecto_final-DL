"""Búsqueda pequeña M0, únicamente train/val. Ejecutar desde la raíz del repo.

No cambia modelos, configuración, splits o predicciones actuales. Requiere las
dependencias clásicas del proyecto, sin torch/transformers ni pesos de M3.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time


def buscar(root, out=None):
    root = Path(root).resolve()
    if not (root / "configs/config.yaml").is_file():
        raise ValueError("Ejecuta desde la raíz del repo o indica --repo.")
    sys.path.insert(0, str(root))
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from src.config import load_config
    from src.data.dataset import load_split
    from src.models.m0_tfidf_svm import build_model

    cfg = load_config(root / "configs/config.yaml")
    output = Path(out) if out else root / cfg["paths"]["results"] / "m0_busqueda_val"
    if not output.is_absolute():
        output = root / output
    if output.exists():
        raise ValueError("La carpeta de salida ya existe; se conserva. Usa --out con otra carpeta para una nueva corrida.")
    data_dir = root / cfg["paths"]["processed"]
    train = load_split(cfg, "train", data_dir=data_dir)
    val = load_split(cfg, "val", data_dir=data_dir)
    if train.empty or val.empty:
        raise ValueError("Train/val no pueden estar vacíos.")
    if train.case_id.isna().any() or val.case_id.isna().any() or set(train.case_id) & set(val.case_id):
        raise ValueError("Hay IDs nulos o compartidos entre train y val.")
    labels = list(cfg["areas"])
    if set(train.area_legal) != set(labels) or set(val.area_legal) != set(labels):
        raise ValueError("Se requieren las mismas clases en train/val para esta búsqueda.")
    if train.area_legal.value_counts().min() < 3:
        raise ValueError("Esta búsqueda requiere al menos tres ejemplos de train por clase para calibración interna.")
    candidates = [(1., (1, 2)), (.5, (1, 2)), (2., (1, 2)),
                  (.5, (1, 3)), (1., (1, 3)), (2., (1, 3))]
    metadata = dict(split="val", seed=int(cfg["seed"]), n_train=len(train), n_val=len(val),
                    source_model="src.models.m0_tfidf_svm.build_model", labels=labels,
                    python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, pandas=pd.__version__,
                    changes=["C", "ngram_range"],
                    selection="F1 macro máximo en validación; primer máximo en empates. Baseline primero.",
                    adoption="Ninguna: no se modificó la configuración ni se sustituyeron predicciones.",
                    caveat="Búsqueda exploratoria en val; no reportar su mejor resultado como evaluación independiente.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configuracion_busqueda.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    for index, (c, ngrams) in enumerate(candidates, 1):
        print(f"[{index}/6] C={c:g}, n-gramas={ngrams}", flush=True)
        model = build_model(train, cfg["seed"])
        tfidf_name, _ = model.steps[0]
        classifier_name, _ = model.steps[-1]
        model.set_params(**{f"{tfidf_name}__ngram_range": ngrams,
                            f"{classifier_name}__estimator__C": c})
        started = time.perf_counter()
        model.fit(train.texto, train.area_legal)
        elapsed = time.perf_counter() - started
        started = time.perf_counter()
        predicted = model.predict(val.texto)
        inference_ms = 1000 * (time.perf_counter() - started) / len(val)
        result = dict(orden=index, C=c, ngram_min=ngrams[0], ngram_max=ngrams[1],
                      es_base=index==1, f1_macro=f1_score(val.area_legal, predicted, labels=labels, average="macro", zero_division=0),
                      exactitud=accuracy_score(val.area_legal, predicted),
                      train_seconds=elapsed, inference_ms_per_text=inference_ms,
                      n_features=len(model.steps[0][1].vocabulary_))
        result.update({f"recall_{label}":float(score) for label,score in zip(labels,recall_score(
            val.area_legal,predicted,labels=labels,average=None,zero_division=0))})
        results.append(result)
        pd.DataFrame(results).to_csv(output / "resultados_busqueda.csv", index=False)
        print(f"F1 macro={result['f1_macro']:.4f} · entrenamiento={elapsed:.1f} s", flush=True)
        del model
        gc.collect()
    best = max(results, key=lambda row: row["f1_macro"])
    summary = dict(estado="completa", configuraciones=len(results), baseline=results[0],
                   mejor_exploratorio=best, delta_f1_vs_base=best["f1_macro"]-results[0]["f1_macro"],
                   conclusion="Revisar antes de decidir adopción. No cambió ningún modelo activo ni se abrió test.")
    (output / "resumen_busqueda.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Búsqueda terminada. Comparte resultados_busqueda.csv y resumen_busqueda.json de:", output, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    buscar(args.repo, args.out)
