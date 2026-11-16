"""Mide el costo de M0 y M1 en esta máquina sin tocar sus predicciones.

M0 y M1 se reentrenan en una carpeta temporal, cada uno en su propio proceso (para que el pico de
memoria sea del modelo y no del otro), y se completan sus `<id>_val_run.json` con dispositivo, pico de
memoria y arquitectura. Si ya había `train_seconds` o `inference_ms_per_text` se conservan y la medida
nueva queda en `medicion_repetida`. También revisa que las predicciones nuevas sean iguales a las
oficiales, para saber si el reentrenamiento es determinista.

Uso: python -m src.evaluation.measure_cost --models m0 m1
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.config import load_config, path_of

_CODE = {"m0": "from src.models.m0_tfidf_svm import run; run(load_config(), 'val', out_dir={out!r})",
         "m1": "from src.models.m1_cnn_scratch import run; run(load_config(), 'val', out_dir={out!r}, plot=False)"}
_KEYS = ("device", "peak_memory_mb", "machine", "system")


def measure(model, results):
    with tempfile.TemporaryDirectory() as tmp:
        code = "from src.config import load_config\n" + _CODE[model].format(out=tmp)
        subprocess.run([sys.executable, "-c", code], check=True)
        new = json.loads((Path(tmp) / f"{model}_val_run.json").read_text(encoding="utf-8"))
        same = pd.read_csv(Path(tmp) / f"{model}_val_preds.csv").equals(pd.read_csv(results / f"{model}_val_preds.csv"))
    path = results / f"{model}_val_run.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {k: v for k, v in new.items()}
    if "train_seconds" in info and path.exists():
        info["medicion_repetida"] = {k: new[k] for k in ("train_seconds", "inference_ms_per_text")}
    info.update({k: new[k] for k in _KEYS})
    info["predicciones_identicas_al_repetir"] = bool(same)
    path.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return {k: info.get(k) for k in ("train_seconds", "inference_ms_per_text", "peak_memory_mb", "device",
                                     "machine", "predicciones_identicas_al_repetir", "medicion_repetida")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["m0", "m1"], choices=list(_CODE))
    args = ap.parse_args()
    results = path_of(load_config(), "results")
    for m in args.models:
        print(m, json.dumps(measure(m, results), ensure_ascii=False))
