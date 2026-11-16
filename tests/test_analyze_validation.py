import json

import numpy as np
import pandas as pd

from src.config import load_value_params
from src.evaluation.preds import write_preds
from src.value.analyze_validation import analyze

LABELS = ["laboral", "civil"]


def _write(results, name, seed):
    rng = np.random.default_rng(seed)
    y = np.random.default_rng(0).choice(LABELS, 60)  # mismas etiquetas en todos los modelos
    q = rng.uniform(0.5, 0.99, 60)
    hit = rng.random(60) < 0.93                      # acierta ~93 %: el valor depende de las probabilidades
    probs = []
    for t, qi, h in zip(y, q, hit):
        guess = 0 if (t == "laboral") == h else 1
        probs.append([qi, 1 - qi] if guess == 0 else [1 - qi, qi])
    write_preds(results / name, [f"c{i}" for i in range(60)], y, probs, LABELS)


def test_analyze_reads_calibrated_files_only_when_asked(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    for k, m in enumerate(("m0", "m1", "m2", "m3", "m4")):
        _write(res, f"{m}_val_preds.csv", k)
        _write(res, f"{m}_cal_val_preds.csv", k + 100)  # distintas a propósito
    params = load_value_params()
    analyze(res, tmp_path / "raw", params)
    analyze(res, tmp_path / "cal", params, calibrated=True)
    raw = pd.read_csv(tmp_path / "raw" / "resumen.csv")
    cal = pd.read_csv(tmp_path / "cal" / "resumen.csv")
    assert not raw["monthly_value"].equals(cal["monthly_value"])
    assert json.loads((tmp_path / "raw" / "parametros_y_verificacion.json").read_text())["calibrated"] is False
    assert json.loads((tmp_path / "cal" / "parametros_y_verificacion.json").read_text())["calibrated"] is True
