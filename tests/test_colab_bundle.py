import zipfile

import pandas as pd
import pytest

from src.colab_bundle import TOP, build_bundle
from src.config import load_config


def _cfg(tmp_path):
    cfg = load_config()
    cfg["paths"]["processed"] = str(tmp_path / "processed")
    cfg["paths"]["results"] = str(tmp_path / "results")
    cfg["paths"]["artifacts"] = str(tmp_path / "artifacts")
    (tmp_path / "processed").mkdir()
    (tmp_path / "results").mkdir()
    return cfg


def _write_splits(tmp_path, origen="saij"):
    for name in ("train", "val", "test"):
        pd.DataFrame({"case_id": ["a"], "texto": ["x"], "area_legal": ["laboral"], "origen": [origen]}).to_csv(
            tmp_path / "processed" / f"{name}.csv", index=False)
    for f in ("m1_val_preds.csv", "m1_val_run.json", "m3_val_preds.csv", "m2_test_preds.csv"):
        (tmp_path / "results" / f).write_text("x", encoding="utf-8")


def test_bundle_has_code_train_val_and_old_results_but_never_test(tmp_path):
    cfg = _cfg(tmp_path)
    _write_splits(tmp_path)
    out, _ = build_bundle(tmp_path / "b.zip", cfg)
    names = set(zipfile.ZipFile(out).namelist())
    assert f"{TOP}/src/models/m3_bert_finetune.py" in names and f"{TOP}/configs/config.yaml" in names
    assert f"{TOP}/data/processed/train.csv" in names and f"{TOP}/data/processed/val.csv" in names
    assert f"{TOP}/experiments/results/m1_val_preds.csv" in names and f"{TOP}/experiments/results/m1_val_run.json" in names
    assert not any("test.csv" in n or "test_preds" in n or "/raw/" in n or "/interim/" in n for n in names)
    assert f"{TOP}/experiments/results/m3_val_preds.csv" not in names  # solo M0 a M2: M3 se calcula en Colab
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def test_bundle_refuses_non_public_data(tmp_path):
    cfg = _cfg(tmp_path)
    _write_splits(tmp_path, origen="bufete")
    with pytest.raises(SystemExit):
        build_bundle(tmp_path / "b.zip", cfg)
