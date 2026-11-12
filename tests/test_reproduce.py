import pandas as pd
import pytest

from src.config import load_config
from src.data.split import save_splits, split_dataframe
from src.data.synthetic import make_synthetic
from src.reproduce import reproduce


def _data(tmp_path):
    cfg = load_config()
    df = make_synthetic(cfg["areas"], n_per_area=40)
    (tmp_path / "data").mkdir()
    save_splits(split_dataframe(df, cfg["split"]["ratios"], cfg["seed"]), tmp_path / "data")
    return cfg, tmp_path / "data", tmp_path / "res"


def test_val_then_final_with_guards(tmp_path):
    cfg, data, res = _data(tmp_path)
    md = reproduce(cfg, "val", ["m0", "m1"], data_dir=data, out_dir=res, n_boot=100).read_text(encoding="utf-8")
    assert "| m0 |" in md and "| m1 |" in md and "m0 → m1" in md and "Valor mensual" in md
    assert "Calibración por temperatura" in md and (res / "m1_cal_val_preds.csv").exists()

    with pytest.raises(SystemExit):  # el test no se toca sin confirmar
        reproduce(cfg, "final", ["m0"], data_dir=data, out_dir=res)
    final = reproduce(cfg, "final", ["m0", "m1"], data_dir=data, out_dir=res, confirm_frozen=True, n_boot=100)
    assert "final (test)" in final.read_text(encoding="utf-8")
    cal_test, raw_test = pd.read_csv(res / "m1_cal_test_preds.csv"), pd.read_csv(res / "m1_test_preds.csv")
    assert (cal_test["y_pred"] == raw_test["y_pred"]).all()  # calibrar no cambia la clase predicha
    with pytest.raises(SystemExit):  # y se usa una sola vez
        reproduce(cfg, "final", ["m0"], data_dir=data, out_dir=res, confirm_frozen=True)
    reproduce(cfg, "final", ["m0"], data_dir=data, out_dir=res, confirm_frozen=True, again=True, n_boot=100)


def test_final_needs_val_predictions_to_choose_tau(tmp_path):
    cfg, data, res = _data(tmp_path)
    with pytest.raises(SystemExit):
        reproduce(cfg, "final", ["m0"], data_dir=data, out_dir=res, confirm_frozen=True, n_boot=50)


def test_rerun_skips_finished_models_and_force_repeats_them(tmp_path):
    cfg, data, res = _data(tmp_path)
    reproduce(cfg, "val", ["m1"], data_dir=data, out_dir=res, n_boot=50)
    info = res / "m1_val_run.json"
    first = info.stat().st_mtime_ns
    reproduce(cfg, "val", ["m1"], data_dir=data, out_dir=res, n_boot=50)
    assert info.stat().st_mtime_ns == first  # no se volvió a entrenar
    reproduce(cfg, "val", ["m1"], data_dir=data, out_dir=res, n_boot=50, force=True)
    assert info.stat().st_mtime_ns != first
