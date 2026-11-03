"""Prueba de punta a punta con datos SINTÉTICOS: solo verifica que las piezas se conectan."""
from src.config import load_config, load_value_params, param_values
from src.data.dataset import load_split
from src.data.schema import validate
from src.data.split import save_splits, split_dataframe
from src.data.synthetic import make_synthetic
from src.evaluation.metrics import compute_metrics
from src.evaluation.preds import read_preds
from src.models import m0_tfidf_svm
from src.value.value_model import optimal_tau, value_curve


def test_synthetic_pipeline(tmp_path):
    cfg = load_config()
    df = make_synthetic(cfg["areas"], n_per_area=40)
    assert validate(df, cfg["areas"]) == []
    save_splits(split_dataframe(df, cfg["split"]["ratios"], cfg["seed"]), tmp_path)

    out = m0_tfidf_svm.run(cfg, "val", data_dir=tmp_path, out_dir=tmp_path)
    preds = read_preds(out)
    assert {"case_id", "y_true", "y_pred", "conf"} <= set(preds.columns)
    assert compute_metrics(preds["y_true"], preds["y_pred"], cfg["areas"])["accuracy"] > 0.3

    params = param_values(load_value_params())
    tau, _ = optimal_tau(value_curve(preds, params))
    assert 0.0 <= tau <= 1.0


def test_test_split_is_closed(tmp_path):
    cfg = load_config()
    try:
        load_split(cfg, "test", data_dir=tmp_path)
    except PermissionError:
        return
    raise AssertionError("test debería estar cerrado sin final=True")
