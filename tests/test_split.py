import pandas as pd

from src.data.split import split_dataframe
from src.data.synthetic import make_synthetic

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
AREAS = ["laboral", "civil", "mercantil"]


def test_split_is_stratified_and_complete():
    df = make_synthetic(AREAS, n_per_area=40)
    out = split_dataframe(df, RATIOS, seed=1)
    assert set(out["split"]) == {"train", "val", "test"}
    assert len(out) == len(df)
    for split in ("train", "val", "test"):
        assert set(out[out["split"] == split]["area_legal"]) == set(AREAS)


def test_split_is_reproducible():
    df = make_synthetic(AREAS, n_per_area=40)
    a = split_dataframe(df, RATIOS, seed=7)["split"].tolist()
    b = split_dataframe(df, RATIOS, seed=7)["split"].tolist()
    assert a == b


def test_temporal_test_is_most_recent():
    df = make_synthetic(["laboral"], n_per_area=40)
    out = split_dataframe(df, RATIOS, seed=1, temporal=True)
    fecha = pd.to_datetime(out["fecha"])
    assert fecha[out["split"] == "test"].min() >= fecha[out["split"] == "train"].max()


def test_tiny_class_does_not_crash():
    df = make_synthetic(["laboral"], n_per_area=2)
    out = split_dataframe(df, RATIOS, seed=1)
    assert len(out) == 2


def test_paths_can_be_overridden_with_environment_variables(monkeypatch, tmp_path):
    from src.config import load_config, path_of
    monkeypatch.setenv("DL_PATH_RESULTS", str(tmp_path / "drive" / "results"))
    cfg = load_config()
    assert path_of(cfg, "results") == tmp_path / "drive" / "results"
    assert str(path_of(cfg, "processed")).endswith("data/processed")  # lo demás no cambia
