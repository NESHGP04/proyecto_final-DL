import pandas as pd

from src.config import load_config
from src.data.dataset import load_split
from src.data.make_splits import make_splits


def _source(tmp_path, areas):
    rows = []
    for a in areas:
        for i in range(30):
            rows.append({"case_id": f"{a}{i}", "texto": f"texto del área {a} con detalle número {i} sobre su materia particular {a * 2} {i * 7}",
                         "area_legal": a, "canal": "documento", "fecha": "", "fuente_texto": "nativo"})
    # un caso casi idéntico a otro: debe caer fuera de val/test si quedó en ellos
    rows.append({**rows[0], "case_id": "dup", "texto": rows[0]["texto"] + " extra"})
    path = tmp_path / "src.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_make_splits_writes_valid_stratified_files_without_near_duplicates(tmp_path):
    cfg = load_config()
    info = make_splits(_source(tmp_path, cfg["areas"]), tmp_path / "out", cfg, near_dup=0.9)
    for name in ("train", "val", "test"):
        df = load_split(cfg, name, final=True, data_dir=tmp_path / "out")
        assert set(df["area_legal"]) == set(cfg["areas"])
    assert sum(info["totales"].values()) <= 6 * 30 + 1
    train = load_split(cfg, "train", data_dir=tmp_path / "out")
    val = load_split(cfg, "val", data_dir=tmp_path / "out")
    assert not set(train["texto"]) & set(val["texto"])
    assert (tmp_path / "out" / "splits_info.json").exists()
