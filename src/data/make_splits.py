"""Arma train/val/test desde un CSV de textos etiquetados (hoy, SAIJ) con control de casi-duplicados.

Pasos: quita textos repetidos, valida el contrato, parte de forma estratificada por área y descarta
de val y test los textos casi idénticos (similitud coseno de TF-IDF >= umbral) a alguno de train, que
inflarían las métricas. Escribe data/processed/{train,val,test}.csv y splits_info.json (conteos).

Uso: python -m src.data.make_splits [--source data/external/saij_por_area.csv]
OJO: si la fuente no es del bufete, val y test son de otro dominio: declararlo en el informe.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import ROOT, load_config, path_of
from src.data.schema import validate
from src.data.split import save_splits, split_dataframe


def max_similarity_to(train_texts, texts, chunk=1000):
    """Para cada texto, la mayor similitud coseno (TF-IDF) con algún texto de train."""
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(train_texts)
    x_train = vec.transform(train_texts)
    x = vec.transform(texts)
    out = np.zeros(x.shape[0])
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = (x[i:i + chunk] @ x_train.T).max(axis=1).toarray().ravel()
    return out


def make_splits(source, out_dir=None, cfg=None, near_dup=0.9):
    cfg = cfg or load_config()
    out_dir = Path(out_dir or path_of(cfg, "processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source, encoding="utf-8").drop_duplicates("texto").reset_index(drop=True)
    errors = validate(df, cfg["areas"])
    if errors:
        raise ValueError(f"{source} no cumple el contrato: {errors}")

    df = split_dataframe(df, cfg["split"]["ratios"], cfg["seed"], cfg["split"]["temporal"])
    train = df[df["split"] == "train"]
    kept, dropped = [train], {}
    for name in ("val", "test"):
        part = df[df["split"] == name]
        sim = max_similarity_to(train["texto"], part["texto"])
        dropped[name] = int((sim >= near_dup).sum())
        kept.append(part[sim < near_dup])
    final = pd.concat(kept).reset_index(drop=True)
    save_splits(final, out_dir)

    info = {"fuente": str(source), "umbral_casi_duplicado": near_dup, "descartados_por_casi_duplicado": dropped,
            "conteos": {s: final[final["split"] == s]["area_legal"].value_counts().to_dict() for s in ("train", "val", "test")},
            "totales": final["split"].value_counts().to_dict()}
    (out_dir / "splits_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "data" / "external" / "saij_por_area.csv"))
    ap.add_argument("--near-dup", type=float, default=0.9)
    a = ap.parse_args()
    print(json.dumps(make_splits(a.source, near_dup=a.near_dup), ensure_ascii=False, indent=1))
