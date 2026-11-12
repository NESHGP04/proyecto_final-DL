"""Corpus de texto legal sin etiquetas para la adaptación de dominio de M4 (preentrenamiento MLM).

Sale de los 473.251 registros de SAIJ ya filtrados (data/external/saij_filtrado.csv.gz, Apache-2.0). Se
descarta todo lo que pueda dar ventaja a M4 sobre val/test:

- textos idénticos a alguno de train, val o test (train ya entra aparte con include_train_texts);
- textos casi idénticos (similitud coseno TF-IDF >= 0,9) a algún texto de val o test.

Para esa exclusión se LEEN los textos de test.csv, solo para quitarlos del corpus: no se calcula ninguna
métrica con ellos y nada de test entra al corpus ni al entrenamiento.

Es texto del mismo dominio que train/val/test (jurisprudencia argentina): la adaptación acerca el encoder a ese
dominio, no a derecho guatemalteco. Hay que decirlo en el informe.

Escribe data/external/corpus_mlm.txt (un texto por párrafo, separados por línea en blanco; lo lee
corpus_texts de M4) y data/external/corpus_mlm_info.json.

Uso: python -m src.data.dapt_corpus --n 50000
"""
import argparse
import json

import pandas as pd

from src.config import load_config, path_of
from src.data.make_splits import max_similarity_to

MIN_WORDS = 25


def _norm(text):
    return " ".join(str(text).split())


def build_corpus(source, exclude_texts, near_dup_texts, n=50000, seed=42, near_dup=0.9, min_words=MIN_WORDS):
    """Devuelve (DataFrame con id, materia, texto del corpus, dict con los conteos)."""
    df = pd.read_csv(source, encoding="utf-8")
    info = {"registros_fuente": int(len(df))}
    df["texto"] = df["texto"].map(_norm)
    df = df[df["texto"].str.split().str.len() >= min_words]
    info["tras_longitud_minima"] = int(len(df))
    df = df.drop_duplicates("texto")
    info["tras_quitar_duplicados"] = int(len(df))
    banned = {_norm(t) for t in exclude_texts}
    df = df[~df["texto"].isin(banned)]
    info["tras_quitar_identicos_a_train_val_test"] = int(len(df))

    # muestra con margen y luego filtro de casi duplicados contra val y test
    take = min(len(df), int(n * 1.3))
    cand = df.sample(take, random_state=seed).reset_index(drop=True)
    sim = max_similarity_to([_norm(t) for t in near_dup_texts], cand["texto"].tolist())
    cand = cand[sim < near_dup]
    info["candidatos"] = int(take)
    info["descartados_casi_duplicados_de_val_test"] = int(take - len(cand))
    out = cand.head(n).reset_index(drop=True)
    info["textos_en_corpus"] = int(len(out))
    return out, info


def write_corpus(df, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(df["texto"].tolist()) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    args = ap.parse_args()
    cfg = load_config()
    ext, proc = path_of(cfg, "external"), path_of(cfg, "processed")
    texts = {name: pd.read_csv(proc / f"{name}.csv", usecols=["texto"], encoding="utf-8")["texto"].tolist()
             for name in ("train", "val", "test")}
    corpus, info = build_corpus(ext / "saij_filtrado.csv.gz", sum(texts.values(), []), texts["val"] + texts["test"],
                                n=args.n, seed=cfg["seed"])
    write_corpus(corpus, ext / "corpus_mlm.txt")
    if "materia" in corpus:  # solo las 15 materias más frecuentes (hay cientos de variantes)
        info["materias_mas_frecuentes"] = {str(k): int(v) for k, v in corpus["materia"].value_counts().head(15).items()}
    info["palabras_aprox"] = int(corpus["texto"].str.split().str.len().sum())
    (ext / "corpus_mlm_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=1, ensure_ascii=False))
