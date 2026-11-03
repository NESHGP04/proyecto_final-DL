"""M0: TF-IDF + SVM lineal (referencia sin deep learning). Dueña: Cami (CP3).

Uso: python -m src.models.m0_tfidf_svm --split val
El split `test` solo con --final (evaluación final con configuración congelada).
"""
import argparse

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.config import load_config, path_of
from src.data.dataset import load_split
from src.evaluation.preds import write_preds

MODEL_ID = "m0"


def build_model(train_df, seed):
    """TF-IDF (1-2 gramas) + LinearSVC; se calibra para tener probabilidades/confianza."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
    min_count = int(train_df["area_legal"].value_counts().min())
    cv = min(3, min_count)
    clf = CalibratedClassifierCV(svm, cv=cv) if cv >= 2 else svm
    return make_pipeline(tfidf, clf)


def _probs(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)
    scores = model.decision_function(texts)  # sin calibrar: softmax como confianza aproximada
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def run(cfg, eval_split="val", final=False, data_dir=None, out_dir=None):
    train = load_split(cfg, "train", data_dir=data_dir)
    ev = load_split(cfg, eval_split, final=final, data_dir=data_dir)
    model = build_model(train, cfg["seed"])
    model.fit(train["texto"], train["area_legal"])
    labels = list(model.classes_)
    probs = _probs(model, ev["texto"])
    out = f"{out_dir or path_of(cfg, 'results')}/{MODEL_ID}_{eval_split}_preds.csv"
    write_preds(out, ev["case_id"], ev["area_legal"], probs, labels)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    print(run(load_config(), args.split, final=args.final))
