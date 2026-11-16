"""Laboratorio 10 (CC3092): Explainable AI sobre el modelo final (clasificador de texto con BETO).

Método principal: Integrated Gradients (IG) sobre los embeddings de entrada, con referencia = todos los
tokens [PAD] (se conservan [CLS] y [SEP]). Reutiliza `integrated_gradients` de src/explain/explain.py.
NO reentrena nada: carga los pesos guardados con `--save` (artifacts/<modelo>.pt).

Qué genera (todo en experiments/results/xai_lab10/, con semilla fija):
  instancias.csv            las instancias locales elegidas (acierto, error, cercano a τ, subgrupo, error típico)
  local_<rol>_<case_id>.png barras de atribución por palabra (clase predicha y, en errores, clase real)
  local_top_words.csv       palabras con más atribución por instancia
  global_top_words.csv      palabras con mayor atribución media por clase (muestra del conjunto)
  global_por_clase.png      figura de la explicación global
  deletion.csv / .png       prueba de fidelidad: borrar las k palabras más atribuidas vs k al azar
  m0_vs_ig.csv              baseline interpretable (TF-IDF+SVM): términos con más peso vs IG (Jaccard)
  atajos.csv                prueba de atajos: quitar ids/uuid/códigos del texto y ver cuánto cambia la predicción
  run_xai.json              parámetros de diseño, semillas, versiones, tiempos (costo de explicar)

Uso:
  python -m src.explain.xai_lab10 --model m3 --split val
  python -m src.explain.xai_lab10 --model m3 --split test --confirm-final   # solo si ya hicieron la evaluación final

Los textos son casos del bufete o de SAIJ: no publicar fragmentos de texto real de datos confidenciales
en el PDF (usar ejemplos inventados o muy editados).
"""
import argparse
import json
import platform
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config, path_of
from src.data.dataset import load_split
from src.explain.explain import SPECIAL, aggregate_by_class, integrated_gradients, merge_wordpieces, plot_attributions

# Patrones de "ruido" que un modelo podría usar como atajo (ids, uuid, códigos tipo CT001, números largos).
# Se vieron entre las palabras más atribuidas de M3 en experiments/results/explain/summary.json.
DEFAULT_SUSPECT = r"(?i)\buuid\b|\bCT\d{3,}\b|\b\d{4,}\b"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_probs(pred, texts, bs=32):
    """Probabilidades calibradas (con la temperatura T del predictor) para una lista de textos."""
    out = []
    for i in range(0, len(texts), bs):
        out.append(pred.predict_proba(texts[i:i + bs]).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(pred.areas)))


# ---------------------------------------------------------------------------------------------------
# Selección de instancias locales
# ---------------------------------------------------------------------------------------------------
def select_instances(df, tau):
    """df con y_true, y_pred, conf, canal, fuente_texto. Devuelve [(rol, índice)] sin repetir casos."""
    used, chosen = set(), []

    def add(role, frame, key, ascending=False):
        frame = frame[~frame.index.isin(used)]
        if frame.empty:
            return
        idx = frame.sort_values(key, ascending=ascending).index[0]
        used.add(idx)
        chosen.append((role, idx))

    ok, bad = df[df.y_true == df.y_pred], df[df.y_true != df.y_pred]
    add("acierto_alta_confianza", ok, "conf")
    add("error_alta_confianza", bad, "conf")
    near = df.assign(dist=(df.conf - tau).abs())
    add("cerca_del_umbral", near, "dist", ascending=True)
    sub = df[df.fuente_texto == "ocr"]
    if sub.empty:
        sub = df[df.canal == "fisico"]
    if sub.empty:  # sin OCR/físico: subgrupo = la clase con menor recall
        recall = (ok.groupby("y_true").size() / df.groupby("y_true").size()).sort_values()
        sub = df[df.y_true == recall.index[0]]
    if not sub.empty:
        sub_err = sub[sub.y_true != sub.y_pred]
        add("subgrupo_clase_debil", sub_err if len(sub_err) else sub, "conf", ascending=True)
    if not bad.empty:  # error típico: par (real, predicha) más frecuente, caso de confianza mediana
        pair = bad.groupby(["y_true", "y_pred"]).size().idxmax()
        cand = bad[(bad.y_true == pair[0]) & (bad.y_pred == pair[1])]
        cand = cand[~cand.index.isin(used)]
        if not cand.empty:
            idx = cand.sort_values("conf").index[len(cand) // 2]
            used.add(idx)
            chosen.append((f"error_tipico_{pair[0]}_como_{pair[1]}", idx))
    return chosen


# ---------------------------------------------------------------------------------------------------
# IG por token (se reutiliza para local, global y fidelidad)
# ---------------------------------------------------------------------------------------------------
def ig_tokens(pred, text, target, steps):
    model, tok = pred.model, pred.tok
    model.requires_grad_(False)
    enc = tok(text, truncation=True, max_length=pred.max_len, return_tensors="pt").to(pred.device)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    attr, err = integrated_gradients(model.forward_embeds, model.encoder.embeddings.word_embeddings, ids, mask,
                                     target, steps, tok.pad_token_id)
    return ids[0].cpu(), tok.convert_ids_to_tokens(ids[0]), attr, err


def words_from(tokens, attr):
    """Palabras (minúsculas) con atribución normalizada por el máximo |valor|."""
    pairs = merge_wordpieces(tokens, attr)
    scale = max((abs(v) for _, v in pairs), default=1.0) or 1.0
    return [(w.lower(), v / scale) for w, v in pairs]


# ---------------------------------------------------------------------------------------------------
# Fidelidad: borrar las k palabras más atribuidas vs k al azar
# ---------------------------------------------------------------------------------------------------
def deletion_probs(pred, ids, attr, target, ks, rng):
    tok = pred.tok
    special = set(tok.all_special_ids)
    cand = [i for i, t in enumerate(ids.tolist()) if t not in special]
    order = sorted(cand, key=lambda i: -attr[i])  # más empuje positivo hacia la clase primero
    variants = [ids.clone()]
    ks = [k for k in ks if k <= len(cand)]
    for k in ks:
        x = ids.clone()
        x[order[:k]] = tok.mask_token_id
        variants.append(x)
    for k in ks:
        x = ids.clone()
        x[list(rng.choice(cand, size=k, replace=False))] = tok.mask_token_id
        variants.append(x)
    batch = torch.stack(variants).to(pred.device)
    with torch.no_grad():
        p = torch.softmax(pred.model(batch, torch.ones_like(batch)).float() / pred.temperature, 1)[:, target].cpu().numpy()
    return {"p0": float(p[0]), "top": dict(zip(ks, p[1:1 + len(ks)])), "rand": dict(zip(ks, p[1 + len(ks):]))}


# ---------------------------------------------------------------------------------------------------
# Baseline interpretable (M0: TF-IDF + SVM lineal)
# ---------------------------------------------------------------------------------------------------
def m0_top_terms(cfg, k=10):
    from src.models.m0_tfidf_svm import build_model

    train = load_split(cfg, "train")
    m = build_model(train, cfg["seed"])
    m.fit(train["texto"], train["area_legal"])
    vec, clf = m.steps[0][1], m.steps[-1][1]
    terms = np.array(vec.get_feature_names_out())
    if hasattr(clf, "calibrated_classifiers_"):
        coef = np.mean([cc.estimator.coef_ for cc in clf.calibrated_classifiers_], axis=0)
    else:
        coef = clf.coef_
    uni = np.array([" " not in t for t in terms])  # solo unigramas, comparables con palabras de IG
    out = {}
    for ci, cls in enumerate(clf.classes_):
        w = np.where(uni, coef[ci], -np.inf)
        out[cls] = [str(terms[j]) for j in np.argsort(-w)[:k]]
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


# ---------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m3", choices=["m2", "m3", "m4"], help="el modelo que defienden")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-final", action="store_true", help="necesario para leer test (evaluación final hecha)")
    ap.add_argument("--n-per-class", type=int, default=20, help="textos por clase para la explicación global")
    ap.add_argument("--steps", type=int, default=50, help="pasos de IG (más pasos = menor error de completitud)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--suspect", default=DEFAULT_SUSPECT, help="regex de ids/códigos que podrían ser atajos")
    ap.add_argument("--device", default=None, help="forzar cpu/cuda/mps; por defecto el del predictor")
    ap.add_argument("--skip-m0", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_final:
        raise SystemExit("test está cerrado: usar --confirm-final solo si la evaluación final ya se hizo")
    cfg = load_config()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = path_of(cfg, "results") / "xai_lab10"
    out.mkdir(parents=True, exist_ok=True)
    areas, tau = list(cfg["areas"]), float(cfg["confidence_threshold"])

    from app.predictor import Predictor

    pred = Predictor.from_config(cfg, args.model)
    if args.device:
        pred.model.to(args.device)
        pred.device = args.device
    print(f"modelo {args.model} en {pred.device}, T={pred.temperature:.3f}, τ={tau}, split={args.split}")

    # --- predicciones del modelo ya entrenado sobre el conjunto elegido ---------------------------------
    data = load_split(cfg, args.split, final=args.split == "test").reset_index(drop=True)
    probs = predict_probs(pred, data["texto"].tolist())
    df = data.assign(y_true=data["area_legal"], y_pred=[areas[i] for i in probs.argmax(1)], conf=probs.max(1))
    acc = float((df.y_true == df.y_pred).mean())
    print(f"exactitud en {args.split}: {acc:.3f} ({len(df)} textos)")

    # --- explicaciones locales ----------------------------------------------------------------------------
    rows, local_words = [], []
    for role, idx in select_instances(df, tau):
        r = df.loc[idx]
        tgt_pred, tgt_true = areas.index(r.y_pred), areas.index(r.y_true)
        ids, toks, attr, err = ig_tokens(pred, r.texto, tgt_pred, args.steps)
        by = {f"explica: {r.y_pred}": words_from(toks, attr)}
        if tgt_true != tgt_pred:
            _, toks2, attr2, _ = ig_tokens(pred, r.texto, tgt_true, args.steps)
            by[f"clase real: {r.y_true}"] = words_from(toks2, attr2)
        plot_attributions(by, out / f"local_{role}_{r.case_id}.png",
                          f"{role} | {r.case_id} | conf {r.conf:.2f} | real {r.y_true}")
        for name, pairs in by.items():
            for w, v in sorted(pairs, key=lambda p: -abs(p[1]))[:10]:
                local_words.append({"rol": role, "case_id": r.case_id, "explicacion": name, "palabra": w, "atribucion": v})
        rows.append({"rol": role, "case_id": r.case_id, "y_true": r.y_true, "y_pred": r.y_pred, "conf": r.conf,
                     "canal": r.canal, "fuente_texto": r.fuente_texto, "tokens": len(toks),
                     "error_completitud": err})
    pd.DataFrame(rows).to_csv(out / "instancias.csv", index=False, encoding="utf-8")
    pd.DataFrame(local_words).to_csv(out / "local_top_words.csv", index=False, encoding="utf-8")

    # --- explicación global + fidelidad (misma pasada de IG) ------------------------------------------------
    sample = df.groupby("y_true", group_keys=False).apply(
        lambda g: g.sample(n=min(args.n_per_class, len(g)), random_state=args.seed))
    per_text, del_rows, errs, t_ig = [], [], [], []
    for _, r in sample.iterrows():
        tgt = areas.index(r.y_pred)
        t0 = time.perf_counter()
        ids, toks, attr, err = ig_tokens(pred, r.texto, tgt, args.steps)
        t_ig.append(time.perf_counter() - t0)
        errs.append(err)
        per_text.append((r.y_pred, words_from(toks, attr)))
        res = deletion_probs(pred, ids, attr.tolist(), tgt, args.ks, rng)
        for k in res["top"]:
            del_rows.append({"k": k, "p0": res["p0"], "p_borrar_top": res["top"][k], "p_borrar_azar": res["rand"][k]})
    agg = aggregate_by_class(per_text, min_count=3, k=10)
    pd.DataFrame([{"clase": c, "palabra": w, "atribucion_media": m, "n_textos": n}
                  for c, lst in agg.items() for w, m, n in lst]).to_csv(out / "global_top_words.csv", index=False,
                                                                       encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, cls in zip(axes.ravel(), areas):
        lst = agg.get(cls, [])[::-1]
        ax.barh([w for w, _, _ in lst], [m for _, m, _ in lst], color="#2a7ab0")
        ax.set(title=f"{cls}", xlabel="atribución media normalizada")
    fig.suptitle(f"IG global ({args.model}, {args.split}, {args.n_per_class} textos por clase)")
    fig.tight_layout()
    fig.savefig(out / "global_por_clase.png", dpi=130)
    plt.close(fig)

    dele = pd.DataFrame(del_rows).groupby("k").mean().reset_index()
    dele.to_csv(out / "deletion.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(dele.k, dele.p_borrar_top, "o-", label="borrar top-k por IG")
    ax.plot(dele.k, dele.p_borrar_azar, "s--", label="borrar k al azar")
    ax.set(xlabel="k palabras borradas", ylabel="prob. media de la clase predicha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "deletion.png", dpi=130)
    plt.close(fig)

    # --- baseline interpretable y atajos ----------------------------------------------------------------------
    if not args.skip_m0:
        m0 = m0_top_terms(cfg, 10)
        pd.DataFrame([{"clase": c, "top_m0": ", ".join(m0.get(c, [])),
                       "top_ig": ", ".join(w for w, _, _ in agg.get(c, [])),
                       "jaccard": jaccard(m0.get(c, []), [w for w, _, _ in agg.get(c, [])])}
                      for c in areas]).to_csv(out / "m0_vs_ig.csv", index=False, encoding="utf-8")

    pat = re.compile(args.suspect)
    hit = df[df.texto.str.contains(pat)]
    hit = hit.sample(n=min(300, len(hit)), random_state=args.seed) if len(hit) else hit
    shortcut = {"n_con_patron": int((df.texto.str.contains(pat)).sum()), "n_evaluados": len(hit)}
    if len(hit):
        p_clean = predict_probs(pred, [pat.sub(" ", t) for t in hit.texto])
        p_raw = predict_probs(pred, hit.texto.tolist())
        shortcut.update({"cambia_prediccion": float((p_raw.argmax(1) != p_clean.argmax(1)).mean()),
                         "delta_conf_media": float((p_raw.max(1) - p_clean.max(1)).mean()),
                         "exactitud_con": float((np.array(areas)[p_raw.argmax(1)] == hit.y_true.values).mean()),
                         "exactitud_sin": float((np.array(areas)[p_clean.argmax(1)] == hit.y_true.values).mean())})
    pd.DataFrame([shortcut]).to_csv(out / "atajos.csv", index=False, encoding="utf-8")

    # --- parámetros de diseño y costo ---------------------------------------------------------------------------
    import transformers

    info = {"modelo": args.model, "split": args.split, "n_split": len(df), "exactitud": acc, "tau": tau,
            "temperatura": pred.temperature, "semilla": args.seed, "metodo": "Integrated Gradients sobre embeddings",
            "referencia": "todos los tokens [PAD] excepto [CLS] y [SEP]", "pasos_ig": args.steps,
            "regla_integral": "punto medio", "clase_explicada": "la predicha (y la real en errores)",
            "normalizacion": "por el máximo |atribución| de cada texto", "n_global": len(sample),
            "min_count_global": 3, "ks_borrado": args.ks, "reemplazo_borrado": "[MASK]", "patron_atajos": args.suspect,
            "error_completitud_media": float(np.mean(errs)), "error_completitud_max": float(np.max(errs)),
            "segundos_por_explicacion_media": float(np.mean(t_ig)),
            "dispositivo": str(pred.device), "torch": torch.__version__, "transformers": transformers.__version__,
            "maquina": platform.machine(), "sistema": platform.system()}
    (out / "run_xai.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=1, ensure_ascii=False))
    print("guardado en", out)


if __name__ == "__main__":
    main()