"""Explicabilidad con gradientes integrados (Sundararajan et al., 2017) sobre los embeddings de entrada.

Para un texto y una clase objetivo, la atribución de cada token mide cuánto empuja el logit de esa
clase al ir de una referencia (todos los tokens [PAD], dejando [CLS] y [SEP]) al texto real. Se usa
sobre M2 (encoder congelado) y M3/M4 (fine-tuning) con el mismo método, para ver qué palabras usa
cada uno y qué cambia con el fine-tuning. No se usa la atención de BERT: con mean pooling no hay un
token que concentre la decisión y la atención no es una explicación fiable por sí sola.

Los textos son de casos reales: las salidas van a artifacts/explain/ (ignorado por git). Lo que
se publique en el informe debe ser un ejemplo inventado o muy editado.

Uso: python -m src.explain.explain --split val --n 3 --models m2 m3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, path_of
from src.data.dataset import load_split

SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}


def integrated_gradients(forward_embeds, embeddings, ids, mask, target, steps=24, pad_id=0):
    """Atribución por token (T,) a la clase `target`, con regla del punto medio.

    Devuelve (atribuciones, error_de_completitud). Por construcción sum(atribuciones) ≈
    logit(texto) − logit(referencia); el error relativo dice qué tan buena es la aproximación
    (con pocos pasos o redes muy no lineales puede ser alto).
    """
    with torch.no_grad():
        x = embeddings(ids)
        base_ids = ids.clone()
        base_ids[:, 1:-1] = pad_id  # conserva [CLS] y [SEP]
        x0 = embeddings(base_ids)
        f_x = forward_embeds(x, mask)[0, target].item()
        f_0 = forward_embeds(x0, mask)[0, target].item()
    total = torch.zeros_like(x)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        xi = (x0 + alpha * (x - x0)).clone().requires_grad_(True)
        forward_embeds(xi, mask)[0, target].backward()
        total += xi.grad
    attr = ((x - x0) * total / steps).sum(-1)[0]
    gap = f_x - f_0
    err = abs(attr.sum().item() - gap) / max(abs(gap), 1e-9)
    return attr.detach().cpu().numpy(), err


def merge_wordpieces(tokens, scores):
    """Une los trozos '##' en palabras (sumando sus atribuciones) y descarta tokens especiales."""
    words, vals = [], []
    for tok, sc in zip(tokens, scores):
        if tok in SPECIAL:
            continue
        if tok.startswith("##") and words:
            words[-1] += tok[2:]
            vals[-1] += float(sc)
        else:
            words.append(tok)
            vals.append(float(sc))
    return list(zip(words, vals))


def explain_text(model, tok, text, target=None, max_len=256, steps=24, device="cpu"):
    """Devuelve (clase_explicada, [(palabra, atribución)], error_completitud).

    Si target es None se explica la clase que predice el modelo. Las atribuciones van normalizadas
    por el máximo valor absoluto, para poder comparar modelos con escalas de logit distintas.
    """
    model.requires_grad_(False)  # solo interesan los gradientes respecto a los embeddings de entrada
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    if target is None:
        with torch.no_grad():
            target = int(model(ids, mask).argmax(1))
    attr, err = integrated_gradients(model.forward_embeds, model.encoder.embeddings.word_embeddings, ids, mask,
                                     target, steps, tok.pad_token_id)
    pairs = merge_wordpieces(tok.convert_ids_to_tokens(ids[0]), attr)
    scale = max((abs(v) for _, v in pairs), default=1.0) or 1.0
    return target, [(w, v / scale) for w, v in pairs], err


def top_words(pairs, k=5):
    """Las k palabras con mayor atribución positiva (las que más empujan hacia la clase)."""
    return [w for w, v in sorted(pairs, key=lambda p: -p[1])[:k] if v > 0]


def overlap(a, b):
    """Jaccard entre dos listas de palabras (1 = mismas palabras, 0 = ninguna en común)."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def aggregate_by_class(per_text, min_count=2, k=10):
    """Palabras con mayor atribución media por clase. per_text: lista de (clase, [(palabra, valor)])."""
    acc = {}
    for cls, pairs in per_text:
        for w, v in pairs:
            acc.setdefault(cls, {}).setdefault(w, []).append(v)
    return {cls: [(w, float(np.mean(v)), len(v)) for w, v in
                  sorted(((w, v) for w, v in ws.items() if len(v) >= min_count), key=lambda x: -np.mean(x[1]))[:k]]
            for cls, ws in acc.items()}


def plot_attributions(by_model, path, title="", k=8):
    """Barras horizontales de las k palabras con más |atribución| por modelo (una columna por modelo)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(by_model), figsize=(4.6 * len(by_model), 3.6), squeeze=False)
    for ax, (name, pairs) in zip(axes[0], by_model.items()):
        top = sorted(pairs, key=lambda p: -abs(p[1]))[:k][::-1]
        ax.barh([w for w, _ in top], [v for _, v in top],
                color=["#2a7ab0" if v > 0 else "#c8553d" for _, v in top])
        ax.axvline(0, color="black", lw=0.6)
        ax.set(title=name, xlabel="atribución normalizada")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _load(cfg, name, device):
    if name == "m2":
        from src.models.m2_bert_frozen import load_frozen
        return load_frozen(cfg, device=device)
    from src.models.m3_bert_finetune import load_finetuned
    if name == "m4":
        return load_finetuned(cfg, "m4", checkpoint=path_of(cfg, "artifacts") / "m4_dapt", device=device)
    return load_finetuned(cfg, name, device=device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val"])  # test solo en la evaluación final
    ap.add_argument("--n", type=int, default=3, help="casos por clase")
    ap.add_argument("--models", nargs="+", default=["m2", "m3"], choices=["m2", "m3", "m4"])
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    cfg = load_config()
    from src.models.m2_bert_frozen import pick_device
    device = pick_device()
    df = load_split(cfg, args.split)
    sample = df.groupby("area_legal", group_keys=False).head(args.n)
    out = path_of(cfg, "artifacts") / "explain"
    out.mkdir(parents=True, exist_ok=True)
    models = {m: _load(cfg, m, device) for m in args.models}
    areas = list(cfg["areas"])
    summary = []
    for _, row in sample.iterrows():
        by_model, tops = {}, {}
        for name, (tok, model) in models.items():
            cls, pairs, err = explain_text(model, tok, row["texto"], areas.index(row["area_legal"]),
                                           cfg["models"]["m3"]["max_len"], args.steps, device)
            by_model[name] = pairs
            tops[name] = top_words(pairs)
            print(f"{row['case_id']} {name}: clase {areas[cls]} error de completitud {err:.2f} top {tops[name]}")
        plot_attributions(by_model, out / f"{row['case_id']}.png", f"{row['case_id']} ({row['area_legal']})")
        names = list(tops)
        summary.append({"case_id": row["case_id"], "area": row["area_legal"], "top": tops,
                        "overlap": overlap(tops[names[0]], tops[names[-1]]) if len(names) > 1 else None})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("guardado en", out)
