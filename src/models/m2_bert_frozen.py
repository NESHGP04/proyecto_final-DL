"""M2: BETO congelado + cabeza lineal. MODELO BASE de transfer learning. Dueño: Nes (CP1).

El encoder no se entrena: se extraen los embeddings una vez (mean pooling o [CLS]) y solo se
entrena la cabeza de clasificación. La época se elige por F1 macro en val; con eval_split="val"
esa métrica es optimista por la selección (M0 no usa val para elegir nada).

Uso: python -m src.models.m2_bert_frozen --split val [--seed 43]
El split `test` solo con --final (evaluación final con configuración congelada).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.config import load_config, path_of
from src.data.dataset import load_split
from src.evaluation.preds import write_preds

MODEL_ID = "m2"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def pool(hidden, mask, mode):
    """hidden: (B, T, H); mask: (B, T) con 1 en tokens reales. Devuelve (B, H)."""
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    raise ValueError(f"pooling desconocido: {mode}")


def embed_texts(texts, checkpoint, max_len, pooling, batch_size=32, device=None, cache_dir=None):
    """Embeddings del encoder congelado. Devuelve (array float32 (n, H), segundos, n_calculados)."""
    texts = list(texts)
    key = hashlib.sha1("\x00".join([checkpoint, str(max_len), pooling, *texts]).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"emb_{key}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        return np.load(cache), 0.0, 0

    from transformers import AutoModel, AutoTokenizer
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(checkpoint)
    enc = AutoModel.from_pretrained(checkpoint).to(device).eval()
    for prm in enc.parameters():
        prm.requires_grad = False

    order = np.argsort([len(t) for t in texts])  # agrupar por largo reduce el relleno
    out = np.zeros((len(texts), enc.config.hidden_size), dtype=np.float32)
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            batch = tok([texts[j] for j in idx], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
            hidden = enc(**batch).last_hidden_state
            out[idx] = pool(hidden, batch["attention_mask"], pooling).float().cpu().numpy()
    seconds = time.perf_counter() - start
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out, seconds, len(texts)


class Head:
    """Cabeza lineal con estandarización de las características (media y desvío de train)."""

    def __init__(self, mean, std, linear):
        self.mean, self.std, self.linear = mean, std, linear

    def predict_proba(self, X):
        x = torch.as_tensor((X - self.mean) / self.std, dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(self.linear(x), dim=1).numpy()


def train_head(X_train, y_train, X_val, y_val, n_classes, lr=1e-3, epochs=20, batch_size=32, seed=42):
    """Entrena la cabeza (CPU) y devuelve (Head, historial, mejor_época).

    y_* son índices de clase enteros. Pérdida con pesos por clase inversos a la frecuencia. Se
    conserva la época con mayor F1 macro en val.
    """
    torch.manual_seed(seed)
    mean, std = X_train.mean(0), X_train.std(0) + 1e-6
    xt = torch.as_tensor((X_train - mean) / std, dtype=torch.float32)
    xv = torch.as_tensor((X_val - mean) / std, dtype=torch.float32)
    yt, yv = torch.as_tensor(y_train), torch.as_tensor(y_val)

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = torch.as_tensor(len(y_train) / (n_classes * np.maximum(counts, 1)))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    linear = torch.nn.Linear(xt.shape[1], n_classes)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=1e-2)
    gen = torch.Generator().manual_seed(seed)

    history, best, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        linear.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(linear(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        linear.eval()
        with torch.no_grad():
            logits = linear(xv)
            val_loss = loss_fn(logits, yv).item()
            f1 = f1_score(y_val, logits.argmax(1).numpy(), labels=list(range(n_classes)),
                          average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total / len(xt), "val_loss": val_loss, "val_macro_f1": float(f1)})
        if f1 > best:
            best, best_epoch = f1, epoch
            best_state = {k: v.clone() for k, v in linear.state_dict().items()}
    linear.load_state_dict(best_state)
    return Head(mean, std, linear.eval()), history, best_epoch


def run(cfg, eval_split="val", final=False, seed=None, train_df=None, data_dir=None, out_dir=None, cache_dir=None):
    seed = cfg["seed"] if seed is None else seed
    mc = cfg["models"]["m2"]
    areas = list(cfg["areas"])
    to_idx = {a: i for i, a in enumerate(areas)}
    cache_dir = cache_dir or path_of(cfg, "artifacts") / "emb_cache"

    train = train_df if train_df is not None else load_split(cfg, "train", data_dir=data_dir)
    val = load_split(cfg, "val", data_dir=data_dir)
    ev = val if eval_split == "val" else load_split(cfg, eval_split, final=final, data_dir=data_dir)

    device = pick_device()
    embed_s, n_embedded = 0.0, 0
    feats = {}
    for name, df in (("train", train), ("val", val), (eval_split, ev)):
        if name in feats:
            continue
        feats[name], s, n = embed_texts(df["texto"], mc["checkpoint"], mc["max_len"], mc["pooling"],
                                        mc["embed_batch_size"], device, cache_dir)
        embed_s, n_embedded = embed_s + s, n_embedded + n

    y_train = train["area_legal"].map(to_idx).to_numpy()
    y_val = val["area_legal"].map(to_idx).to_numpy()
    start = time.perf_counter()
    head, history, best_epoch = train_head(feats["train"], y_train, feats["val"], y_val, len(areas),
                                           mc["lr"], mc["epochs"], mc["batch_size"], seed)
    head_s = time.perf_counter() - start

    probs = head.predict_proba(feats[eval_split])
    model_id = MODEL_ID if seed == cfg["seed"] else f"{MODEL_ID}_s{seed}"
    out_dir = Path(out_dir or path_of(cfg, "results"))
    preds_path = out_dir / f"{model_id}_{eval_split}_preds.csv"
    write_preds(preds_path, ev["case_id"], ev["area_legal"], probs, areas)

    n_head = sum(p.numel() for p in head.linear.parameters())
    info = {
        "model": model_id, "checkpoint": mc["checkpoint"], "pooling": mc["pooling"], "seed": seed,
        "device": device, "n_train": len(train), "best_epoch": best_epoch,
        "trainable_params": n_head, "embed_seconds": round(embed_s, 2), "texts_embedded": n_embedded,
        "head_train_seconds": round(head_s, 2), "history": history,
    }
    (out_dir / f"{model_id}_{eval_split}_run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return str(preds_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    print(run(load_config(), args.split, final=args.final, seed=args.seed))
