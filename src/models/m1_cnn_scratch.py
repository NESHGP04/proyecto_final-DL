"""M1: modelo propio entrenado desde cero (tokenizador propio + embeddings + CNN 1D).

Sin preentrenamiento: el vocabulario sale solo de train y los embeddings se aprenden con la tarea.
Arquitectura (estilo Kim 2014): embedding -> Conv1d con varios anchos de filtro -> ReLU -> max
global por filtro -> dropout -> capa lineal. Pérdida con pesos por clase inversos a la frecuencia.
Early stopping por F1 macro en val (con eval_split="val" esa métrica es optimista por la selección).

Uso: python -m src.models.m1_cnn_scratch --split val [--seed 43]
El split `test` solo con --final (evaluación final con configuración congelada).
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.config import load_config, path_of
from src.data.dataset import load_split
from src.evaluation.curves import plot_history
from src.evaluation.preds import write_preds

MODEL_ID = "m1"
PAD, UNK = 0, 1


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def build_vocab(texts, min_freq=2, max_vocab=20000):
    """Vocabulario solo con textos de train. 0 = relleno, 1 = desconocida."""
    counts = Counter(tok for t in texts for tok in tokenize(t))
    words = [w for w, c in counts.most_common(max_vocab - 2) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(texts, vocab, max_len):
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = [vocab.get(w, UNK) for w in tokenize(t)][:max_len]
        ids[i, :len(toks)] = toks
    return torch.as_tensor(ids)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, embed_dim=128, filters=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, filters, k) for k in kernel_sizes)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(kernel_sizes), n_classes)

    def forward(self, ids):
        x = self.embed(ids).transpose(1, 2)                      # (B, E, T)
        feats = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.out(self.drop(torch.cat(feats, dim=1)))


def _logits(model, ids, batch_size=256):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(ids[i:i + batch_size]) for i in range(0, len(ids), batch_size)])


def _predict(model, ids):
    return torch.softmax(_logits(model, ids), 1).numpy()


def _macro_f1(y, probs, n_classes):
    return float(f1_score(y, probs.argmax(1), labels=list(range(n_classes)), average="macro", zero_division=0))


def train_model(ids_train, y_train, ids_val, y_val, vocab_size, n_classes, mc, seed=42):
    """Entrena M1 (CPU) y devuelve (modelo, historial, mejor_época)."""
    torch.manual_seed(seed)
    model = TextCNN(vocab_size, n_classes, mc["embed_dim"], mc["filters"], mc["kernel_sizes"], mc["dropout"])
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = torch.as_tensor(len(y_train) / (n_classes * np.maximum(counts, 1)))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    yt, yv = torch.as_tensor(y_train), torch.as_tensor(y_val)
    gen = torch.Generator().manual_seed(seed)

    history, best, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, mc["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(ids_train), generator=gen)
        total = 0.0
        for i in range(0, len(perm), mc["batch_size"]):
            b = perm[i:i + mc["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(ids_train[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        logits_val = _logits(model, ids_val)
        f1 = _macro_f1(y_val, torch.softmax(logits_val, 1).numpy(), n_classes)
        history.append({
            "epoch": epoch, "train_loss": total / len(perm),
            "val_loss": loss_fn(logits_val, yv).item(),
            "train_macro_f1": _macro_f1(y_train, _predict(model, ids_train), n_classes),
            "val_macro_f1": f1,
        })
        if f1 > best:
            best, best_epoch = f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= mc["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def run(cfg, eval_split="val", final=False, seed=None, train_df=None, data_dir=None, out_dir=None,
        overrides=None, tag=None, plot=True):
    """Entrena M1 y escribe predicciones. `overrides` cambia hiperparámetros de config (búsqueda)."""
    seed = cfg["seed"] if seed is None else seed
    mc = {**cfg["models"]["m1"], **(overrides or {})}
    areas = list(cfg["areas"])
    to_idx = {a: i for i, a in enumerate(areas)}

    train = train_df if train_df is not None else load_split(cfg, "train", data_dir=data_dir)
    val = load_split(cfg, "val", data_dir=data_dir)
    ev = val if eval_split == "val" else load_split(cfg, eval_split, final=final, data_dir=data_dir)

    vocab = build_vocab(train["texto"], mc["min_freq"], mc["max_vocab"])
    enc = lambda df: encode(list(df["texto"]), vocab, mc["max_len"])  # noqa: E731
    y_train = train["area_legal"].map(to_idx).to_numpy()
    y_val = val["area_legal"].map(to_idx).to_numpy()

    start = time.perf_counter()
    model, history, best_epoch = train_model(enc(train), y_train, enc(val), y_val,
                                             len(vocab) + 2, len(areas), mc, seed)
    train_s = time.perf_counter() - start

    ids_ev = enc(ev)
    start = time.perf_counter()
    probs = _predict(model, ids_ev)
    infer_ms = 1000 * (time.perf_counter() - start) / len(ev)

    model_id = tag or (MODEL_ID if seed == cfg["seed"] else f"{MODEL_ID}_s{seed}")
    out_dir = Path(out_dir or path_of(cfg, "results"))
    preds_path = out_dir / f"{model_id}_{eval_split}_preds.csv"
    write_preds(preds_path, ev["case_id"], ev["area_legal"], probs, areas)

    info = {
        "model": model_id, "seed": seed, "hparams": {k: mc[k] for k in
                ("max_len", "embed_dim", "filters", "kernel_sizes", "dropout", "lr", "weight_decay", "batch_size")},
        "n_train": len(train), "vocab_size": len(vocab) + 2, "best_epoch": best_epoch,
        "epochs_run": len(history), "best_val_macro_f1": max(h["val_macro_f1"] for h in history),
        "trainable_params": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(train_s, 2), "inference_ms_per_text": round(infer_ms, 3),
        "history": history,
    }
    (out_dir / f"{model_id}_{eval_split}_run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if plot:
        plot_history(history, out_dir / f"{model_id}_{eval_split}_curves.png", title=model_id)
    return str(preds_path)


def search(cfg, grid, data_dir=None, out_dir=None):
    """Búsqueda pequeña de hiperparámetros, SOLO con val. grid: lista de dicts de overrides.

    Guarda en m1_search.json qué se probó y su F1 macro en val; anotar en el informe por qué.
    """
    out_dir = Path(out_dir or path_of(cfg, "results"))
    rows = []
    for i, ov in enumerate(grid):
        run(cfg, "val", data_dir=data_dir, out_dir=out_dir, overrides=ov, tag=f"m1_search{i}", plot=False)
        info = json.loads((out_dir / f"m1_search{i}_val_run.json").read_text(encoding="utf-8"))
        rows.append({"overrides": ov, "best_val_macro_f1": info["best_val_macro_f1"],
                     "best_epoch": info["best_epoch"], "train_seconds": info["train_seconds"]})
    (out_dir / "m1_search.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    print(run(load_config(), args.split, final=args.final, seed=args.seed))
