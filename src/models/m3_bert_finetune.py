"""M3: BETO con fine-tuning completo. Técnica principal.

Misma arquitectura que M2 (encoder + pooling + dropout + capa lineal) con una sola diferencia: aquí
el encoder se descongela. Así el delta M2 -> M3 mide el efecto de hacer fine-tuning y no un cambio
de cabeza. Se usa mean pooling y no el pooler de BERT porque BETO no trae pooler preentrenado.

Receta: AdamW con tasa menor en capas inferiores (layer_decay), warmup lineal, weight decay sin
sesgos ni LayerNorm, recorte de gradiente, pérdida con pesos por clase y early stopping por F1 macro
en val (con eval_split="val" esa métrica es optimista por la selección).

Uso: python -m src.models.m3_bert_finetune --split val [--seed 43] [--save]
El split `test` solo con --final (evaluación final con configuración congelada).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.config import load_config, path_of
from src.data.dataset import load_split
from src.evaluation.curves import plot_history
from src.evaluation.preds import write_preds
from src.models.m2_bert_frozen import peak_memory_mb, pick_device, pool

MODEL_ID = "m3"


class BertClassifier(nn.Module):
    def __init__(self, encoder, n_classes, pooling="mean", dropout=0.1):
        super().__init__()
        self.encoder, self.pooling = encoder, pooling
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.drop(pool(hidden, attention_mask, self.pooling)))


def build_param_groups(model, lr, layer_decay, weight_decay):
    """Grupos de parámetros con tasa menor en las capas inferiores.

    La cabeza usa `lr`; la capa más alta del encoder `lr * layer_decay`; cada capa inferior se
    multiplica otra vez por layer_decay; los embeddings reciben la tasa más baja. Bias y LayerNorm
    van sin weight decay.
    """
    layers = model.encoder.encoder.layer
    n = len(layers)

    def scale(name):
        if name.startswith("encoder.embeddings"):
            return layer_decay ** (n + 1)
        if name.startswith("encoder.encoder.layer."):
            return layer_decay ** (n - int(name.split(".")[3]))
        if name.startswith("encoder.pooler"):
            return layer_decay
        return 1.0  # cabeza

    groups = {}
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        no_decay = name.endswith("bias") or "LayerNorm" in name
        key = (scale(name), 0.0 if no_decay else weight_decay)
        groups.setdefault(key, []).append(prm)
    return [{"params": ps, "lr": lr * sc, "weight_decay": wd} for (sc, wd), ps in groups.items()]


def tokenize_all(tok, texts, max_len):
    enc = tok(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def _batch(ids, mask, idx, device):
    longest = int(mask[idx].sum(1).max())  # recorta el relleno sobrante del lote
    return ids[idx, :longest].to(device), mask[idx, :longest].to(device)


def predict_logits(model, ids, mask, device, batch_size=64):
    model.eval()
    order = torch.argsort(mask.sum(1))
    out = torch.zeros(len(ids), model.head.out_features)
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            out[idx] = model(*_batch(ids, mask, idx, device)).float().cpu()
    return out


def _macro_f1(y, logits, n_classes):
    return float(f1_score(y, logits.argmax(1).numpy(), labels=list(range(n_classes)), average="macro", zero_division=0))


def fit(model, train_data, y_train, val_data, y_val, mc, device, seed=42):
    """Fine-tuning. train_data/val_data = (input_ids, attention_mask). Devuelve (historial, mejor_época)."""
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    n_classes = model.head.out_features
    model.to(device)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = torch.as_tensor(len(y_train) / (n_classes * np.maximum(counts, 1))).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(build_param_groups(model, mc["lr"], mc["layer_decay"], mc["weight_decay"]))
    steps_per_epoch = -(-len(y_train) // mc["batch_size"])
    total_steps = steps_per_epoch * mc["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(mc["warmup_ratio"] * total_steps), total_steps)
    yt = torch.as_tensor(y_train)
    yv_dev = torch.as_tensor(y_val).to(device)
    gen = torch.Generator().manual_seed(seed)

    history, best, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, mc["epochs"] + 1):
        model.train()
        start = time.perf_counter()
        perm = torch.randperm(len(yt), generator=gen)
        total = 0.0
        for i in range(0, len(perm), mc["batch_size"]):
            idx = perm[i:i + mc["batch_size"]]
            ids, mask = _batch(*train_data, idx, device)
            opt.zero_grad()
            loss = loss_fn(model(ids, mask), yt[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), mc["max_grad_norm"])
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        logits_val = predict_logits(model, *val_data, device)
        f1 = _macro_f1(y_val, logits_val, n_classes)
        history.append({
            "epoch": epoch, "train_loss": total / len(yt),
            "val_loss": loss_fn(logits_val.to(device), yv_dev).item(), "val_macro_f1": f1,
            "epoch_seconds": round(time.perf_counter() - start, 2),
        })
        if f1 > best:
            best, best_epoch = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= mc["patience"]:
            break
    model.load_state_dict(best_state)
    return history, best_epoch


def run(cfg, eval_split="val", final=False, seed=None, train_df=None, data_dir=None, out_dir=None,
        overrides=None, tag=None, plot=True, save=False):
    """Fine-tuning de BETO y escritura de predicciones. `save=True` guarda los pesos en artifacts/."""
    from transformers import AutoModel, AutoTokenizer

    seed = cfg["seed"] if seed is None else seed
    mc = {**cfg["models"]["m3"], **(overrides or {})}
    areas = list(cfg["areas"])
    to_idx = {a: i for i, a in enumerate(areas)}

    train = train_df if train_df is not None else load_split(cfg, "train", data_dir=data_dir)
    val = load_split(cfg, "val", data_dir=data_dir)
    ev = val if eval_split == "val" else load_split(cfg, eval_split, final=final, data_dir=data_dir)

    device = pick_device()
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(mc["checkpoint"])
    model = BertClassifier(AutoModel.from_pretrained(mc["checkpoint"]), len(areas), mc["pooling"], mc["dropout"])
    data = {name: tokenize_all(tok, df["texto"], mc["max_len"])
            for name, df in (("train", train), ("val", val), (eval_split, ev))}
    y_train = train["area_legal"].map(to_idx).to_numpy()
    y_val = val["area_legal"].map(to_idx).to_numpy()

    start = time.perf_counter()
    history, best_epoch = fit(model, data["train"], y_train, data["val"], y_val, mc, device, seed)
    train_s = time.perf_counter() - start

    start = time.perf_counter()
    probs = torch.softmax(predict_logits(model, *data[eval_split], device), 1).numpy()
    infer_ms = 1000 * (time.perf_counter() - start) / len(ev)

    model_id = tag or (MODEL_ID if seed == cfg["seed"] else f"{MODEL_ID}_s{seed}")
    out_dir = Path(out_dir or path_of(cfg, "results"))
    preds_path = out_dir / f"{model_id}_{eval_split}_preds.csv"
    write_preds(preds_path, ev["case_id"], ev["area_legal"], probs, areas)

    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "model": model_id, "checkpoint": mc["checkpoint"], "seed": seed, "device": device,
        "hparams": {k: mc[k] for k in ("max_len", "pooling", "dropout", "lr", "layer_decay", "warmup_ratio",
                                       "weight_decay", "max_grad_norm", "batch_size", "epochs", "patience")},
        "n_train": len(train), "best_epoch": best_epoch, "epochs_run": len(history),
        "best_val_macro_f1": max(h["val_macro_f1"] for h in history),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "total_params": n_params,
        "train_seconds": round(train_s, 2), "inference_ms_per_text": round(infer_ms, 2),
        "peak_memory_mb": peak_memory_mb(device), "history": history,
    }
    (out_dir / f"{model_id}_{eval_split}_run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if plot:
        plot_history(history, out_dir / f"{model_id}_{eval_split}_curves.png", title=model_id)
    if save:
        torch.save(model.state_dict(), path_of(cfg, "artifacts") / f"{model_id}.pt")
    return str(preds_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", action="store_true", help="guardar pesos en artifacts/ (para la API)")
    args = ap.parse_args()
    print(run(load_config(), args.split, final=args.final, seed=args.seed, save=args.save))
