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
import os
import shutil
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

    def forward_embeds(self, inputs_embeds, attention_mask):
        """Igual que forward pero desde embeddings de entrada (para gradientes integrados)."""
        hidden = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
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


def _amp_on(device, enabled=True):
    return enabled and str(device).startswith("cuda")


def predict_logits(model, ids, mask, device, batch_size=64):
    model.eval()
    order = torch.argsort(mask.sum(1))
    out = torch.zeros(len(ids), model.head.out_features)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=_amp_on(device)):
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            out[idx] = model(*_batch(ids, mask, idx, device)).float().cpu()
    return out


def _macro_f1(y, logits, n_classes):
    return float(f1_score(y, logits.argmax(1).numpy(), labels=list(range(n_classes)), average="macro", zero_division=0))


def _atomic_save(obj, path):
    """Guarda en un archivo temporal y lo renombra: un corte a mitad de escritura no deja un checkpoint roto."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


_CKPT_KEYS = ("lr", "layer_decay", "weight_decay", "warmup_ratio", "batch_size", "epochs", "max_len")


def fit(model, train_data, y_train, val_data, y_val, mc, device, seed=42, ckpt_dir=None, after_epoch=None, info=None):
    """Fine-tuning. train_data/val_data = (input_ids, attention_mask). Devuelve (historial, mejor_época).

    Con `ckpt_dir` guarda al final de cada época un checkpoint completo (modelo, optimizador,
    programador, escalador, historial, semillas aleatorias) y, si ya existe uno, reanuda desde la
    época siguiente: si Colab se desconecta, basta volver a correr. En GPU usa precisión mixta (fp16).
    `info` (dict opcional) recibe resumed_from_epoch, train_seconds_total y amp.
    """
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
    use_amp = _amp_on(device, mc.get("amp", True))
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    yt = torch.as_tensor(y_train)
    yv_dev = torch.as_tensor(y_val).to(device)

    ckpt = Path(ckpt_dir) if ckpt_dir else None
    history, best, best_state, best_epoch, start_epoch, elapsed, done = [], -1.0, None, 0, 1, 0.0, False
    resumed_from = 0
    if ckpt is not None and (ckpt / "state.pt").exists():
        st = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
        mine = {"seed": seed, "n_train": len(y_train), **{k: mc[k] for k in _CKPT_KEYS}}
        if {k: st["config"].get(k) for k in mine} != mine:
            raise ValueError(f"el checkpoint de {ckpt} es de otra configuración (semilla, datos o hiperparámetros): "
                             "bórralo o cambia ckpt_dir")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if scaler is not None and st["scaler"] is not None:
            scaler.load_state_dict(st["scaler"])
        _set_rng_state(st["rng"])
        history, best, best_epoch, elapsed, done = st["history"], st["best"], st["best_epoch"], st["elapsed"], st["done"]
        start_epoch, resumed_from = st["epoch"] + 1, st["epoch"]
        if best_epoch and (ckpt / "best.pt").exists():
            best_state = torch.load(ckpt / "best.pt", map_location="cpu", weights_only=False)
        print(f"  reanudando desde la época {st['epoch']} de {mc['epochs']} (ya entrenados {elapsed / 60:.1f} min)", flush=True)

    session = time.perf_counter()
    for epoch in range(start_epoch, mc["epochs"] + 1):
        if done:
            break
        model.train()
        start = time.perf_counter()
        perm = torch.randperm(len(yt), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        total = 0.0
        for i in range(0, len(perm), mc["batch_size"]):
            idx = perm[i:i + mc["batch_size"]]
            ids, mask = _batch(*train_data, idx, device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(ids, mask)
            loss = loss_fn(logits.float(), yt[idx].to(device))
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), mc["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
            else:
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
        improved = f1 > best
        if improved:
            best, best_epoch = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        done = (not improved and epoch - best_epoch >= mc["patience"]) or epoch == mc["epochs"]
        print(f"  época {epoch}/{mc['epochs']}: pérdida {history[-1]['train_loss']:.3f}, F1 val {f1:.3f}"
              f"{' (mejor)' if improved else ''}, {history[-1]['epoch_seconds'] / 60:.1f} min", flush=True)
        if ckpt is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
            if improved:
                _atomic_save(best_state, ckpt / "best.pt")
            _atomic_save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                          "scaler": scaler.state_dict() if scaler is not None else None, "history": history,
                          "best": best, "best_epoch": best_epoch, "done": done, "rng": _rng_state(),
                          "elapsed": elapsed + time.perf_counter() - session,
                          "config": {"seed": seed, "n_train": len(y_train), **{k: mc[k] for k in _CKPT_KEYS}}},
                         ckpt / "state.pt")
            (ckpt / "status.json").write_text(json.dumps(
                {"epoca": epoch, "epocas_totales": mc["epochs"], "f1_val": round(f1, 4), "mejor_epoca": best_epoch,
                 "mejor_f1_val": round(best, 4), "terminado": done, "minutos": round((elapsed + time.perf_counter() - session) / 60, 1)}),
                encoding="utf-8")
        if after_epoch is not None:
            after_epoch(epoch)
        if done:
            break
    model.load_state_dict(best_state)
    if info is not None:
        info.update({"resumed_from_epoch": resumed_from, "amp": use_amp,
                     "train_seconds_total": elapsed + time.perf_counter() - session})
    return history, best_epoch


def load_finetuned(cfg, model_id="m3", checkpoint=None, weights_dir=None, device="cpu"):
    """Reconstruye (tokenizador, modelo) desde los pesos guardados con --save. Para M4 pasar
    checkpoint=<artifacts/m4_dapt>, que es el encoder con el que se hizo el fine-tuning."""
    from transformers import AutoModel, AutoTokenizer

    mc = cfg["models"]["m3"]
    checkpoint = str(checkpoint or mc["checkpoint"])
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = BertClassifier(AutoModel.from_pretrained(checkpoint), len(cfg["areas"]), mc["pooling"], mc["dropout"])
    weights = Path(weights_dir or path_of(cfg, "artifacts")) / f"{model_id}.pt"
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return tok, model.to(device).eval()


def run(cfg, eval_split="val", final=False, seed=None, train_df=None, data_dir=None, out_dir=None,
        overrides=None, tag=None, plot=True, save=False, ckpt_dir=None, resume=True, keep_ckpt=False):
    """Fine-tuning de BETO y escritura de predicciones. `save=True` guarda los pesos en artifacts/.

    Con `ckpt_dir` se guarda un checkpoint por época y se reanuda si se corta; con `resume=True` una
    corrida ya terminada (mismas predicciones y mismo número de textos de entrenamiento) se omite.
    """
    from transformers import AutoModel, AutoTokenizer

    seed = cfg["seed"] if seed is None else seed
    mc = {**cfg["models"]["m3"], **(overrides or {})}
    areas = list(cfg["areas"])
    to_idx = {a: i for i, a in enumerate(areas)}
    model_id = tag or (MODEL_ID if seed == cfg["seed"] else f"{MODEL_ID}_s{seed}")
    out_dir = Path(out_dir or path_of(cfg, "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{model_id}_{eval_split}_preds.csv"
    run_json = out_dir / f"{model_id}_{eval_split}_run.json"

    train = train_df if train_df is not None else load_split(cfg, "train", data_dir=data_dir)
    if resume and preds_path.exists() and run_json.exists():
        prev = json.loads(run_json.read_text(encoding="utf-8"))
        if prev.get("n_train") == len(train) and prev.get("seed") == seed:
            print(f"{model_id} ({eval_split}) ya está terminado: se omite", flush=True)
            return str(preds_path)
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

    fit_info = {}
    history, best_epoch = fit(model, data["train"], y_train, data["val"], y_val, mc, device, seed,
                              ckpt_dir=ckpt_dir, info=fit_info)
    train_s = fit_info["train_seconds_total"]

    start = time.perf_counter()
    probs = torch.softmax(predict_logits(model, *data[eval_split], device), 1).numpy()
    infer_ms = 1000 * (time.perf_counter() - start) / len(ev)
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
        "peak_memory_mb": peak_memory_mb(device), "amp": fit_info["amp"],
        "resumed_from_epoch": fit_info["resumed_from_epoch"], "history": history,
    }
    run_json.write_text(json.dumps(info, indent=1), encoding="utf-8")
    if plot:
        plot_history(history, out_dir / f"{model_id}_{eval_split}_curves.png", title=model_id)
    if save:
        weights = path_of(cfg, "artifacts") / f"{model_id}.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save(model.state_dict(), weights)
    if ckpt_dir is not None and not keep_ckpt:
        shutil.rmtree(ckpt_dir, ignore_errors=True)  # el checkpoint ya no hace falta; libera espacio en Drive
    return str(preds_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", action="store_true", help="guardar pesos en artifacts/ (para la API)")
    ap.add_argument("--force", action="store_true", help="reentrenar aunque ya exista una corrida terminada")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"] if args.seed is None else args.seed
    model_id = MODEL_ID if seed == cfg["seed"] else f"{MODEL_ID}_s{seed}"
    ckpt = path_of(cfg, "artifacts") / "ckpt" / f"{model_id}_{args.split}"  # un checkpoint por época, se reanuda solo
    print(run(cfg, args.split, final=args.final, seed=args.seed, save=args.save, ckpt_dir=ckpt, resume=not args.force))
