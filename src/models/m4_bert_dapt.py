"""M4 (opcional): M3 precedido de adaptación de dominio (preentrenamiento MLM sobre texto legal).

Paso 1 (dapt): se continúa el preentrenamiento de BETO con enmascarado de palabras (MLM) sobre
texto legal en español: los archivos .txt de data/external/ y, si include_train_texts, los textos
de train sin etiquetas. Paso 2: fine-tuning igual que M3, pero partiendo del encoder adaptado. Así
el delta M3 -> M4 mide el efecto de la adaptación de dominio. Costo total = dapt + fine-tuning.

Uso: python -m src.models.m4_bert_dapt --split val
El split `test` solo con --final (evaluación final con configuración congelada).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.config import load_config, path_of
from src.data.dataset import load_split
from src.models import m3_bert_finetune
from src.models.m2_bert_frozen import peak_memory_mb, pick_device

MODEL_ID = "m4"


def corpus_texts(cfg, include_train=True, data_dir=None, external_dir=None):
    """Textos para la adaptación: un párrafo (separado por línea en blanco) por texto."""
    mc = cfg["models"]["m4"]
    texts = []
    ext = Path(external_dir or path_of(cfg, "external"))
    for f in sorted(ext.glob(mc["external_glob"])):
        for para in f.read_text(encoding="utf-8").split("\n\n"):
            if len(para.split()) >= 5:
                texts.append(" ".join(para.split()))
    if include_train:
        texts.extend(load_split(cfg, "train", data_dir=data_dir)["texto"].tolist())
    return texts


def dapt(cfg, texts, out_dir, seed=None, overrides=None):
    """Continúa el preentrenamiento MLM sobre `texts` y guarda encoder + tokenizador en out_dir."""
    from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              get_linear_schedule_with_warmup)

    seed = cfg["seed"] if seed is None else seed
    mc = {**cfg["models"]["m4"], **(overrides or {})}
    checkpoint = (overrides or {}).get("checkpoint", cfg["models"]["m3"]["checkpoint"])
    if len(texts) < 2:
        raise ValueError("no hay textos para la adaptación: poner .txt en data/external/ o usar los de train")

    device = pick_device()
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model, loading = AutoModelForMaskedLM.from_pretrained(checkpoint, output_loading_info=True)
    mlm_head_pretrained = not any(k.startswith("cls.predictions") for k in loading["missing_keys"])
    model.to(device).train()
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=mc["mlm_probability"])
    feats = [{"input_ids": ids} for ids in tok(texts, truncation=True, max_length=mc["dapt_max_len"])["input_ids"]]

    steps = -(-len(feats) // mc["dapt_batch_size"]) * mc["dapt_epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=mc["dapt_lr"], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    gen = torch.Generator().manual_seed(seed)

    history, start = [], time.perf_counter()
    for epoch in range(1, mc["dapt_epochs"] + 1):
        perm = torch.randperm(len(feats), generator=gen).tolist()
        total, n = 0.0, 0
        for i in range(0, len(perm), mc["dapt_batch_size"]):
            batch = collator([feats[j] for j in perm[i:i + mc["dapt_batch_size"]]])
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        history.append({"epoch": epoch, "mlm_loss": total / n})
    seconds = time.perf_counter() - start

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"texts": len(feats), "epochs": mc["dapt_epochs"], "seconds": round(seconds, 2), "device": device,
            "peak_memory_mb": peak_memory_mb(device), "mlm_head_pretrained": mlm_head_pretrained,
            "history": history}


def run(cfg, eval_split="val", final=False, seed=None, train_df=None, data_dir=None, out_dir=None,
        overrides=None, tag=None, plot=True, save=False, dapt_dir=None, external_dir=None):
    """Adaptación de dominio (si no existe `dapt_dir`) y fine-tuning. Escribe m4_<split>_preds.csv."""
    seed = cfg["seed"] if seed is None else seed
    dapt_dir = Path(dapt_dir or path_of(cfg, "artifacts") / "m4_dapt")
    out_dir = Path(out_dir or path_of(cfg, "results"))
    model_id = tag or (MODEL_ID if seed == cfg["seed"] else f"{MODEL_ID}_s{seed}")

    dapt_info = None
    if not (dapt_dir / "config.json").exists():
        texts = corpus_texts(cfg, cfg["models"]["m4"]["include_train_texts"], data_dir, external_dir)
        dapt_info = dapt(cfg, texts, dapt_dir, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{MODEL_ID}_dapt_run.json").write_text(json.dumps(dapt_info, indent=1), encoding="utf-8")
    elif (out_dir / f"{MODEL_ID}_dapt_run.json").exists():
        dapt_info = json.loads((out_dir / f"{MODEL_ID}_dapt_run.json").read_text(encoding="utf-8"))

    path = m3_bert_finetune.run(cfg, eval_split, final=final, seed=seed, train_df=train_df, data_dir=data_dir,
                                out_dir=out_dir, overrides={**(overrides or {}), "checkpoint": str(dapt_dir)},
                                tag=model_id, plot=plot, save=save)
    info_path = out_dir / f"{model_id}_{eval_split}_run.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if dapt_info:
        info["dapt"] = {k: v for k, v in dapt_info.items() if k != "history"}
        info["total_train_seconds"] = round(info["train_seconds"] + dapt_info["seconds"], 2)
    info_path.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    print(run(load_config(), args.split, final=args.final, seed=args.seed, save=args.save))
