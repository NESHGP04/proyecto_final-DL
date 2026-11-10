"""Predictor de inferencia: texto -> (clase, confianza, probabilidades). Lo usan la API y el dashboard."""
import time

import torch

from src.config import load_config, path_of
from src.models.m2_bert_frozen import pick_device


class Predictor:
    """Envuelve un clasificador (BertClassifier de M3/M4) con su tokenizador y la lista de áreas."""

    def __init__(self, model, tok, areas, max_len=256, device="cpu"):
        self.model, self.tok, self.areas, self.max_len, self.device = model.eval(), tok, list(areas), max_len, device

    @classmethod
    def from_config(cls, cfg=None, model_id=None):
        cfg = cfg or load_config()
        model_id = model_id or cfg["serving"]["model"]
        device = pick_device()
        from src.models.m3_bert_finetune import load_finetuned
        checkpoint = path_of(cfg, "artifacts") / "m4_dapt" if model_id.startswith("m4") else None
        tok, model = load_finetuned(cfg, model_id, checkpoint=checkpoint, device=device)
        return cls(model, tok, cfg["areas"], cfg["models"]["m3"]["max_len"], device)

    def predict_proba(self, texts):
        """Lista de textos -> tensor (n, n_áreas) de probabilidades."""
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(enc["input_ids"], enc["attention_mask"]).float(), 1).cpu()

    def classify(self, texts):
        """Devuelve una lista de dicts {area_legal, confianza, probabilidades} por texto."""
        probs = self.predict_proba(texts)
        conf, idx = probs.max(1)
        return [{"area_legal": self.areas[i], "confianza": float(c),
                 "probabilidades": {a: float(p) for a, p in zip(self.areas, row)}}
                for i, c, row in zip(idx.tolist(), conf.tolist(), probs)]


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, 1000 * (time.perf_counter() - start)
