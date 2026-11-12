"""Predictor de inferencia: texto -> (clase, confianza, probabilidades). Lo usan la API y el dashboard."""
import time

import torch

from src.config import load_config, path_of
from src.models.m2_bert_frozen import pick_device


def load_temperature(cfg, model_id):
    """T guardada junto a los pesos (artifacts/<modelo>_temperature.json); 1.0 si no existe."""
    import json
    f = path_of(cfg, "artifacts") / f"{model_id}_temperature.json"
    return float(json.loads(f.read_text(encoding="utf-8"))["temperature"]) if f.exists() else 1.0


class Predictor:
    """Envuelve un clasificador (BertClassifier de M3/M4) con su tokenizador y la lista de áreas."""

    def __init__(self, model, tok, areas, max_len=256, device="cpu", temperature=1.0):
        self.model, self.tok, self.areas, self.max_len, self.device = model.eval(), tok, list(areas), max_len, device
        self.temperature = float(temperature)  # T de calibración (1.0 = sin calibrar); ver src/evaluation/calibration.py

    @classmethod
    def from_config(cls, cfg=None, model_id=None):
        cfg = cfg or load_config()
        model_id = model_id or cfg["serving"]["model"]
        device = pick_device()
        from src.models.m3_bert_finetune import load_finetuned
        checkpoint = path_of(cfg, "artifacts") / "m4_dapt" if model_id.startswith("m4") else None
        tok, model = load_finetuned(cfg, model_id, checkpoint=checkpoint, device=device)
        return cls(model, tok, cfg["areas"], cfg["models"]["m3"]["max_len"], device, load_temperature(cfg, model_id))

    def predict_proba(self, texts):
        """Lista de textos -> tensor (n, n_áreas) de probabilidades."""
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"]).float() / self.temperature
            return torch.softmax(logits, 1).cpu()

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
