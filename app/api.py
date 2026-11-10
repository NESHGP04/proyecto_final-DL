"""API de inferencia (FastAPI).

POST /classify        {"texto": "..."} -> área, confianza y acción ("enrutar" si confianza >= τ,
                      "revision" si no), con la latencia de la petición.
POST /classify_batch  {"textos": [...]} -> lo mismo para varios textos (mejor throughput).
GET  /health          estado y modelo cargado.
GET  /stats           conteos y latencias desde que arrancó (insumo del plan de monitoreo).

No se guarda el texto de los casos: el registro de monitoreo (artifacts/api_log.csv) solo tiene
hora, área, confianza, acción y latencia. τ sale de configs/config.yaml (confidence_threshold).

Arranque: uvicorn app.api:app   (requiere pesos guardados con --save en artifacts/)
"""
import csv
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.predictor import Predictor, timed
from src.config import load_config, path_of

LOG_FIELDS = ["timestamp", "area_legal", "confianza", "accion", "latencia_ms"]


class ClassifyIn(BaseModel):
    texto: str


class BatchIn(BaseModel):
    textos: list[str]


def create_app(predictor, threshold=0.70, max_chars=20000, log_path=None, model_name="m3"):
    app = FastAPI(title="Clasificador de casos legales")
    state = {"lock": threading.Lock(), "latencias": [], "acciones": {"enrutar": 0, "revision": 0}}

    def _log(rows):
        if log_path is None:
            return
        path = Path(log_path)
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_FIELDS)
            w.writerows(rows)

    def _validate(text):
        if not text.strip():
            raise HTTPException(422, "el texto está vacío")
        if len(text) > max_chars:
            raise HTTPException(413, f"el texto supera {max_chars} caracteres")

    def _respond(texts):
        for t in texts:
            _validate(t)
        results, ms = timed(predictor.classify, texts)
        out, rows = [], []
        for r in results:
            accion = "enrutar" if r["confianza"] >= threshold else "revision"
            out.append({**r, "accion": accion, "umbral": threshold})
            rows.append([datetime.now(timezone.utc).isoformat(timespec="seconds"), r["area_legal"],
                         round(r["confianza"], 4), accion, round(ms / len(texts), 2)])
        with state["lock"]:
            state["latencias"].extend([ms / len(texts)] * len(texts))
            for o in out:
                state["acciones"][o["accion"]] += 1
            _log(rows)
        return out, ms

    @app.get("/health")
    def health():
        return {"estado": "ok", "modelo": model_name, "umbral": threshold}

    @app.post("/classify")
    def classify(body: ClassifyIn):
        out, ms = _respond([body.texto])
        return {**out[0], "latencia_ms": round(ms, 2)}

    @app.post("/classify_batch")
    def classify_batch(body: BatchIn):
        if not body.textos:
            raise HTTPException(422, "no hay textos")
        out, ms = _respond(body.textos)
        return {"resultados": out, "latencia_ms_total": round(ms, 2)}

    @app.get("/stats")
    def stats():
        lat = sorted(state["latencias"])
        total = sum(state["acciones"].values())
        return {"predicciones": total, **state["acciones"],
                "tasa_revision": state["acciones"]["revision"] / total if total else None,
                "latencia_ms_p50": statistics.median(lat) if lat else None,
                "latencia_ms_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None, "desde": time.time()}

    return app


def _default_app():
    cfg = load_config()
    return create_app(Predictor.from_config(cfg), cfg["confidence_threshold"], cfg["serving"]["max_text_chars"],
                      path_of(cfg, "artifacts") / "api_log.csv", cfg["serving"]["model"])


def __getattr__(name):  # `uvicorn app.api:app` carga el modelo solo al arrancar, no al importar el módulo
    if name == "app":
        return _default_app()
    raise AttributeError(name)
