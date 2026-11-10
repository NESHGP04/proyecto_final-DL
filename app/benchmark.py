"""Latencia, throughput y costo por predicción de un Predictor.

Uso: python -m app.benchmark --texts-file ruta.txt [--cost-per-hour 0.50]
El costo por hora de cómputo NO se inventa: se pasa con --cost-per-hour (precio de la instancia o
estimación de la máquina) y, si no se da, solo se reportan tiempos.
"""
import argparse
import statistics
import time


def benchmark(classify, texts, n_single=30, batch_size=16, cost_per_hour=None, warmup=3):
    """classify: callable(list[str]) -> lista. Mide 1 texto por llamada y por lotes."""
    for t in texts[:warmup]:
        classify([t])
    single = []
    for i in range(n_single):
        t0 = time.perf_counter()
        classify([texts[i % len(texts)]])
        single.append(1000 * (time.perf_counter() - t0))
    batch = [texts[i % len(texts)] for i in range(batch_size)]
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        classify(batch)
    per_text_batch = 1000 * (time.perf_counter() - t0) / (reps * batch_size)
    single.sort()
    res = {
        "latencia_ms_p50": statistics.median(single),
        "latencia_ms_p95": single[int(0.95 * (len(single) - 1))],
        "ms_por_texto_en_lote": per_text_batch,
        "throughput_textos_por_s_lote": 1000 / per_text_batch,
        "predicciones_por_hora_lote": 3.6e6 / per_text_batch,
    }
    if cost_per_hour is not None:
        res["costo_por_1000_predicciones"] = cost_per_hour / res["predicciones_por_hora_lote"] * 1000
    return res


if __name__ == "__main__":
    from app.predictor import Predictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--texts-file", required=True, help="un texto por línea (usar datos de prueba, no reales en logs)")
    ap.add_argument("--cost-per-hour", type=float, default=None)
    args = ap.parse_args()
    texts = [t for t in open(args.texts_file, encoding="utf-8").read().splitlines() if t.strip()]
    for k, v in benchmark(Predictor.from_config().classify, texts, cost_per_hour=args.cost_per_hour).items():
        print(f"{k}: {v:.3f}")
