"""API de inferencia (FastAPI).

POST /classify {texto} -> {area_legal, confianza, accion: "enrutar" | "revision"} según τ
(configs/config.yaml: confidence_threshold). Medir latencia, throughput y costo por predicción.
"""
