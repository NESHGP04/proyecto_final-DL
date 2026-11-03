"""API de inferencia (FastAPI). Dueño: Nes (CP5).

POST /classify {texto} -> {area_legal, confianza, accion: "enrutar" | "revision"} según τ
(configs/config.yaml: confidence_threshold). Medir latencia, throughput y costo por predicción.
"""
