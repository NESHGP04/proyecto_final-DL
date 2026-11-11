"""Carga de configuración compartida (configs/config.yaml y configs/value_params.yaml)."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    """Lee config.yaml. Las rutas de `paths` se pueden cambiar con variables de entorno DL_PATH_<CLAVE>
    (p. ej. DL_PATH_RESULTS=/content/drive/MyDrive/proyecto/results), útil en Colab para que los
    resultados y los checkpoints vivan en Google Drive y sobrevivan a una desconexión."""
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in cfg.get("paths", {}):
        override = os.environ.get(f"DL_PATH_{key.upper()}")
        if override:
            cfg["paths"][key] = override  # una ruta absoluta reemplaza a ROOT en path_of
    return cfg


def load_value_params(path=None):
    """Devuelve el yaml completo: {nombre: {value, min, max, fuente, unidad}}."""
    path = Path(path) if path else ROOT / "configs" / "value_params.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def param_values(params):
    """Aplana {nombre: {value, ...}} a {nombre: value}."""
    return {k: float(v["value"]) for k, v in params.items()}


def path_of(cfg, key):
    return ROOT / cfg["paths"][key]
