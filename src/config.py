"""Carga de configuración compartida (configs/config.yaml y configs/value_params.yaml)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
