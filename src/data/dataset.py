"""Carga de splits con validación de esquema y guarda del test cerrado."""
import pandas as pd

from src.config import path_of
from src.data.schema import validate


def load_split(cfg, name, final=False, data_dir=None):
    """Lee data/processed/<name>.csv y valida el contrato.

    `test` solo se puede leer con final=True (evaluación final con la configuración congelada).
    """
    if name == "test" and not final:
        raise PermissionError("test está cerrado: solo se usa en la evaluación final (final=True)")
    base = data_dir or path_of(cfg, "processed")
    df = pd.read_csv(f"{base}/{name}.csv", encoding="utf-8")
    errors = validate(df, cfg["areas"])
    if errors:
        raise ValueError(f"{name}.csv no cumple el contrato: {errors}")
    return df
