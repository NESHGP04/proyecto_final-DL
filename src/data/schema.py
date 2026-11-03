"""Contrato de datos procesados (docs/CONTEXTO.md 7.1)."""
import pandas as pd

COLUMNS = ["case_id", "texto", "area_legal", "canal", "fecha", "fuente_texto"]
CANALES = {"correo", "formulario", "documento", "fisico"}
FUENTES = {"nativo", "ocr"}


def validate(df, areas):
    """Devuelve una lista de errores (vacía si el DataFrame cumple el contrato)."""
    errors = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"faltan columnas: {missing}"]

    if df["case_id"].duplicated().any():
        errors.append("case_id duplicados")
    if df["texto"].isna().any() or (df["texto"].astype(str).str.strip() == "").any():
        errors.append("hay textos vacíos")
    bad_area = set(df["area_legal"].dropna()) - set(areas)
    if bad_area or df["area_legal"].isna().any():
        errors.append(f"area_legal fuera de la lista o vacía: {sorted(map(str, bad_area))}")
    if not set(df["canal"].dropna()) <= CANALES:
        errors.append("canal fuera de {correo, formulario, documento, fisico}")
    if not set(df["fuente_texto"].dropna()) <= FUENTES:
        errors.append("fuente_texto fuera de {nativo, ocr}")
    fechas = df["fecha"].dropna()
    if len(fechas) and pd.to_datetime(fechas, format="%Y-%m-%d", errors="coerce").isna().any():
        errors.append("fecha con formato distinto de YYYY-MM-DD")
    return errors
