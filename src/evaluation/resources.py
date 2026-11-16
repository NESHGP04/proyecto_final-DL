"""Medidas de recursos de la máquina donde corre un modelo (para la tabla de costos).

El pico de memoria es el máximo del proceso entero (datos, librerías y modelo), no solo del modelo.
"""
import platform
import resource
import sys


def peak_rss_mb():
    """Pico de memoria residente del proceso, en MB (macOS lo da en bytes y Linux en KB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024)


def machine_info():
    """Arquitectura y sistema. Ojo: un Python x86_64 en un Mac con Apple Silicon corre bajo Rosetta."""
    return {"machine": platform.machine(), "system": platform.system()}
