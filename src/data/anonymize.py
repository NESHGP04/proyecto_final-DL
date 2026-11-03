"""Anonimización de texto. Dueña: Cami (CP1).

Ya trae regex para correo, DPI, NIT y teléfono de Guatemala, y reemplazo de nombres a partir de
una lista. Falta (Cami): direcciones y nombres sin lista (p. ej. NER en español), y revisar a mano
una muestra para confirmar que no queda nada sensible. Los patrones son heurísticos: no
garantizan anonimización completa.
"""
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DPI = re.compile(r"\b\d{4}\s?\d{5}\s?\d{4}\b")           # 13 dígitos, p. ej. 1234 56789 0101
_NIT = re.compile(r"\b\d{4,9}-[\dkK]\b")                  # p. ej. 1234567-8
_TEL = re.compile(r"(?:\+?502[\s-]?)?\b[2-7]\d{3}[\s-]?\d{4}\b")  # 8 dígitos, prefijo opcional

# El orden importa: DPI y NIT antes que teléfono para que no se corten en pedazos.
_PATTERNS = [(_EMAIL, "[CORREO]"), (_DPI, "[DPI]"), (_NIT, "[NIT]"), (_TEL, "[TELEFONO]")]


def mask_patterns(text):
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return text


def mask_names(text, names):
    """Reemplaza cada nombre de `names` (sin distinguir mayúsculas) por [PERSONA]."""
    for name in sorted(names, key=len, reverse=True):
        if name.strip():
            text = re.sub(re.escape(name), "[PERSONA]", text, flags=re.IGNORECASE)
    return text


def anonymize(text, names=()):
    return mask_patterns(mask_names(text, names))
