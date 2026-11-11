"""Limpieza del texto de OCR: ruido de sellos y bordes, folios, encabezados repetidos y guiones.

Trabaja sobre listas de líneas por página (la salida de src/data/ocr.py) y devuelve las líneas del
documento ya limpias, para que la segmentación pueda detectar títulos de sección por línea.
"""
import re
import unicodedata
from collections import Counter

_FOLIO = re.compile(r"(?:p[aá]g(?:ina)?\.?\s*)?\d{1,3}(?:\s*(?:de|/)\s*\d{1,3})?", re.I)


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_line(s):
    """Forma comparable de una línea: sin tildes, minúsculas y solo letras y dígitos."""
    return re.sub(r"[^a-z0-9]+", " ", strip_accents(s).lower()).strip()


def clean_page_lines(text, min_alpha_share=0.5):
    """Líneas de una página sin ruido: muy cortas, solo número de folio o casi sin letras."""
    out = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw)).strip()
        if sum(c.isalnum() for c in line) < 3 or _FOLIO.fullmatch(line):
            continue
        if sum(c.isalpha() for c in line) / len(line) < min_alpha_share:
            continue
        out.append(line)
    return out


def remove_repeated_lines(pages_lines, min_share=0.5, min_pages=4):
    """Quita encabezados y pies que se repiten en al menos `min_share` de las páginas."""
    if len(pages_lines) < min_pages:
        return pages_lines
    seen = Counter(n for page in pages_lines for n in {normalize_line(l) for l in page} if n)
    limit = min_share * len(pages_lines)
    return [[l for l in page if seen[normalize_line(l)] < limit] for page in pages_lines]


def join_hyphenation(lines):
    """Une palabras cortadas con guion al final de línea ('indemniza-' + 'ción')."""
    out = []
    for line in lines:
        if out and out[-1].endswith("-") and line[:1].islower():
            out[-1] = out[-1][:-1] + line
        else:
            out.append(line)
    return out


def clean_document(pages):
    """pages: lista de textos de página en orden. Devuelve las líneas limpias del documento."""
    pages_lines = remove_repeated_lines([clean_page_lines(p) for p in pages])
    return join_hyphenation([l for page in pages_lines for l in page])
