"""Segmentación de escritos judiciales: quedarse con la narración de los hechos y trocearla.

Los escritos de demanda tienen membrete (tipo de juicio y juzgado), datos de las partes, hechos,
fundamentos de derecho, pruebas y petición. Solo los hechos se parecen a lo que un cliente cuenta en
su primer contacto, y el membrete y los fundamentos nombran el área, lo que delataría la etiqueta.
"""
import re

from src.data.clean import normalize_line

# nombre de sección -> títulos (ya normalizados: sin tildes ni signos) que la abren
SECTIONS = {
    "exposicion": {"expongo", "exposicion", "exposicion de hechos", "antecedentes"},
    "hechos": {"hechos", "los hechos", "hechos que fundamentan la demanda", "relacion de hechos"},
    "derecho": {"derecho", "fundamento de derecho", "fundamentos de derecho", "fundamentacion de derecho",
                "fundamento legal", "fundamentos legales"},
    "pruebas": {"prueba", "pruebas", "medios de prueba", "ofrecimiento de prueba", "prueba documental"},
    "peticion": {"peticion", "peticiones", "petitorio", "peticion de tramite", "petitoria"},
    "notificaciones": {"lugar para recibir notificaciones", "notificaciones", "lugar para notificaciones"},
}
_TITLE = {t: name for name, titles in SECTIONS.items() for t in titles}


def heading_of(line):
    """Nombre de sección si la línea es (o empieza con) un título conocido seguido de ':' o sola."""
    head, sep, rest = line.partition(":")
    norm = normalize_line(head)
    if norm in _TITLE and (sep or len(norm.split()) <= 5) and (not sep or len(head) <= 60):
        return _TITLE[norm], rest.strip()
    return None, line


def find_sections(lines):
    """Lista de (índice de línea, sección) en orden."""
    return [(i, name) for i, line in enumerate(lines) if (name := heading_of(line)[0])]


def extract_section(lines, name="hechos"):
    """Líneas de la sección `name` (sin el título) o None si el documento no la tiene."""
    marks = find_sections(lines)
    for k, (i, sec) in enumerate(marks):
        if sec != name:
            continue
        end = next((j for j, other in marks[k + 1:] if other != name), len(lines))
        first = heading_of(lines[i])[1]
        return ([first] if first else []) + lines[i + 1:end]
    return None


def chunk_words(text, target=150, min_last=40):
    """Trocea en fragmentos de ~`target` palabras cortando al final de una oración."""
    sentences = re.split(r"(?<=[.;])\s+", re.sub(r"\s+", " ", text).strip())
    chunks, cur, n = [], [], 0
    for s in sentences:
        cur.append(s)
        n += len(s.split())
        if n >= target:
            chunks.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        tail = " ".join(cur)
        if chunks and n < min_last:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)
    return chunks
