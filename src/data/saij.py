"""Datos externos: jurisprudencia argentina (SAIJ) etiquetada por materia, como complemento de entrenamiento.

Fuente: https://huggingface.co/datasets/marianbasti/jurisprudencia-Argentina-SAIJ (licencia Apache-2.0).
Cada registro trae `materia` (área del derecho) y `texto` (resumen de la doctrina del fallo).
Limitaciones que hay que declarar en el informe: es de Argentina y no de Guatemala, son resúmenes de
fallos y no consultas de clientes, la materia es una etiqueta de un analista y no del bufete, y la
etiqueta `familia` es débil: SAIJ no la tiene y sale de palabras clave dentro de CIVIL.

Dos etapas, para poder cambiar el mapeo de materias sin volver a descargar 2,5 GB:
  1) python -m src.data.saij --download            baja en streaming y guarda solo materia + texto
                                                   (data/external/saij_filtrado.csv.gz y saij_resumen.json)
  2) python -m src.data.saij --build --per-area 3000   mapea a las áreas del proyecto y arma
                                                   data/external/saij_por_area.csv (contrato de datos + origen)
Los datos externos nunca van a val.csv ni test.csv: solo a entrenamiento.
"""
import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from src.config import ROOT, load_config
from src.data.clean import normalize_line, strip_accents

URL = "https://huggingface.co/datasets/marianbasti/jurisprudencia-Argentina-SAIJ/resolve/main/dataset.jsonl"
EXTERNAL = ROOT / "data" / "external"
FIELDS = ["id", "materia", "texto", "anio", "tipo_tribunal"]

# materia (normalizada, sin tildes y separada por '-') -> área del proyecto
_OTROS = {"ADMINISTRATIVO", "CONSTITUCIONAL", "SEGURIDAD SOCIAL", "TRIBUTARIO", "FISCAL", "IMPOSITIVO", "ADUANERO",
          "AMBIENTAL", "DERECHOS HUMANOS", "ELECTORAL", "MINERO", "NAVEGACION", "MILITAR", "DERECHO PUBLICO",
          "INTERNACIONAL", "AGRARIO", "ECONOMICO", "TRANSITO", "PREVISIONAL"}


# SAIJ no tiene materia "familia" (en Argentina el derecho de familia está dentro de CIVIL): los textos de
# CIVIL que mencionan estas figuras se reclasifican como familia. Es una etiqueta débil por palabras clave.
_FAMILIA = re.compile(r"\b(divorcio|cuota alimentaria|alimentos provisorios|obligacion alimentaria|patria potestad|"
                      r"tenencia de (?:los )?(?:hijos|menores?)|regimen de visitas|filiacion|adopcion|union convivencial|"
                      r"sociedad conyugal|nulidad de matrimonio|violencia familiar|bien de familia|"
                      r"interes superior del nino|cuidado personal del hijo)\b")


def normalize_materia(m):
    s = strip_accents(str(m)).upper()
    s = re.sub(r"\s*[-–/]\s*", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def map_materia(materia):
    """Área del proyecto para una materia de SAIJ, o None si es ambigua o no tiene equivalente claro.

    Se excluyen a propósito las materias ambiguas: 'CIVIL-COMERCIAL' (mezcla dos áreas) y 'PROCESAL'
    a secas (derecho procesal genérico sin área).
    """
    norm = normalize_materia(materia)
    parts = set(norm.split("-"))            # 'SEGURIDAD SOCIAL' queda como una sola parte
    words = set(re.split(r"[-\s]+", norm))   # 'DERECHO DEL TRABAJO' -> DERECHO, DEL, TRABAJO
    if words & {"LABORAL", "TRABAJO"}:
        return "laboral"
    if "PENAL" in words:
        return "penal"
    if "FAMILIA" in words:
        return "familia"
    if {"CIVIL", "COMERCIAL"} <= words:
        return None
    if "COMERCIAL" in words:
        return "mercantil"
    if "CIVIL" in words:
        return "civil"
    if parts <= _OTROS and parts:
        return "otros"
    return None


def _text_key(texto):
    return hashlib.blake2b(normalize_line(texto).encode(), digest_size=8).digest()


def filter_records(records, min_words=15, seen=None):
    """Registros con materia y texto suficientemente largo, sin textos repetidos. Devuelve filas FIELDS."""
    seen = set() if seen is None else seen
    for d in records:
        materia, texto = d.get("materia"), d.get("texto")
        if not isinstance(materia, str) or not isinstance(texto, str) or not materia.strip():
            continue
        texto = re.sub(r"\s+", " ", texto).strip()
        if len(texto.split()) < min_words:
            continue
        key = _text_key(texto)
        if key in seen:
            continue
        seen.add(key)
        yield {"id": d.get("numero-sumario") or d.get("guid") or d.get("id-infojus") or "",
               "materia": materia.strip(), "texto": texto, "anio": str(d.get("fecha") or "")[:4],
               "tipo_tribunal": d.get("tipo-tribunal") or ""}


def stream_records(url=URL, retries=8, chunk=1 << 20, log=sys.stderr):
    """Registros del JSONL descargados en streaming; si se corta la conexión, reanuda donde quedó."""
    import requests

    offset, done, attempts = 0, False, 0
    last_log = 0
    while not done:
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                buf = b""
                for part in r.iter_content(chunk):
                    buf += part
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        offset += len(line) + 1
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                pass
                    if offset - last_log >= 200 << 20:
                        last_log = offset
                        print(f"  {offset / 1e6:,.0f} MB leídos", file=log, flush=True)
                if buf.strip():
                    try:
                        yield json.loads(buf)
                    except json.JSONDecodeError:
                        pass
                done = True
        except (requests.RequestException, OSError) as e:
            attempts += 1
            if attempts > retries:
                raise
            print(f"  conexión interrumpida ({type(e).__name__}); reanudando en {offset / 1e6:,.0f} MB", file=log, flush=True)
            time.sleep(min(30, 2 ** attempts))


def download(out_dir=EXTERNAL, url=URL, min_words=15, records=None):
    """Etapa 1: baja, filtra y guarda materia + texto. Devuelve el resumen."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen, materias, years, lens = set(), Counter(), Counter(), []
    n_in = 0

    def counted(it):
        nonlocal n_in
        for d in it:
            n_in += 1
            yield d

    with gzip.open(out / "saij_filtrado.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in filter_records(counted(records if records is not None else stream_records(url)), min_words, seen):
            w.writerow(row)
            materias[row["materia"]] += 1
            years[row["anio"]] += 1
            lens.append(len(row["texto"].split()))
    lens.sort()
    summary = {"registros_leidos": n_in, "registros_guardados": len(lens),
               "palabras_mediana": lens[len(lens) // 2] if lens else 0,
               "palabras_p10": lens[len(lens) // 10] if lens else 0, "palabras_p90": lens[9 * len(lens) // 10] if lens else 0,
               "materias": materias.most_common(), "anios_top": years.most_common(8)}
    (out / "saij_resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def build_by_area(out_dir=EXTERNAL, per_area=3000, seed=42, areas=None, min_words=20, max_words=400):
    """Etapa 2: mapea materias a las áreas del proyecto y deja a lo sumo `per_area` textos por área."""
    import random

    areas = areas or load_config()["areas"]
    out = Path(out_dir)
    by_area = {a: [] for a in areas}
    skipped = Counter()
    with gzip.open(out / "saij_filtrado.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            area = map_materia(row["materia"])
            etiqueta = "materia"
            if area == "civil" and _FAMILIA.search(strip_accents(row["texto"]).lower()):
                area, etiqueta = "familia", "palabras_clave"
            row["etiqueta"] = etiqueta
            n = len(row["texto"].split())
            if area not in by_area:
                skipped[normalize_materia(row["materia"])] += 1
            elif min_words <= n <= max_words:
                by_area[area].append(row)
    rng = random.Random(seed)
    rows = []
    for area in areas:
        pool = by_area[area]
        rng.shuffle(pool)
        for k, r in enumerate(pool[:per_area], 1):
            rows.append({"case_id": f"J-{area[:3]}-{k:05d}", "texto": r["texto"], "area_legal": area,
                         "canal": "documento", "fecha": "", "fuente_texto": "nativo", "grupo_id": "saij",
                         "origen": "saij", "materia_original": r["materia"], "etiqueta": r["etiqueta"]})
    with open(out / "saij_por_area.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"disponibles": {a: len(by_area[a]) for a in areas}, "guardados": dict(Counter(r["area_legal"] for r in rows)),
            "materias_sin_mapear": skipped.most_common(25)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--per-area", type=int, default=3000)
    a = ap.parse_args()
    if a.download:
        print(json.dumps(download(), ensure_ascii=False, indent=1))
    if a.build:
        print(json.dumps(build_by_area(per_area=a.per_area), ensure_ascii=False, indent=1))
