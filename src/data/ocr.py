"""OCR local de casos físicos escaneados (PDF o imagen) con Tesseract en español.

Todo corre en la máquina: ningún texto ni imagen sale a servicios externos. Cada página se
renderiza a 300 dpi (eso corrige escaneos con resolución vertical menor que la horizontal), se
preprocesa, se reconoce, y se guarda el texto por página junto con estadísticas de confianza que no
contienen texto (aptas para revisar o citar).

Uso:
  python -m src.data.ocr --compare --pages 6      # compara variantes de preprocesamiento por confianza
  python -m src.data.ocr --workers 4   # modo gray: en las pruebas, deskew y binarizar no mejoraron la confianza
Requiere el idioma español en artifacts/tessdata/spa.traineddata (tessdata_best de tesseract-ocr).
"""
import argparse
import csv
import os
import re
import shlex
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import ROOT

TESSDATA = ROOT / "artifacts" / "tessdata"
MODES = ("gray", "deskew", "otsu", "sauvola")
# El orden importa: "RESOLUCIÓN QUE ADMITE AMPLIACIÓN DE DEMANDA" es una resolución, no una demanda.
TIPOS = [("resoluci", "resolucion"), ("ampliaci", "ampliacion"), ("demanda", "demanda"), ("car", "caratula"),
         ("desistimiento", "desistimiento"), ("pliego", "pliego_posiciones"), ("subsanaci", "subsanacion"),
         ("acuerdo", "acuerdo")]


# ---------------------------------------------------------------- imagen
def render_page(pdf, page, dpi=300):
    """Página `page` (1 = primera) del PDF como imagen en gris."""
    from pdf2image import convert_from_path
    return convert_from_path(str(pdf), dpi=dpi, first_page=page, last_page=page, grayscale=True)[0].convert("L")


def otsu_threshold(arr):
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    w0 = np.cumsum(hist)
    m = np.cumsum(hist * np.arange(256))
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m[-1] * w0 - m * total) ** 2 / (w0 * w1)
    ties = np.flatnonzero(between == np.nanmax(between))  # imagen casi binaria: varios umbrales empatan
    return int(ties.mean())


def estimate_skew(img, max_angle=3.0, step=0.25):
    """Ángulo (grados) que hay que rotar la imagen para enderezar las líneas de texto.

    Prueba rotaciones sobre una versión reducida y elige la que hace más contrastado el perfil de
    filas (líneas de texto bien alineadas = filas muy oscuras y muy claras).
    """
    small = img.copy()
    small.thumbnail((700, 1100))
    arr = np.asarray(small)
    ink = Image.fromarray(((arr < otsu_threshold(arr)) * 255).astype(np.uint8))

    def score(angle):
        rows = np.asarray(ink.rotate(angle, resample=Image.BILINEAR, fillcolor=0)).sum(axis=1).astype(np.float64)
        return rows.var()

    coarse = np.arange(-max_angle, max_angle + 1e-9, step)
    best = max(coarse, key=score)
    fine = np.arange(best - step, best + step + 1e-9, step / 5)
    return float(max(fine, key=score))


def deskew(img):
    angle = estimate_skew(img)
    return img if abs(angle) < 0.05 else img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)


def sauvola(arr, window=51, k=0.2, r=128.0):
    """Binarización local (Sauvola) con imágenes integrales: tolera sombras y fondo desigual."""
    a = arr.astype(np.float64)
    half = window // 2
    pad = np.pad(a, half + 1, mode="edge")
    ii = pad.cumsum(0).cumsum(1)
    ii2 = (pad ** 2).cumsum(0).cumsum(1)
    h, w = a.shape

    def box(s):
        return (s[window:window + h, window:window + w] - s[0:h, window:window + w]
                - s[window:window + h, 0:w] + s[0:h, 0:w]) / (window * window)

    mean = box(ii)
    std = np.sqrt(np.maximum(box(ii2) - mean ** 2, 0))
    return a > mean * (1 + k * (std / r - 1))


def preprocess(img, mode="sauvola"):
    if mode not in MODES:
        raise ValueError(f"modo desconocido: {mode}")
    if mode == "gray":
        return img
    img = deskew(img)
    if mode == "deskew":
        return img
    arr = np.asarray(img)
    if mode == "otsu":
        return Image.fromarray(((arr > otsu_threshold(arr)) * 255).astype(np.uint8))
    return Image.fromarray((sauvola(arr) * 255).astype(np.uint8))


# ---------------------------------------------------------------- reconocimiento
def ocr_image(img, tessdata_dir=TESSDATA, psm=4, lang="spa"):
    """Reconoce una imagen. Devuelve {text, n_words, mean_conf, low_conf_share} (conf 0-100)."""
    import pytesseract

    cfg = f"--tessdata-dir {shlex.quote(str(tessdata_dir))} --psm {psm}"
    d = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    lines, confs, key, cur = [], [], None, []
    for i, word in enumerate(d["text"]):
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if k != key and cur:
            lines.append(" ".join(cur))
            cur = []
        key = k
        conf = float(d["conf"][i])
        if word.strip() and conf >= 0:
            cur.append(word.strip())
            confs.append(conf)
    if cur:
        lines.append(" ".join(cur))
    return {"text": "\n".join(lines), "n_words": len(confs),
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "low_conf_share": float(np.mean(np.array(confs) < 60)) if confs else 1.0}


def tipo_documento(filename):
    """Tipo genérico del documento a partir de palabras del nombre (sin conservar el nombre)."""
    base = re.sub(r"[^a-z ]", "", filename.lower().replace("á", "a").replace("é", "e").replace("ó", "o"))
    for key, tipo in TIPOS:
        if key in base:
            return tipo
    return "otro"


def _page_task(args):
    pdf, page, doc_id, out_dir, mode, dpi, psm, tessdata = args
    os.environ["OMP_THREAD_LIMIT"] = "1"
    res = ocr_image(preprocess(render_page(pdf, page, dpi), mode), tessdata, psm)
    (Path(out_dir) / f"{doc_id}_p{page:02d}.txt").write_text(res.pop("text"), encoding="utf-8")
    return {"doc_id": doc_id, "page": page, **res}


def page_count(pdf):
    import subprocess
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"^Pages:\s+(\d+)", out, re.M).group(1))


def ocr_folder(raw_dir, out_dir, mode="gray", dpi=300, psm=4, workers=4, tessdata=TESSDATA):
    """OCR de todos los PDF de raw_dir. Escribe <doc_id>_p<N>.txt, inventario.csv y paginas.csv.

    doc_id = D01, D02… por orden alfabético de archivo: los nombres de archivo pueden contener
    nombres de personas y no se copian a ninguna salida.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv = write_inventory(raw_dir, out)
    pdfs = sorted(Path(raw_dir).glob("*.pdf"))
    tasks = [(pdf, p, row["doc_id"], out, mode, dpi, psm, tessdata)
             for pdf, row in zip(pdfs, inv) for p in range(1, row["paginas"] + 1)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        stats = list(ex.map(_page_task, tasks))
    _write_csv(out / "paginas.csv", stats)
    return stats


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_inventory(raw_dir, out_dir):
    """inventario.csv: doc_id (D01… por orden alfabético de archivo), tipo genérico y páginas."""
    pdfs = sorted(Path(raw_dir).glob("*.pdf"))
    inv = [{"doc_id": f"D{i:02d}", "tipo_documento": tipo_documento(pdf.name), "paginas": page_count(pdf)}
           for i, pdf in enumerate(pdfs, 1)]
    _write_csv(Path(out_dir) / "inventario.csv", inv)
    return inv


# ---------------------------------------------------------------- calidad
def character_error_rate(reference, hypothesis):
    """CER = distancia de edición entre caracteres / longitud de la referencia."""
    ref, hyp = list(reference), list(hypothesis)
    if not ref:
        return float(len(hyp) > 0)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def ocr_file(path, mode="gray", dpi=300, psm=4, tessdata=TESSDATA):
    """Texto de un PDF o imagen (todas las páginas unidas)."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        pages = [render_page(path, p, dpi) for p in range(1, page_count(path) + 1)]
    else:
        pages = [Image.open(path).convert("L")]
    return "\n\n".join(ocr_image(preprocess(p, mode), tessdata, psm)["text"] for p in pages)


def compare_modes(raw_dir, n_pages=6, dpi=300, psm=4, tessdata=TESSDATA):
    """Compara variantes de preprocesamiento en páginas repartidas entre los documentos.

    Se decide por la confianza media de Tesseract y el número de palabras, sin leer el texto.
    """
    pdfs = sorted(Path(raw_dir).glob("*.pdf"))
    sample = [(pdfs[i % len(pdfs)], 1 + (i // len(pdfs))) for i in range(n_pages)]
    rows = []
    for mode in MODES:
        for pdf, page in sample:
            r = ocr_image(preprocess(render_page(pdf, page, dpi), mode), tessdata, psm)
            rows.append({"mode": mode, "n_words": r["n_words"], "mean_conf": r["mean_conf"], "low": r["low_conf_share"]})
    summary = {}
    for mode in MODES:
        g = [r for r in rows if r["mode"] == mode]
        summary[mode] = {"palabras_por_pagina": float(np.mean([r["n_words"] for r in g])),
                         "conf_media": float(np.mean([r["mean_conf"] for r in g])),
                         "palabras_baja_conf": float(np.mean([r["low"] for r in g]))}
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "interim" / "ocr"))
    ap.add_argument("--mode", default="gray", choices=MODES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--pages", type=int, default=6)
    a = ap.parse_args()
    if a.compare:
        for mode, s in compare_modes(a.input, a.pages).items():
            print(f"{mode:8s} palabras/pág {s['palabras_por_pagina']:.0f}  conf {s['conf_media']:.1f}  baja conf {s['palabras_baja_conf']:.2f}")
    else:
        stats = ocr_folder(a.input, a.out, a.mode, workers=a.workers)
        print(f"{len(stats)} páginas en {a.out}; conf media {np.mean([s['mean_conf'] for s in stats]):.1f}")
