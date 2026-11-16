"""Lectura local de PDF para el dashboard: texto nativo y OCR por página.

No llama a servicios externos ni guarda texto, nombres de archivo o documentos
en el repositorio. Los temporales de OCR se eliminan al salir, también si falla.
"""
from dataclasses import dataclass
import io
import math
import operator
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_SELECTED_PAGES = 10
MAX_EXTRACTED_CHARS = 100_000
MIN_NATIVE_ALNUM = 40


@dataclass(frozen=True)
class PDFText:
    text: str
    total_pages: int
    first_page: int
    last_page: int
    pages: tuple


def _reader(data):
    if not isinstance(data, bytes) or not data or b"%PDF-" not in data[:1024]:
        raise ValueError("El archivo no contiene un PDF válido.")
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("El PDF supera el límite de 50 MB.")
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("Falta el lector PDF. Instala requirements-dashboard.txt en el entorno del dashboard.") from None
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        encrypted = reader.is_encrypted
    except Exception:
        raise ValueError("No se pudo leer el PDF. Comprueba que esté completo y no esté dañado.") from None
    if encrypted:
        raise ValueError("El PDF está protegido. Usa una copia que puedas abrir sin contraseña.")
    try:
        count = len(reader.pages)
    except Exception:
        raise ValueError("No se pudieron leer las páginas del PDF.") from None
    if not count:
        raise ValueError("El PDF no contiene páginas.")
    return reader


def pdf_page_count(data):
    return len(_reader(data).pages)


def ocr_status(tessdata_dir=None, language="spa"):
    """Diagnóstico sin contenido de documentos ni rutas personales en la salida."""
    custom = Path(tessdata_dir) if tessdata_dir is not None else ROOT / "artifacts/tessdata"
    local_model = custom / f"{language}.traineddata"
    model_dir = custom if local_model.is_file() and local_model.stat().st_size else None
    missing = [tool for tool in ("tesseract", "pdftoppm", "pdfinfo") if shutil.which(tool) is None]
    languages = set()
    if "tesseract" not in missing:
        command = [shutil.which("tesseract"), "--list-langs"]
        if model_dir is not None:
            command += ["--tessdata-dir", str(model_dir)]
        try:
            result = subprocess.run(command, capture_output=True, timeout=15, check=False)
            if result.returncode == 0:
                languages = set(result.stdout.decode("utf-8", errors="replace").splitlines())
        except (OSError, subprocess.TimeoutExpired):
            pass
    return dict(ready=not missing and language in languages, missing=missing,
                language_ready=language in languages,
                model_source="proyecto" if model_dir is not None else "sistema")


def _ocr_page(pdf_path, number, tessdata_dir=None, language="spa"):
    try:
        from src.data.ocr import preprocess, render_page
        image = preprocess(render_page(pdf_path, number, dpi=300), mode="gray")
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            png = buffer.getvalue()
        finally:
            image.close()
    except ImportError:
        raise ValueError("Faltan dependencias de OCR. Instala requirements-dashboard.txt.") from None
    except Exception:
        raise ValueError(f"No se pudo renderizar la página {number} para OCR. Comprueba Poppler y el PDF.") from None
    custom = Path(tessdata_dir) if tessdata_dir is not None else ROOT / "artifacts/tessdata"
    command = [shutil.which("tesseract") or "tesseract", "stdin", "stdout", "-l", language, "--psm", "4", "--oem", "1"]
    if (custom / f"{language}.traineddata").is_file() and (custom / f"{language}.traineddata").stat().st_size:
        command += ["--tessdata-dir", str(custom)]
    try:
        result = subprocess.run(command, input=png, capture_output=True, timeout=90, check=False)
    except subprocess.TimeoutExpired:
        raise ValueError(f"El OCR de la página {number} superó el tiempo de espera. Prueba un rango menor.") from None
    except OSError:
        raise ValueError("No se pudo ejecutar Tesseract localmente.") from None
    if result.returncode != 0:
        raise ValueError(f"Falló el OCR de la página {number}. Comprueba Tesseract y el idioma español.")
    return result.stdout.decode("utf-8", errors="replace").strip()


def extract_pdf(data, first_page=1, last_page=None, force_ocr=False, progress=None,
                tessdata_dir=None, language="spa"):
    """Texto de un rango explícito de páginas físicas; no clasifica ni trunca.

    Auto usa OCR cuando hay menos de 40 caracteres alfanuméricos nativos.
    Una capa de texto parcial puede requerir force_ocr y revisión del usuario.
    """
    reader = _reader(data)
    total = len(reader.pages)
    try:
        first = operator.index(first_page)
        last = total if last_page is None else operator.index(last_page)
    except TypeError:
        raise ValueError("El rango de páginas debe usar números enteros.") from None
    if not 1 <= first <= last <= total:
        raise ValueError("El rango de páginas no es válido para este PDF.")
    if last-first+1 > MAX_SELECTED_PAGES:
        raise ValueError("Selecciona como máximo 10 páginas por extracción.")
    planned = []
    for number in range(first, last+1):
        page = reader.pages[number-1]
        try:
            native = "" if force_ocr else (page.extract_text() or "").strip()
        except Exception:
            native = ""
        use_ocr = force_ocr or sum(char.isalnum() for char in native) < MIN_NATIVE_ALNUM
        if use_ocr:
            try:
                pixels = float(page.mediabox.width)*float(page.mediabox.height)*(300/72)**2
            except Exception:
                raise ValueError(f"No se pudo determinar el tamaño de la página {number}.") from None
            if not math.isfinite(pixels) or not 0 < pixels <= 25_000_000:
                raise ValueError(f"La página {number} es demasiado grande para OCR a 300 dpi.")
        planned.append((number, native, use_ocr))
    needs_ocr = any(item[2] for item in planned)
    if needs_ocr:
        status = ocr_status(tessdata_dir, language)
        if not status["ready"]:
            raise ValueError("Se necesita OCR local para este rango. Comprueba Tesseract, Poppler y el idioma español; consulta LEEME_PDF.txt.")
    texts, pages = [], []
    with tempfile.TemporaryDirectory(prefix="practica_legal_pdf_") as temporary:
        path = Path(temporary) / "documento.pdf"
        if needs_ocr:
            path.write_bytes(data)
        size = 0
        for done, (number, native, use_ocr) in enumerate(planned, 1):
            text = _ocr_page(path, number, tessdata_dir, language) if use_ocr else native
            method = "OCR local" if use_ocr else "Texto nativo"
            if not text.strip():
                method = "Sin texto recuperado"
            texts.append(text)
            pages.append(dict(pagina=number, metodo=method, caracteres=len(text)))
            size += len(text) + 2
            if size > MAX_EXTRACTED_CHARS:
                raise ValueError("Se extrajo demasiado texto. Selecciona menos páginas; no se recortó ni clasificó el documento.")
            if progress is not None:
                progress(done, len(planned))
    text = "\n\n".join(t for t in texts if t.strip()).strip()
    if not text:
        raise ValueError("No se recuperó texto del rango seleccionado. Revisa el escaneo o elige otras páginas.")
    return PDFText(text=text, total_pages=total, first_page=first, last_page=last, pages=tuple(pages))
