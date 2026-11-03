"""OCR de casos físicos escaneados. Dueña: Cami (CP2).

Pendiente: convertir cada PDF/imagen de data/raw a texto (p. ej. pdf2image + pytesseract con el
idioma `spa`), guardar el texto en data/interim/ y marcar fuente_texto = "ocr". Medir la tasa de
error por carácter (CER) contra la muestra transcrita a mano y reportarla.
"""


def ocr_file(path):
    """Devuelve el texto extraído de un PDF o imagen escaneada."""
    raise NotImplementedError("Cami, CP2")


def character_error_rate(reference, hypothesis):
    """CER = distancia de edición entre caracteres / longitud de la referencia."""
    raise NotImplementedError("Cami, CP2")
