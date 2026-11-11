"""Anonimización de texto.

Ya trae regex para correo, DPI, NIT y teléfono de Guatemala, y reemplazo de nombres a partir de
una lista. Falta (Cami): direcciones y nombres sin lista (p. ej. NER en español), y revisar a mano
una muestra para confirmar que no queda nada sensible. Los patrones son heurísticos: no
garantizan anonimización completa.
"""
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DPI = re.compile(r"\b\d{4}\s?\d{5}\s?\d{4}\b")           # 13 dígitos, p. ej. 1234 56789 0101
_NIT = re.compile(r"\b\d{4,9}-[\dkK]\b")                  # p. ej. 1234567-8
_TEL = re.compile(r"(?:\+?502[\s-]?)?\b[2-7]\d{3}[\s-]?\d{4}\b|\b\d{4}\s?[-–]\s?\d{4}\b")  # 8 dígitos (o 4-4 con guion, por errores de OCR)

# El orden importa: DPI y NIT antes que teléfono para que no se corten en pedazos.
_CASILLERO = re.compile(r"\bMR\s?\d{5,}\b", re.I)                 # casillero electrónico del Organismo Judicial
_EXPEDIENTE = re.compile(r"\b\d{3,6}\s?-\s?\d{4}\s?-\s?\d{2,6}(?:\s?-\s?\w{1,4})?\b")  # p. ej. 01173-2026-00123
_COLEGIADO = re.compile(r"(colegiad[oa]\s+(?:activ[oa]\s+)?(?:n[úu]mero|no\.?|n[úu]m\.?)\s*)[\d.\s]{2,10}", re.I)
_DIRECCION = re.compile(r"\(\s*\d{1,3}\s*[-–—]\s*\d{1,3}\s*\)")           # (25-86) de una dirección
_EMPRESA = re.compile(r"\b(?:[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ&.]+,?\s+){1,5}(?:SOCIEDAD AN[OÓ]NIMA|S\.\s?A\.)")
_NUMERALES = (r"cero|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|"
              r"diecis[eé]is|diecisiete|dieciocho|diecinueve|veinte|veinti\w+|treinta|cuarenta|cincuenta|sesenta|"
              r"setenta|ochenta|noventa|cien|ciento|doscientos|trescientos|cuatrocientos|quinientos|seiscientos|"
              r"setecientos|ochocientos|novecientos|mil|espacio|guion")
# 6 o más numerales seguidos (con 'y' de enlace): un CUI, NIT o teléfono deletreado. 'treinta y tres' (edad) no llega.
_NUM_ESCRITO = re.compile(rf"\b(?:(?:{_NUMERALES})\b(?:\s+y\b)?\s*){{6,}}", re.I)

# El orden importa: DPI y NIT antes que teléfono para que no se corten en pedazos.
_PATTERNS = [(_EMAIL, "[CORREO]"), (_CASILLERO, "[CASILLERO]"), (_EXPEDIENTE, "[EXPEDIENTE]"), (_DPI, "[DPI]"),
             (_NIT, "[NIT]"), (_TEL, "[TELEFONO]"), (_DIRECCION, "[DIRECCION]"), (_NUM_ESCRITO, "[NUMERO_ESCRITO] ")]


def mask_patterns(text):
    text = _COLEGIADO.sub(r"\1[COLEGIADO]", text)
    text = _EMPRESA.sub("[EMPRESA]", text)
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def mask_names(text, names):
    """Reemplaza cada nombre de `names` (sin distinguir mayúsculas) por [PERSONA]."""
    for name in sorted(names, key=len, reverse=True):
        if name.strip():
            text = re.sub(re.escape(name), "[PERSONA]", text, flags=re.IGNORECASE)
    return text


def anonymize(text, names=()):
    return mask_patterns(mask_names(text, names))


# --- nombres: candidatos automáticos para revisar a mano -------------------------------------------
# Palabras que en mayúsculas son vocabulario legal y no nombres propios.
_STOP = set("""juicio ordinario laboral juez jueza juzgado primera segunda tercera instancia trabajo prevision social
guatemala departamento municipio ciudad zona calle avenida edificio nivel torre sociedad anonima expongo hechos
primero segundo tercero cuarto quinto sexto septimo octavo noveno decimo peticion pruebas prueba derecho
fundamento fundamentos indemnizacion despido directo sin causa justificada danos perjuicios costas judiciales
demanda ampliacion resolucion auto sentencia audiencia previo subsanacion desistimiento actor demandado
demandante demandada patrono trabajador trabajadora notifiquese oficial secretario pliego posiciones acuerdo
senor senora licenciado licenciada abogado abogada notario sala corte suprema justicia organismo judicial
republica codigo articulo articulos ley decreto constitucion politica por para con del de la las los el y en
a al que se su sus un una no si es son como mas muy o u ni se contra ante segun sobre hasta desde cuando""".split())
_HONORIFICO = re.compile(r"\b(?:se[ñn]or[a]?|licenciad[oa]|lic\.|abogad[oa]|notari[oa]|doctor[a]?|dr\.|ing\.)\s+"
                         r"((?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+){1,4})")
_TITLECASE = re.compile(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:de\s+(?:la\s+|las\s+|los\s+)?|del\s+)?[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+\b")
_MAYUSCULAS = re.compile(r"\b(?:[A-ZÁÉÍÓÚÑ]{2,}\s+){1,}[A-ZÁÉÍÓÚÑ]{2,}\b")


def candidate_names(text):
    """Posibles nombres de personas: secuencias en MAYÚSCULAS o de palabras con mayúscula inicial con
    alguna palabra fuera del vocabulario legal, y nombres tras un tratamiento (señor, licenciado…).
    Es una ayuda para que una persona arme la lista final: incluye falsos positivos (instituciones,
    lugares) y puede omitir nombres escritos en minúsculas o con errores de OCR."""
    from src.data.clean import strip_accents

    found = set()
    for m in _MAYUSCULAS.finditer(text):
        words = m.group(0).split()
        while words and strip_accents(words[0].lower()) in _STOP:
            words.pop(0)
        while words and strip_accents(words[-1].lower()) in _STOP:
            words.pop()
        if len(words) >= 2:
            found.add(" ".join(words))
    for m in _TITLECASE.finditer(text):
        words = m.group(0).split()
        while words and strip_accents(words[0].lower()) in _STOP:
            words.pop(0)
        while words and strip_accents(words[-1].lower()) in _STOP:
            words.pop()
        if len([w for w in words if w[0].isupper()]) >= 2:
            found.add(" ".join(words))
    for m in _HONORIFICO.finditer(text):
        name = " ".join(m.group(1).split())
        if len(name.split()) >= 2:
            found.add(name)
    return sorted(found)


def load_names(path):
    """Lista de nombres a enmascarar, uno por línea (líneas vacías o con # se ignoran)."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def residual_report(text):
    """Cuenta posibles datos personales que sobreviven, sin devolver el texto."""
    from src.data.clean import strip_accents

    caps = [m.group(0).split() for m in _MAYUSCULAS.finditer(text)]
    return {
        "digitos_largos": len(re.findall(r"\d[\d\s.-]{6,}\d", text)),
        "casillero": len(_CASILLERO.findall(text)),
        "correo": len(_EMAIL.findall(text)),
        "mayusculas_no_legales": sum(any(strip_accents(w.lower()) not in _STOP for w in ws) for ws in caps),
    }
