"""Datos sintéticos SOLO para probar que el código corre.

Ningún resultado que se reporte puede venir de aquí (docs/CONTEXTO.md, regla 7). El vocabulario es
genérico e inventado; no representa casos reales.
"""
import random

import pandas as pd

_VOCAB = {
    "laboral": ["despido", "salario", "indemnización", "prestaciones", "patrono", "contrato de trabajo", "vacaciones"],
    "civil": ["escritura", "propiedad", "herencia", "contrato de arrendamiento", "daños", "demanda civil", "sucesión"],
    "mercantil": ["sociedad anónima", "factura", "cobro", "socios", "registro mercantil", "pagaré", "proveedor"],
    "penal": ["denuncia", "ministerio público", "audiencia", "medida de coerción", "fiscalía", "querella", "delito"],
    "familia": ["pensión alimenticia", "custodia", "divorcio", "régimen de visitas", "matrimonio", "menores", "patria potestad"],
    "otros": ["consulta", "asesoría", "trámite", "documentos", "revisión", "información", "cita"],
}
_FILLER = ["necesito ayuda con", "quisiera consultar sobre", "mi caso trata de", "solicito asesoría por", "tengo un problema con"]
_CANALES = ["correo", "formulario", "documento", "fisico"]


def make_synthetic(areas, n_per_area=40, seed=42):
    rng = random.Random(seed)
    rows = []
    k = 0
    for area in areas:
        words = _VOCAB.get(area, _VOCAB["otros"])
        for _ in range(n_per_area):
            k += 1
            # mezcla ruido de otra área para que no sea trivial
            other = rng.choice(list(_VOCAB.values()))
            text = f"{rng.choice(_FILLER)} {rng.choice(words)} y {rng.choice(words)}; también {rng.choice(other)}."
            canal = rng.choice(_CANALES)
            rows.append({
                "case_id": f"S{k:04d}",
                "texto": text,
                "area_legal": area,
                "canal": canal,
                "fecha": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                "fuente_texto": "ocr" if canal == "fisico" else "nativo",
            })
    return pd.DataFrame(rows)
