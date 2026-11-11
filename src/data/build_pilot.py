"""De texto de OCR a fragmentos anonimizados listos para revisar (conjunto piloto de los 12 PDFs).

Flujo: páginas de OCR -> limpieza -> hechos de demandas/ampliaciones -> fragmentos de ~150 palabras
-> anonimización (reglas + nombres candidatos + nombres curados) -> informe de residuos.

Salidas en data/interim/piloto/ (ignorado por git):
  fragmentos.csv          fragmentos anonimizados (revisión_humana = no hasta que Cami los revise)
  nombres_candidatos.txt  nombres detectados automáticamente: Cami los revisa y guarda la lista final
                          en nombres_caso.txt (esa lista, si existe, se aplica además de los candidatos)
  informe.json            solo conteos y estadísticas, sin texto: se puede compartir

Uso: python -m src.data.build_pilot
"""
import argparse
import csv
import json
from pathlib import Path

from src.config import ROOT
from src.data.anonymize import anonymize, candidate_names, load_names, residual_report
from src.data.clean import clean_document
from src.data.segment import chunk_words, extract_section, find_sections

FUENTES_DE_HECHOS = {"demanda", "ampliacion"}


def _pages(ocr_dir, doc_id):
    return [p.read_text(encoding="utf-8") for p in sorted(Path(ocr_dir).glob(f"{doc_id}_p*.txt"))]


def build_pilot(ocr_dir, out_dir, area="laboral", target_words=150):
    ocr_dir, out = Path(ocr_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(ocr_dir / "inventario.csv", encoding="utf-8") as f:
        inventory = list(csv.DictReader(f))

    docs, names = {}, set()
    for row in inventory:
        lines = clean_document(_pages(ocr_dir, row["doc_id"]))
        docs[row["doc_id"]] = lines
        names.update(candidate_names("\n".join(lines)))
    curated = load_names(out / "nombres_caso.txt")
    all_names = sorted(names | set(curated), key=len, reverse=True)
    (out / "nombres_candidatos.txt").write_text(
        "# Candidatos automáticos (pueden tener falsos positivos y omitir nombres). Revisar y guardar la lista\n"
        "# final, uno por línea, en nombres_caso.txt. Este archivo contiene datos personales: no compartir.\n"
        + "\n".join(sorted(names)) + "\n", encoding="utf-8")

    report, rows = {"documentos": [], "fragmentos": 0}, []
    for row in inventory:
        lines = docs[row["doc_id"]]
        marks = [name for _, name in find_sections(lines)]
        info = {"doc_id": row["doc_id"], "tipo": row["tipo_documento"], "paginas": int(row["paginas"]),
                "lineas_limpias": len(lines), "palabras": sum(len(l.split()) for l in lines), "secciones": marks}
        if row["tipo_documento"] in FUENTES_DE_HECHOS:
            hechos = extract_section(lines, "hechos")
            info["hechos_encontrados"] = hechos is not None
            if hechos:
                info["palabras_hechos"] = sum(len(l.split()) for l in hechos)
                for k, chunk in enumerate(chunk_words(" ".join(hechos), target_words), 1):
                    rows.append({"case_id": f"P{row['doc_id']}-{k:02d}", "texto": anonymize(chunk, all_names),
                                 "area_legal": area, "canal": "fisico", "fecha": "", "fuente_texto": "ocr",
                                 "grupo_id": "EXP01", "tipo_documento": row["tipo_documento"],
                                 "revision_humana": "no"})
        report["documentos"].append(info)

    report["fragmentos"] = len(rows)
    report["palabras_fragmentos"] = sum(len(r["texto"].split()) for r in rows)
    report["nombres_candidatos"] = len(names)
    report["nombres_curados"] = len(curated)
    res = [residual_report(r["texto"]) for r in rows]
    report["residuos_tras_anonimizar"] = {k: sum(r[k] for r in res) for k in res[0]} if res else {}
    report["tokens_marcadores"] = {t: sum(r["texto"].count(t) for r in rows) for t in
                                   ("[CORREO]", "[CASILLERO]", "[EXPEDIENTE]", "[DPI]", "[NIT]", "[TELEFONO]",
                                    "[DIRECCION]", "[NUMERO_ESCRITO]", "[EMPRESA]", "[PERSONA]", "[COLEGIADO]")}
    if rows:
        with open(out / "fragmentos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "informe.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", default=str(ROOT / "data" / "interim" / "ocr"))
    ap.add_argument("--out", default=str(ROOT / "data" / "interim" / "piloto"))
    a = ap.parse_args()
    print(json.dumps(build_pilot(a.ocr, a.out), ensure_ascii=False, indent=1))
