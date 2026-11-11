import csv
import json

from src.data.build_pilot import build_pilot


def _fake_ocr(tmp_path):
    ocr = tmp_path / "ocr"
    ocr.mkdir()
    demanda = ["JUZGADO DE TRABAJO", "EXPONGO: soy Luis Gómez Paz, DPI 1234 56789 0101.", "HECHOS:",
               "PRIMERO: trabajé para la empresa diez años y cobraba mi salario puntual cada mes del año.",
               "SEGUNDO: un día me dijeron que ya no regresara a trabajar y no me pagaron nada de lo debido.",
               "FUNDAMENTO DE DERECHO", "artículos del código de trabajo", "PETICIÓN: que se declare con lugar"]
    (ocr / "D01_p01.txt").write_text("\n".join(demanda), encoding="utf-8")
    (ocr / "D02_p01.txt").write_text("RESOLUCIÓN\nse admite a trámite la demanda presentada", encoding="utf-8")
    (ocr / "inventario.csv").write_text("doc_id,tipo_documento,paginas\nD01,demanda,1\nD02,resolucion,1\n", encoding="utf-8")
    return ocr


def test_pilot_keeps_only_hechos_masks_pii_and_reports_without_text(tmp_path):
    out = tmp_path / "piloto"
    rep = build_pilot(_fake_ocr(tmp_path), out, target_words=20)
    assert rep["fragmentos"] >= 1 and rep["documentos"][0]["hechos_encontrados"] is True
    rows = list(csv.DictReader(open(out / "fragmentos.csv", encoding="utf-8")))
    joined = " ".join(r["texto"] for r in rows)
    assert "trabajé para la empresa" in joined and "FUNDAMENTO" not in joined and "JUZGADO" not in joined
    assert all(r["area_legal"] == "laboral" and r["canal"] == "fisico" and r["revision_humana"] == "no" for r in rows)
    assert all(r["case_id"].startswith("PD01-") for r in rows)  # la resolución no aporta fragmentos
    texto_informe = (out / "informe.json").read_text(encoding="utf-8")
    assert "trabajé" not in texto_informe and "Gómez" not in texto_informe
    assert json.loads(texto_informe)["residuos_tras_anonimizar"]["digitos_largos"] == 0
    assert "Gómez Paz" in (out / "nombres_candidatos.txt").read_text(encoding="utf-8")
