import gzip
import json

from src.data.saij import build_by_area, download, filter_records, map_materia, normalize_materia

LARGO = " ".join(["palabra"] * 30)


def test_map_materia_covers_variants_and_excludes_ambiguous():
    assert map_materia("LABORAL") == "laboral" and map_materia("Derecho del Trabajo") == "laboral"
    assert map_materia("PENAL") == "penal" and map_materia("PROCESAL-PENAL") == "penal" and map_materia("PENAL - PROCESAL") == "penal"
    assert map_materia("CIVIL") == "civil" and map_materia("PROCESAL-CIVIL") == "civil"
    assert map_materia("COMERCIAL") == "mercantil" and map_materia("FAMILIA") == "familia"
    assert map_materia("CIVIL - COMERCIAL") is None and map_materia("CIVIL-COMERCIAL") is None
    assert map_materia("PROCESAL") is None and map_materia("ADMINISTRATIVO") == "otros"
    assert map_materia("SEGURIDAD SOCIAL") == "otros" and map_materia("ASTROLOGIA") is None
    assert normalize_materia("Civil – Comercial") == "CIVIL-COMERCIAL"


def test_filter_drops_unlabeled_short_and_duplicate_texts():
    recs = [{"materia": "LABORAL", "texto": LARGO, "numero-sumario": "1", "fecha": "2020-01-01", "tipo-tribunal": "LB"},
            {"materia": None, "texto": LARGO + " x"}, {"materia": "PENAL", "texto": "muy corto"},
            {"materia": "PENAL", "texto": LARGO.upper()}, {"materia": "PENAL", "texto": LARGO + " distinto"}, {"texto": LARGO}]
    rows = list(filter_records(recs))
    assert [r["materia"] for r in rows] == ["LABORAL", "PENAL"]  # el duplicado en mayúsculas se descarta
    assert rows[0]["anio"] == "2020" and rows[0]["id"] == "1"


def test_download_and_build_with_fake_stream(tmp_path):
    recs = [{"materia": m, "texto": f"{LARGO} {m} {i}", "numero-sumario": f"{m}{i}"}
            for m in ("LABORAL", "PENAL", "CIVIL", "CIVIL - COMERCIAL", "ADMINISTRATIVO") for i in range(8)]
    summary = download(tmp_path, records=iter(recs))
    assert summary["registros_guardados"] == 40 and dict(summary["materias"])["LABORAL"] == 8
    with gzip.open(tmp_path / "saij_filtrado.csv.gz", "rt", encoding="utf-8") as f:
        assert f.readline().strip() == "id,materia,texto,anio,tipo_tribunal"
    res = build_by_area(tmp_path, per_area=5, areas=["laboral", "civil", "penal", "mercantil", "familia", "otros"], min_words=5)
    assert res["guardados"] == {"laboral": 5, "civil": 5, "penal": 5, "otros": 5} and res["disponibles"]["familia"] == 0
    assert res["materias_sin_mapear"][0][0] == "CIVIL-COMERCIAL"
    first = open(tmp_path / "saij_por_area.csv", encoding="utf-8").read().splitlines()[1]
    assert "saij" in first and "materia" in first  # filas con origen y tipo de etiqueta


def test_family_keywords_move_civil_texts_to_familia_as_weak_label(tmp_path):
    recs = [{"materia": "CIVIL", "texto": f"{LARGO} divorcio vincular {i}", "numero-sumario": f"f{i}"} for i in range(4)]
    recs += [{"materia": "CIVIL", "texto": f"{LARGO} contrato de compraventa {i}", "numero-sumario": f"c{i}"} for i in range(4)]
    download(tmp_path, records=iter(recs))
    res = build_by_area(tmp_path, per_area=10, areas=["civil", "familia"], min_words=5)
    assert res["guardados"] == {"civil": 4, "familia": 4}
    import csv
    rows = list(csv.DictReader(open(tmp_path / "saij_por_area.csv", encoding="utf-8")))
    assert {r["etiqueta"] for r in rows if r["area_legal"] == "familia"} == {"palabras_clave"}
    assert {r["etiqueta"] for r in rows if r["area_legal"] == "civil"} == {"materia"}
