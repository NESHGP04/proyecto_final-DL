from src.data.anonymize import anonymize, candidate_names, load_names, residual_report
from src.data.clean import clean_document, clean_page_lines, join_hyphenation, remove_repeated_lines
from src.data.segment import chunk_words, extract_section, find_sections, heading_of

# Todo el texto de estas pruebas es inventado.


def test_clean_page_drops_noise_and_folios():
    page = "Ana Pérez trabaja aquí\n12\n|| ~~ ..\nPágina 3 de 10\nsegunda   línea   real\n#$%&/(\n"
    assert clean_page_lines(page) == ["Ana Pérez trabaja aquí", "segunda línea real"]


def test_repeated_headers_removed_only_with_enough_pages():
    pages = [["ORGANISMO JUDICIAL", f"texto {i} distinto"] for i in range(5)]
    cleaned = remove_repeated_lines(pages)
    assert all("ORGANISMO JUDICIAL" not in p for p in cleaned) and cleaned[0] == ["texto 0 distinto"]
    assert remove_repeated_lines(pages[:2]) == pages[:2]


def test_hyphenation_joined_and_document_clean():
    assert join_hyphenation(["la indemniza-", "ción del trabajador", "fin-", "Mayúscula"]) == [
        "la indemnización del trabajador", "fin-", "Mayúscula"]
    assert clean_document(["uno dos tres\n7", "cuatro cinco seis"]) == ["uno dos tres", "cuatro cinco seis"]


def test_headings_and_hechos_extraction():
    lines = ["JUICIO ORDINARIO", "EXPONGO: soy trabajador", "HECHOS:", "PRIMERO: me contrataron.", "SEGUNDO: me despidieron.",
             "FUNDAMENTO DE DERECHO", "artículo 78", "PETICIÓN: que se declare"]
    assert [n for _, n in find_sections(lines)] == ["exposicion", "hechos", "derecho", "peticion"]
    assert extract_section(lines, "hechos") == ["PRIMERO: me contrataron.", "SEGUNDO: me despidieron."]
    assert extract_section(lines, "pruebas") is None
    assert heading_of("Los hechos ocurrieron el lunes por la tarde en la oficina")[0] is None


def test_hechos_title_with_text_on_same_line():
    assert extract_section(["Hechos: primero me contrataron.", "segundo me despidieron."], "hechos") == [
        "primero me contrataron.", "segundo me despidieron."]


def test_chunk_words_cuts_at_sentences_and_merges_short_tail():
    text = " ".join(f"Oración número {i} con varias palabras más." for i in range(40))
    chunks = chunk_words(text, target=30, min_last=15)
    assert all(c.endswith(".") for c in chunks) and len(chunks) > 3
    assert " ".join(chunks).split() == text.split()


def test_new_patterns_masked_without_touching_ordinary_numbers():
    text = ("Casillero MR00018144, expediente 01173-2026-00123, colegiado activo número 12345, "
            "ubicada en (25-86) zona 10; CUI dos mil doscientos espacio noventa y siete mil noventa y siete "
            "cero ciento uno (2200 97097 0101). Tiene treinta y tres años y trabajó 5 años.")
    out = anonymize(text)
    for leaked in ("MR00018144", "01173", "12345", "25-86", "noventa y siete", "2200"):
        assert leaked not in out
    assert "treinta y tres años" in out and "5 años" in out
    assert "[CASILLERO]" in out and "[EXPEDIENTE]" in out and "[COLEGIADO]" in out


def test_company_and_candidate_names_and_residuals():
    text = "Demanda contra TRANSPORTES DEL SUR, SOCIEDAD ANONIMA. El señor Luis Alberto Gómez Paz habló. MARIA JOSE LOPEZ TORRES firma. JUICIO ORDINARIO LABORAL."
    assert "[EMPRESA]" in anonymize(text)
    cands = candidate_names(text)
    assert "MARIA JOSE LOPEZ TORRES" in cands and "Luis Alberto Gómez Paz" in cands
    assert not any("JUICIO" in c for c in cands)
    masked = anonymize(text, names=cands)
    assert "LOPEZ" not in masked and "Gómez" not in masked
    assert residual_report("MARIA JOSE LOPEZ y MR12345678 y 1234567890")["mayusculas_no_legales"] == 1


def test_load_names_ignores_comments(tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("# revisado\nAna López\n\nBanco Azul\n", encoding="utf-8")
    assert load_names(f) == ["Ana López", "Banco Azul"]
    assert load_names(tmp_path / "no_existe.txt") == []


def test_phone_with_dash_even_if_ocr_changed_the_first_digit():
    assert anonymize("llamar al 9123-4567 o al 2456 7890") == "llamar al [TELEFONO] o al [TELEFONO]"
