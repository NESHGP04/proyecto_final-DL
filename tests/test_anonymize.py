from src.data.anonymize import anonymize


def test_patterns_are_masked():
    text = "Escríbame a juan.perez@mail.com o al +502 5555-1234. DPI 1234 56789 0101, NIT 1234567-8."
    out = anonymize(text)
    for token in ("[CORREO]", "[TELEFONO]", "[DPI]", "[NIT]"):
        assert token in out
    for leaked in ("juan.perez", "5555", "56789", "1234567"):
        assert leaked not in out


def test_names_from_list():
    assert anonymize("Habla María López sobre su caso", names=["maría lópez"]) == "Habla [PERSONA] sobre su caso"
