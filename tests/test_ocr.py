import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.data.ocr import (TESSDATA, character_error_rate, estimate_skew, ocr_image, otsu_threshold, preprocess,
                          sauvola, tipo_documento)


def _lines_image(angle=0.0):
    img = Image.new("L", (600, 800), 255)
    d = ImageDraw.Draw(img)
    for y in range(60, 740, 40):
        d.rectangle([40, y, 560, y + 14], fill=0)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)


def test_cer():
    assert character_error_rate("hola mundo", "hola mundo") == 0.0
    assert character_error_rate("abcd", "abxd") == pytest.approx(0.25)
    assert character_error_rate("abcd", "") == 1.0 and character_error_rate("", "") == 0.0


def test_skew_estimate_recovers_rotation():
    assert estimate_skew(_lines_image(2.0)) == pytest.approx(-2.0, abs=0.35)
    assert abs(estimate_skew(_lines_image(0.0))) < 0.35


def test_binarizers_give_two_levels_and_otsu_splits_modes():
    arr = np.asarray(_lines_image())
    assert 0 < otsu_threshold(arr) < 255
    for mode in ("otsu", "sauvola"):
        out = np.asarray(preprocess(_lines_image(1.0), mode))
        assert set(np.unique(out)) <= {0, 255} and (out == 0).mean() > 0.01
    assert sauvola(arr).shape == arr.shape
    with pytest.raises(ValueError):
        preprocess(_lines_image(), "nope")


def test_tipo_documento_does_not_keep_names():
    assert tipo_documento("RESOLUCIÓN ADMITE A TRÁMITE.pdf") == "resolucion"
    assert tipo_documento("AMPLIACIÓN DE DEMANDA.pdf") == "ampliacion"
    assert tipo_documento("CARÁTULA EXPEDIENTE JUAN PEREZ.pdf") == "caratula"
    assert tipo_documento("otro papel.pdf") == "otro"


@pytest.mark.skipif(not (TESSDATA / "spa.traineddata").exists(), reason="falta el idioma español de Tesseract")
def test_ocr_reads_a_rendered_sentence():
    img = Image.new("L", (1400, 200), 255)
    ImageDraw.Draw(img).text((30, 50), "El trabajador solicita indemnizacion por despido", fill=0,
                             font=ImageFont.load_default(size=48))
    r = ocr_image(preprocess(img, "sauvola"), psm=7)
    assert character_error_rate("El trabajador solicita indemnizacion por despido", r["text"].strip()) < 0.15
    assert r["mean_conf"] > 60
