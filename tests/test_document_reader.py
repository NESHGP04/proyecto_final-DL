"""Pruebas con PDF inventados; no usan documentos, pesos ni datos del bufete."""
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.document_reader import extract_pdf, ocr_status, pdf_page_count, _ocr_page

SENTENCE = "El trabajador solicita orientacion laboral por despido y salario pendiente."


def native_pdf(texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            stream = DecodedStreamObject()
            literal = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 35 700 Td ({literal}) Tj ET".encode("latin-1"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class DocumentReaderTests(unittest.TestCase):
    def test_native_pdf_needs_no_ocr_and_respects_range(self):
        data = native_pdf([SENTENCE, "Segunda pagina. " + SENTENCE])
        with patch("app.document_reader.ocr_status", side_effect=AssertionError("No debe usar OCR")):
            result = extract_pdf(data, 2, 2)
        self.assertEqual(pdf_page_count(data), 2)
        self.assertIn("Segunda pagina", result.text)
        self.assertEqual(result.first_page, 2)
        self.assertEqual(result.pages[0]["metodo"], "Texto nativo")

    def test_mixed_pdf_ocr_only_where_needed_and_removes_temporary(self):
        data = native_pdf([SENTENCE, None])
        progress = []
        with patch("app.document_reader.ocr_status", return_value={"ready": True}), \
             patch("app.document_reader._ocr_page", return_value="Texto OCR ficticio de una pagina escaneada.") as mock:
            result = extract_pdf(data, 1, 2, progress=lambda done,n: progress.append((done,n)))
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(mock.call_args.args[1], 2)
        self.assertFalse(Path(mock.call_args.args[0]).exists())
        self.assertEqual([p["metodo"] for p in result.pages], ["Texto nativo", "OCR local"])
        self.assertEqual(progress, [(1,2),(2,2)])

    def test_force_ocr_replaces_native_layer(self):
        with patch("app.document_reader.ocr_status", return_value={"ready": True}), \
             patch("app.document_reader._ocr_page", return_value="Otra lectura ficticia"):
            result = extract_pdf(native_pdf([SENTENCE]), 1, 1, force_ocr=True)
        self.assertEqual(result.text, "Otra lectura ficticia")

    def test_missing_ocr_stops_instead_of_sending_partial_native_text(self):
        with patch("app.document_reader.ocr_status", return_value={"ready": False}):
            with self.assertRaisesRegex(ValueError, "OCR local"):
                extract_pdf(native_pdf([SENTENCE, None]), 1, 2)

    def test_failed_ocr_cleans_temporary_and_returns_no_partial_result(self):
        paths = []
        def fail(path, *_):
            paths.append(path)
            raise ValueError("Fallo OCR simulado")
        with patch("app.document_reader.ocr_status", return_value={"ready": True}), \
             patch("app.document_reader._ocr_page", side_effect=fail):
            with self.assertRaises(ValueError):
                extract_pdf(native_pdf([SENTENCE, None]), 1, 2)
        self.assertFalse(Path(paths[0]).exists())

    def test_unreadable_document_is_not_an_empty_classification(self):
        with patch("app.document_reader.ocr_status", return_value={"ready": True}), \
             patch("app.document_reader._ocr_page", return_value=""):
            with self.assertRaisesRegex(ValueError, "No se recuperó texto"):
                extract_pdf(native_pdf([None]), 1, 1)

    def test_bad_encrypted_and_empty_pdf(self):
        with self.assertRaises(ValueError):
            pdf_page_count(b"Esto no es PDF")
        writer = PdfWriter()
        writer.add_blank_page(612,792)
        writer.encrypt("password-de-prueba")
        buf = io.BytesIO(); writer.write(buf)
        with self.assertRaisesRegex(ValueError, "protegido"):
            pdf_page_count(buf.getvalue())
        writer = PdfWriter(); buf = io.BytesIO(); writer.write(buf)
        with self.assertRaisesRegex(ValueError, "no contiene páginas"):
            pdf_page_count(buf.getvalue())

    def test_invalid_ranges_and_limit(self):
        data = native_pdf([SENTENCE])
        for first,last in ((0,1),(2,2),(1,2),(1.5,1)):
            with self.subTest(first=first,last=last), self.assertRaises(ValueError):
                extract_pdf(data,first,last)
        with self.assertRaisesRegex(ValueError, "máximo 10"):
            extract_pdf(native_pdf([None]*11),1,11)
        with patch("app.document_reader.MAX_PDF_BYTES",10):
            with self.assertRaisesRegex(ValueError,"50 MB"):
                pdf_page_count(data)

    def test_text_size_limit_does_not_silently_truncate(self):
        with patch("app.document_reader.MAX_EXTRACTED_CHARS",10):
            with self.assertRaisesRegex(ValueError,"demasiado texto"):
                extract_pdf(native_pdf([SENTENCE]),1,1)

    def test_spanish_is_required_no_silent_english_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("app.document_reader.shutil.which", side_effect=lambda name: "/bin/"+name), \
                 patch("app.document_reader.subprocess.run", return_value=subprocess.CompletedProcess([],0,stdout=b"List of languages:\neng\nosd\n",stderr=b"")):
                status = ocr_status(td,language="spa")
        self.assertFalse(status["ready"])
        self.assertFalse(status["language_ready"])

    def test_ocr_errors_do_not_echo_process_output(self):
        from PIL import Image
        image = Image.new("L",(100,100),255)
        with patch("src.data.ocr.render_page",return_value=image), \
             patch("app.document_reader.subprocess.run",return_value=subprocess.CompletedProcess([],1,stdout=b"",stderr=b"CONTENIDO_PRIVADO_NO_MOSTRAR")):
            with self.assertRaises(ValueError) as caught:
                _ocr_page("ficticio.pdf",1)
        self.assertNotIn("CONTENIDO_PRIVADO",str(caught.exception))

    @unittest.skipUnless(ocr_status(language="eng")["ready"],"Requiere Poppler y Tesseract eng para smoke test real")
    def test_real_scanned_pdf_pipeline_with_available_english_language(self):
        # El entorno de pruebas tiene eng, no spa. Esto prueba integración real
        # de render/OCR sin afirmar calidad en español ni CER del proyecto.
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new("RGB",(1500,450),"white")
        ImageDraw.Draw(image).text((40,100),"El trabajador solicita pago por despido",fill="black",font=ImageFont.load_default(size=48))
        buf = io.BytesIO(); image.save(buf,format="PDF",resolution=150); image.close()
        self.assertFalse(PdfReader(io.BytesIO(buf.getvalue())).pages[0].extract_text())
        result = extract_pdf(buf.getvalue(),1,1,language="eng")
        self.assertIn("trabajador",result.text.casefold())
        self.assertIn("despido",result.text.casefold())
        self.assertEqual(result.pages[0]["metodo"],"OCR local")


if __name__=="__main__":
    unittest.main()
