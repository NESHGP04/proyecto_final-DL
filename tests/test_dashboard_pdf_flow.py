"""Flujo del dashboard con componentes simulados; no sustituye revisión visual."""
import sys
import types
import unittest
from unittest.mock import patch

from app import dashboard
from tests.test_document_reader import SENTENCE, native_pdf


class State(dict):
    def __getattr__(self,key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    def __setattr__(self,key,value):
        self[key]=value


class Context:
    def __init__(self,ui): self.ui=ui
    def __enter__(self): return self.ui
    def __exit__(self,*args): pass


class Progress:
    def progress(self,*args,**kwargs): return self
    def empty(self): pass


class Uploaded:
    def __init__(self,data,name="ficticio.pdf"):
        self.data,self.name,self.size=data,name,len(data)
    def getvalue(self): return self.data


class FakeUI(types.ModuleType):
    def __init__(self,data):
        super().__init__("streamlit")
        self.session_state=State(tipo_entrada="Cargar archivo")
        self.sidebar=Context(self)
        self.uploaded=Uploaded(data)
        self.extract=False
        self.submit=False
        self.messages=[]
        self.text_fields=[]
    def columns(self,n): return [self]*n
    def expander(self,*args,**kwargs): return Context(self)
    def form(self,*args,**kwargs): return Context(self)
    def spinner(self,*args,**kwargs): return Context(self)
    def progress(self,*args,**kwargs): return Progress()
    def file_uploader(self,*args,**kwargs): return self.uploaded
    def number_input(self,label,**kwargs):
        return self.session_state.setdefault(kwargs["key"],kwargs.get("value",kwargs["min_value"]))
    def checkbox(self,label,**kwargs): return self.session_state.setdefault(kwargs["key"],False)
    def slider(self,label,a,b,value,step): return value
    def text_input(self,label,value): return value
    def button(self,label,**kwargs):
        return kwargs.get("key")=="extraer_pdf" and self.extract and not kwargs.get("disabled",False)
    def form_submit_button(self,label,**kwargs):
        return label.startswith("Clasificar") and self.submit and not kwargs.get("disabled",False)
    def selectbox(self,label,options,**kwargs):
        return None if kwargs.get("index")==None and "index" in kwargs else list(options)[0]
    def text_area(self,label,**kwargs):
        key=kwargs["key"];self.text_fields.append(key)
        return self.session_state.get(key,SENTENCE)
    def success(self,text): self.messages.append(("success",text))
    def error(self,text): self.messages.append(("error",text))
    def __getattr__(self,name): return lambda *args,**kwargs: None


class DashboardPDFFlowTests(unittest.TestCase):
    def setUp(self):
        self.ui=FakeUI(native_pdf([SENTENCE, "Segunda pagina: "+SENTENCE]))
        self.patch=patch.dict(sys.modules,{"streamlit":self.ui});self.patch.start()
    def tearDown(self): self.patch.stop()
    def prepare(self):
        self.ui.extract=True;dashboard.main();self.ui.extract=False
    def test_prepare_requires_separate_classification_and_keeps_no_text_in_queue(self):
        self.prepare()
        self.assertEqual(self.ui.session_state.cola,[])
        self.assertIn("pdf_review",self.ui.text_fields)
        self.assertNotIn("entrada",self.ui.text_fields)
        self.ui.submit=True;dashboard.main()
        self.assertEqual(len(self.ui.session_state.cola),1)
        self.assertNotIn("texto",self.ui.session_state.cola[0])
        dashboard.main()
        self.assertEqual(len(self.ui.session_state.cola),1)
    def test_unprepared_pdf_cannot_be_submitted(self):
        self.ui.submit=True;dashboard.main()
        self.assertEqual(self.ui.session_state.cola,[])
    def test_page_change_invalidates_prepared_text(self):
        self.prepare();self.ui.session_state.pdf_first=2;self.ui.submit=True
        self.ui.text_fields=[];dashboard.main()
        self.assertNotIn("pdf_review",self.ui.text_fields)
        self.assertEqual(self.ui.session_state.cola,[])
    def test_new_file_invalidates_previous_preview_even_without_callback(self):
        self.prepare()
        self.ui.uploaded=Uploaded(native_pdf(["Documento distinto. "+SENTENCE]))
        self.ui.submit=True;dashboard.main()
        self.assertEqual(self.ui.session_state.cola,[])
    def test_failed_extraction_cannot_reuse_previous_text(self):
        self.prepare();self.ui.extract=True;self.ui.submit=True
        with patch("app.document_reader.extract_pdf",side_effect=ValueError("Error de prueba")):
            dashboard.main()
        self.assertNotIn("pdf_prepared_for",self.ui.session_state)
        self.assertEqual(self.ui.session_state.cola,[])
    def test_edited_preview_is_the_text_submitted(self):
        self.prepare();self.ui.session_state.pdf_review="Ejemplo inventado divorcio pension alimenticia";self.ui.submit=True
        with patch("app.dashboard.demo_classify",wraps=dashboard.demo_classify) as spy:
            dashboard.main()
        self.assertEqual(spy.call_args.args[0],self.ui.session_state.pdf_review)
    def test_txt_still_supported_and_rtf_rejected(self):
        self.ui.uploaded=Uploaded(SENTENCE.encode(),name="ejemplo.txt");self.ui.submit=True
        dashboard.main();self.assertEqual(len(self.ui.session_state.cola),1)
        self.ui.uploaded=Uploaded(b"{\\rtf1 texto enriquecido}",name="ejemplo.txt")
        dashboard.main();self.assertEqual(len(self.ui.session_state.cola),1)
        self.assertTrue(any(kind=="error" and "enriquecido" in text for kind,text in self.ui.messages))


if __name__=="__main__": unittest.main()
