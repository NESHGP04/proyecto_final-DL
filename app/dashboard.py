"""Dashboard CP5. Ejecutar desde la raíz: streamlit run app/dashboard.py.

La demo usa reglas ilustrativas, no un modelo entrenado. El modo API llama al
servicio local de Nes. La cola y el CSV de correcciones no contienen texto.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import secrets
import threading
import unicodedata
from urllib.parse import urlsplit
from uuid import uuid4
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOG_FIELDS = ["hora_utc", "case_id", "origen", "area_modelo", "area_corregida", "confianza"]
LOG_LOCK = threading.Lock()
DEMO_EXAMPLES = {
    "Laboral": "Ejemplo ficticio: solicito orientación por un despido y el pago de mi salario pendiente.",
    "Familia": "Ejemplo ficticio: necesito orientación sobre divorcio y pensión alimenticia.",
    "Ambiguo": "Ejemplo ficticio: tengo un problema y quisiera saber con quién puedo hablar.",
}
KEYWORDS = {
    "laboral": ("despido", "salario", "trabajador", "empleador"),
    "civil": ("arrendamiento", "herencia", "propiedad", "danos"),
    "mercantil": ("sociedad", "accionista", "comercio", "mercantil"),
    "penal": ("delito", "robo", "denuncia", "estafa"),
    "familia": ("divorcio", "pension", "custodia", "alimenticia"),
    "otros": ("tributario", "impuesto", "administrativo", "constitucional"),
}

# Estilo visual adaptado de la referencia de Cami. No contiene datos del ejemplo.
DASHBOARD_STYLE = """
<style>
  .block-container {
    max-width: 1440px; padding-top: 5.5rem; padding-bottom: 3rem;
  }
  h1 {
    font-size: 2.15rem !important; font-weight: 700 !important;
    letter-spacing: -.035em;
  }
  h2, h3 {letter-spacing: -.02em;}
  [data-testid="stSidebar"] {border-right: 1px solid rgba(120,138,158,.2);}
  [data-testid="stSidebar"] .stButton {margin-bottom: .28rem;}
  [data-testid="stSidebar"] .stButton button {
    min-height: 3.15rem; border-radius: 10px; padding: .7rem .85rem;
    text-align: left; justify-content: flex-start;
    border: 1px solid rgba(120,138,158,.2);
  }
  [data-testid="stSidebar"] .stButton button p {
    text-align: left; font-size: .91rem;
  }
  [data-testid="stMetric"] {
    background: var(--secondary-background-color, #f3f6f9);
    border: 1px solid rgba(120,138,158,.22); border-top: 3px solid #39728d;
    border-radius: 12px; padding: 1.05rem 1.1rem; min-height: 150px;
    box-shadow: 0 3px 10px rgba(20,39,57,.035); height: 100%;
  }
  [data-testid="stMetricLabel"] {min-height: 2.5rem; align-items: flex-start;}
  [data-testid="stMetricLabel"] p {
    font-size: .86rem; white-space: normal; line-height: 1.35;
  }
  [data-testid="stMetricValue"] {
    font-size: clamp(1.35rem, 2.1vw, 2rem); font-weight: 650;
    letter-spacing: -.03em;
  }
  [data-testid="stMetricDelta"] {font-size: .83rem;}
  [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"] {
    border-radius: 12px;
  }
  [data-testid="stForm"] {border-color: rgba(120,138,158,.22);}
  [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border-radius: 10px; overflow: hidden;
  }
  [data-testid^="stBaseButton-primary"], button[kind="primary"] {
    background: #234761 !important; border-color: #234761 !important; color: white !important;
    border-radius: 10px; box-shadow: 0 3px 8px rgba(25,52,74,.15);
  }
  [data-testid^="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
    background: #2b5875 !important; border-color: #39728d !important; color: white !important;
  }
  [data-testid^="stBaseButton-primary"] p {color: white !important;}
  button:focus-visible {
    outline-color: #39728d !important; box-shadow: 0 0 0 2px rgba(57,114,141,.25) !important;
  }
  [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-download"] {
    border-radius: 10px; border-color: rgba(120,138,158,.3);
  }
  .pl-brand {
    font-size: 1.9rem; font-weight: 800; letter-spacing: -.055em;
    line-height: 1.15; margin-bottom: .65rem; white-space: nowrap;
  }
  .pl-kicker {
    font-size: .7rem; letter-spacing: .16em; text-transform: uppercase;
    opacity: .65; margin: 1.5rem 0 .7rem;
  }
  .pl-nav {display: grid; gap: .45rem; margin-bottom: 1.4rem;}
  .pl-nav a {
    display: block; min-height: 3.15rem; box-sizing: border-box;
    padding: .85rem; border: 1px solid rgba(120,138,158,.22);
    border-radius: 10px; color: inherit !important;
    text-decoration: none !important; font-size: .91rem; font-weight: 500;
  }
  .pl-nav a:first-child {
    background: #234761; border-color: #234761; color: white !important;
    box-shadow: 0 3px 8px rgba(25,52,74,.15);
  }
  .pl-nav a:hover {border-color: #39728d;}
  .pl-note {
    border-top: 1px solid rgba(120,138,158,.25); padding-top: 1rem;
    margin-top: 1.4rem; font-size: .87rem; line-height: 1.6;
  }
  .pl-anchor {scroll-margin-top: 5rem;}
  @media (max-width: 800px) {
    .block-container {padding-top: 5rem;}
    h1 {font-size: 1.7rem !important;}
    [data-testid="stMetric"] {min-height: 130px;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    .pl-brand {font-size: 1.65rem;}
  }
</style>
"""


def validate_text(text, max_chars):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ingresa un texto antes de clasificar.")
    if len(text) > max_chars:
        raise ValueError(f"El texto supera el límite de {max_chars:,} caracteres.")
    return text


def validate_response(result, areas):
    """Rechaza respuestas incompletas o inconsistentes antes de mostrar una decisión."""
    try:
        area = result["area_legal"]
        confidence = float(result["confianza"])
        threshold = float(result["umbral"])
        raw = result["probabilidades"]
        if set(raw) != set(areas) or area not in areas:
            raise ValueError
        probs = {a: float(raw[a]) for a in areas}
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values()):
            raise ValueError
        if not (math.isfinite(confidence) and math.isfinite(threshold)
                and 0 <= confidence <= 1 and 0 <= threshold <= 1):
            raise ValueError
        if not math.isclose(sum(probs.values()), 1, abs_tol=1e-5):
            raise ValueError
        if not math.isclose(probs[area], max(probs.values()), abs_tol=1e-5):
            raise ValueError
        if not math.isclose(confidence, probs[area], abs_tol=1e-5):
            raise ValueError
        action = "enrutar" if confidence >= threshold else "revision"
        if result["accion"] != action:
            raise ValueError
        return dict(area_legal=area, confianza=confidence, probabilidades=probs,
                    umbral=threshold, accion=action)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        raise ValueError("La API devolvió una respuesta incompatible. Revisa su contrato y configuración.") from None


def demo_classify(text, areas, threshold):
    normalized = "".join(c for c in unicodedata.normalize("NFD", text.lower())
                         if unicodedata.category(c) != "Mn")
    words = set(re.findall(r"\w+", normalized))
    scores = {a: .25 + 2 * sum(k in words for k in KEYWORDS.get(a, ())) for a in areas}
    total = sum(scores.values())
    probs = {a: s / total for a, s in scores.items()}
    area = max(probs, key=probs.get)
    confidence = probs[area]
    return validate_response(dict(area_legal=area, confianza=confidence, probabilidades=probs,
                                  umbral=threshold,
                                  accion="enrutar" if confidence >= threshold else "revision"), areas)


def classify_api(text, base_url, areas, client=None):
    import httpx
    parsed = urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("Usa la dirección local de la API, por ejemplo http://127.0.0.1:8000.")
    try:
        if client is None:
            with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as local:
                response = local.post(base_url.rstrip("/") + "/classify", json={"texto": text})
        else:
            response = client.post(base_url.rstrip("/") + "/classify", json={"texto": text})
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"La API respondió con HTTP {exc.response.status_code}. Revisa que el servicio esté listo.") from None
    except (httpx.RequestError, httpx.InvalidURL):
        raise ValueError("No se pudo completar la consulta. Comprueba que la API local esté ejecutándose.") from None
    try:
        payload = response.json()
    except ValueError:
        raise ValueError("La API no devolvió JSON válido.") from None
    return validate_response(payload, areas)


def queue_record(result, source):
    """Construye una lista explícita de campos: nunca incluye el texto del caso."""
    return dict(case_id=uuid4().hex[:12], hora_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                origen=source, area_modelo=result["area_legal"], confianza=result["confianza"],
                umbral=result["umbral"], accion=result["accion"], area_corregida="",
                hora_revision_utc="", estado="Pendiente")


def submission_token(text, source, settings, salt):
    """Huella privada por sesión para reconocer reenvíos consecutivos sin guardar texto."""
    payload = json.dumps([source, settings, text.strip().replace("\r\n", "\n")],
                         ensure_ascii=False).encode("utf-8")
    return hashlib.blake2b(payload, key=salt, digest_size=20).hexdigest()


def save_correction(record, corrected, areas, path):
    if corrected not in areas or record["area_modelo"] not in areas:
        raise ValueError("Selecciona un área válida.")
    if record["estado"] != "Pendiente":
        raise ValueError("Este caso ya fue revisado.")
    row = dict(hora_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               case_id=record["case_id"], origen=record["origen"], area_modelo=record["area_modelo"],
               area_corregida=corrected, confianza=record["confianza"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        new = not path.exists() or path.stat().st_size == 0
        if not new:
            with path.open(encoding="utf-8", newline="") as f:
                if next(csv.reader(f), None) != LOG_FIELDS:
                    raise ValueError("El registro existente tiene otro formato; conserva el archivo antes de continuar.")
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
    return dict(record, area_corregida=corrected, hora_revision_utc=row["hora_utc"], estado="Revisado")


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    import streamlit as st
    from src.config import load_config, path_of

    cfg = load_config()
    areas = cfg["areas"]
    max_chars = cfg["serving"]["max_text_chars"]
    st.set_page_config(page_title="Práctica Legal · Clasificación", page_icon="⚖️", layout="wide")
    st.markdown(DASHBOARD_STYLE, unsafe_allow_html=True)
    st.caption("PRÁCTICA LEGAL / CLASIFICACIÓN DE CASOS")
    st.title("Práctica Legal")
    st.caption("Área sugerida, revisión humana y registro de decisiones.")
    st.session_state.setdefault("cola", [])
    st.session_state.setdefault("ultimo", None)
    if "envio_salt" not in st.session_state:
        st.session_state["envio_salt"] = secrets.token_bytes(32)
    st.session_state.setdefault("modo_clasificacion", "Demo simulada")
    st.session_state.setdefault("tipo_entrada", "Pegar texto")

    def seleccionar_modo(value):
        st.session_state["modo_clasificacion"] = value

    def seleccionar_entrada(value):
        st.session_state["tipo_entrada"] = value

    with st.sidebar:
        st.markdown('<div class="pl-brand">Práctica Legal</div>', unsafe_allow_html=True)
        st.markdown("**Clasificación de casos legales**")
        st.markdown('<div class="pl-kicker">Espacio de trabajo</div>', unsafe_allow_html=True)
        st.markdown('''<nav class="pl-nav" aria-label="Secciones del dashboard">
          <a href="#nuevo-caso" target="_self">1. Nuevo caso</a>
          <a href="#ultimo-resultado" target="_self">2. Último resultado</a>
          <a href="#cola-de-casos" target="_self">3. Cola y revisión</a>
        </nav>''', unsafe_allow_html=True)
        st.header("Clasificación")
        st.caption("Modo")
        modes = st.columns(2)
        for col, label in zip(modes, ("Demo simulada", "API local")):
            col.button(label, key="modo_" + label, use_container_width=True,
                       type="primary" if st.session_state.modo_clasificacion == label else "secondary",
                       on_click=seleccionar_modo, args=(label,))
        mode = st.session_state.modo_clasificacion
        source = "demo" if mode == "Demo simulada" else "api"
        if source == "demo":
            st.warning("Simulación por palabras clave. No usa M0–M4 y sus cifras no son métricas del proyecto.")
            threshold = st.slider("Umbral de demostración", 0.0, 1.0,
                                  float(cfg["confidence_threshold"]), .01)
        else:
            base_url = st.text_input("Dirección de la API", "http://127.0.0.1:8000")
            st.caption("Se aplica el umbral devuelto por la API. Este panel no modifica su configuración.")
        st.caption("El enrutamiento se muestra como propuesta; este panel no envía casos a personas.")
        st.markdown('<div class="pl-note"><strong>Práctica Legal</strong><br>'
                    'Área sugerida, revisión humana y registro de decisiones.</div>',
                    unsafe_allow_html=True)
    st.markdown('<div id="nuevo-caso" class="pl-anchor"></div>', unsafe_allow_html=True)
    st.subheader("Nuevo caso")
    st.caption("Entrada")
    inputs = st.columns(2)
    for col, label in zip(inputs, ("Pegar texto", "Cargar archivo")):
        col.button(label, key="tipo_" + label, use_container_width=True,
                   type="primary" if st.session_state.tipo_entrada == label else "secondary",
                   on_click=seleccionar_entrada, args=(label,))
    kind = st.session_state.tipo_entrada
    if source == "demo" and kind == "Pegar texto":
        example = st.selectbox("Ejemplo inventado", list(DEMO_EXAMPLES))
        if st.button("Cargar ejemplo"):
            st.session_state["entrada"] = DEMO_EXAMPLES[example]
    text, uploaded = "", None
    with st.form("clasificar", clear_on_submit=False):
        if kind == "Pegar texto":
            text = st.text_area("Texto del caso", height=180, key="entrada", max_chars=max_chars)
        else:
            uploaded = st.file_uploader("Archivo TXT en UTF-8", type=["txt"])
        submit = st.form_submit_button("Clasificar y agregar a la cola", type="primary")
    st.caption("La cola no conserva el texto. El campo de entrada permanece visible hasta que lo limpies.")
    if st.button("Limpiar entrada"):
        # Se aplica antes de crear el widget en la próxima ejecución.
        st.session_state["limpiar_pendiente"] = True
        st.session_state.pop("ultimo_envio", None)
        st.rerun()
    if submit:
        try:
            if kind == "Cargar archivo":
                if uploaded is None:
                    raise ValueError("Selecciona un archivo TXT.")
                if uploaded.size > 4 * max_chars:
                    raise ValueError("El archivo supera el tamaño permitido.")
                try:
                    text = uploaded.getvalue().decode("utf-8-sig")
                except UnicodeDecodeError:
                    raise ValueError("El archivo debe estar codificado en UTF-8.") from None
            validate_text(text, max_chars)
            settings = threshold if source == "demo" else base_url.rstrip("/")
            token = submission_token(text, source, settings, st.session_state.envio_salt)
            previous = st.session_state.get("ultimo_envio", {})
            duplicate = (previous.get("token") == token
                         and any(r["case_id"] == previous.get("case_id") for r in st.session_state.cola))
            if duplicate:
                st.info(f"Este envío ya se agregó a la cola como caso {previous['case_id']}. No se creó otra copia.")
            else:
                with st.spinner("Clasificando…"):
                    result = (demo_classify(text, areas, threshold) if source == "demo"
                              else classify_api(text, base_url, areas))
                record = queue_record(result, source)
                st.session_state.cola.append(record)
                st.session_state.ultimo = dict(result=result, origen=source, case_id=record["case_id"])
                st.session_state["ultimo_envio"] = dict(token=token, case_id=record["case_id"])
                st.success(f"Caso {record['case_id']} clasificado y agregado a la cola correctamente.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    st.markdown('<div id="ultimo-resultado" class="pl-anchor"></div>', unsafe_allow_html=True)
    last = st.session_state.ultimo
    if last and last["origen"] == source:
        result = last["result"]
        st.subheader("Último resultado")
        a, b, c = st.columns(3)
        a.metric("Área sugerida", result["area_legal"].capitalize())
        b.metric("Confianza", f"{result['confianza']:.1%}")
        c.metric("Umbral aplicado", f"{result['umbral']:.2f}")
        st.info("Propuesta: enrutar al equipo de " + result["area_legal"] if result["accion"] == "enrutar"
                else "Acción: revisión humana por confianza insuficiente.")
        st.bar_chart({"Área": list(result["probabilidades"]),
                      "Probabilidad": list(result["probabilidades"].values())},
                     x="Área", y="Probabilidad", horizontal=True)
        st.caption(f"ID: {last['case_id']} · Origen: {last['origen']}")
    st.markdown('<div id="cola-de-casos" class="pl-anchor"></div>', unsafe_allow_html=True)
    st.subheader("Cola de la sesión")
    queue = [r for r in st.session_state.cola if r["origen"] == source]
    st.caption("Se separan los casos demo y API. La cola se pierde al cerrar la sesión; las revisiones se guardan en CSV.")
    if not queue:
        st.info("Todavía no hay casos en este modo.")
        return
    st.dataframe(queue, hide_index=True, use_container_width=True)
    pending = [r for r in queue if r["estado"] == "Pendiente"]
    if pending:
        with st.form("revisar"):
            selected = st.selectbox("Caso a revisar", [r["case_id"] for r in pending])
            corrected = st.selectbox("Área confirmada por la persona revisora", areas, index=None,
                                     placeholder="Selecciona el área después de revisar")
            confirm = st.form_submit_button("Guardar revisión")
        if confirm:
            if corrected is None:
                st.error("Selecciona el área confirmada.")
            else:
                original = next(r for r in pending if r["case_id"] == selected)
                try:
                    path = path_of(cfg, "artifacts") / f"dashboard_correcciones_{source}.csv"
                    updated = save_correction(original, corrected, areas, path)
                    st.session_state.cola = [updated if r["case_id"] == selected else r
                                            for r in st.session_state.cola]
                    st.rerun()
                except (ValueError, OSError):
                    st.error("No se pudo guardar la revisión. Comprueba el formato y los permisos del registro local.")
    reviewed = [r for r in queue if r["estado"] == "Revisado"]
    if reviewed:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)
        st.download_button("Descargar revisiones de esta sesión", buf.getvalue().encode("utf-8-sig"),
                           file_name=f"revisiones_{source}.csv", mime="text/csv")


if __name__ == "__main__":
    # El estado de widgets debe limpiarse antes de instanciarlos.
    import streamlit as st
    if st.session_state.pop("limpiar_pendiente", False):
        st.session_state["entrada"] = ""
    main()
