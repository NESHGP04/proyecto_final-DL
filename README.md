# proyecto_final-DL

Clasificación y enrutamiento automático de casos entrantes de un bufete de abogados en Guatemala por área legal (laboral, civil, mercantil, penal, familia, otros), con modelos de deep learning propios, transfer learning y fine-tuning, y un modelo de valor que traduce las métricas técnicas en valor de negocio. CC3092 Deep Learning y Sistemas Inteligentes, UVG.

Autoras: Camila Richter (23183) y Marinés García (23391).

**Estado:** en desarrollo. El código está completo y probado con datos sintéticos; las métricas reportables esperan el conjunto de datos real del bufete (confidencial, no está en este repo).

## Qué se compara

| ID | Modelo | Papel |
|---|---|---|
| M0 | TF-IDF + SVM lineal | Referencia sin deep learning |
| M1 | Embeddings desde cero + CNN 1D | Modelo propio, sin preentrenamiento |
| M2 | BETO congelado + media de tokens + capa lineal | Modelo base de transfer learning |
| M3 | M2 con el encoder descongelado | Fine-tuning |
| M4 | M3 precedido de preentrenamiento MLM sobre texto legal | Adaptación de dominio (opcional) |

M2 y M3 comparten arquitectura: el delta M2 → M3 mide solo el efecto del fine-tuning. BETO es `dccuchile/bert-base-spanish-wwm-cased` (CC BY 4.0 según su ficha, con una reserva sobre los datos con que se entrenó para uso comercial).

No se usa ningún LLM por API ni modelo generativo.

## Instalación

Requiere Python 3.12. Las versiones de `requirements.txt` están verificadas en un Mac con Python x86_64 (torch 2.2.2 es la última versión para ese build); con Python arm64 nativo o en Colab se pueden usar versiones más nuevas de torch y transformers.

```bash
python3.12 -m venv venv-final
source venv-final/bin/activate
pip install -r requirements.txt
```

Verificar que todo funciona:

```bash
pytest
```

Los tests usan datos sintéticos. Los que cargan BETO descargan unos 440 MB la primera vez a la caché de Hugging Face.

## Datos

Los datos del bufete son confidenciales y no se versionan (`data/` y `artifacts/` están en `.gitignore`). El código espera tres archivos UTF-8 en `data/processed/`: `train.csv`, `val.csv` y `test.csv`, con las columnas:

| Columna | Descripción |
|---|---|
| `case_id` | Identificador anónimo, ej. `C0001` |
| `texto` | Texto ya anonimizado |
| `area_legal` | Una de las áreas de `configs/config.yaml` |
| `canal` | `correo`, `formulario`, `documento` o `fisico` |
| `fecha` | `YYYY-MM-DD` o vacío |
| `fuente_texto` | `nativo` u `ocr` |

`src/data/schema.py` valida este formato al cargar. El `test.csv` está cerrado: `load_split` lo rechaza salvo en la evaluación final.

Para probar el código sin datos reales, `src/data/synthetic.py` genera casos inventados. Ningún resultado que se reporte puede salir de ellos.

## Reproducir las métricas

Las áreas, semillas, rutas e hiperparámetros están en `configs/config.yaml`; los parámetros del modelo de valor, en `configs/value_params.yaml` (cada uno con valor, rango y fuente).

1. Entrenar y evaluar todos los modelos en validación, con 3 semillas:

   ```bash
   python -m src.reproduce --stage val --models m0 m1 m2 m3 --seeds 42 43 44
   ```

2. Con la configuración ya congelada, evaluar en test una sola vez:

   ```bash
   python -m src.reproduce --stage final --confirm-frozen --models m0 m1 m2 m3 --seeds 42 43 44
   ```

   El umbral de confianza τ se elige con las predicciones de validación y se aplica a test. Si ya existen predicciones de test, el comando se niega a correr (`--again` lo permite; hay que declararlo en el informe).

Ambos escriben `experiments/results/summary_<etapa>.md` con F1 macro e intervalo de confianza, exactitud, recall por clase, valor mensual, deltas entre versiones con intervalo pareado y costos.

M3 tarda mucho en un Mac; hay un cuaderno para Colab con GPU en `notebooks/colab_m3.ipynb`.

Otras piezas, cada una con su comando:

```bash
python -m src.models.m3_bert_finetune --split val --save    # guarda pesos en artifacts/ (para la API)
python -m src.models.m2_bert_frozen --split val --save      # guarda la cabeza (para explicabilidad)
python -m src.evaluation.learning_curve --model m1 --seeds 42 43 44
python -m src.explain.explain --split val --n 3 --models m2 m3
```

## Aplicación

Con los pesos de M3 guardados:

```bash
uvicorn app.api:app
```

| Ruta | Qué hace |
|---|---|
| `POST /classify` | `{"texto": "..."}` → área, confianza y acción (`enrutar` si la confianza ≥ τ, `revision` si no) |
| `POST /classify_batch` | Lo mismo para varios textos |
| `GET /health` | Estado y modelo cargado |
| `GET /stats` | Conteos y latencias (insumo del monitoreo) |

La API no guarda el texto de los casos. El registro de monitoreo (`artifacts/api_log.csv`) tiene solo hora, área, confianza, acción y latencia. Latencia y throughput se miden con `python -m app.benchmark --texts-file <archivo>`; el costo por predicción solo se calcula si se pasa `--cost-per-hour`.

## Estructura

```
configs/              configuración y parámetros del modelo de valor
data/                 datos del bufete (no versionados)
src/data/             esquema, anonimización, OCR, split, datos sintéticos
src/models/           M0 a M4
src/evaluation/       métricas, comparación, errores, curvas, curva de aprendizaje
src/value/            modelo de valor y sensibilidad
src/explain/          gradientes integrados
src/reproduce.py      reproduce las tablas del informe
app/                  API, predictor, benchmark y dashboard
experiments/results/  predicciones y resúmenes
artifacts/            pesos entrenados (no versionados)
tests/
```

## Limitaciones

- Las métricas reales dependen de los datos del bufete; este repo no los incluye.
- La anonimización por reglas (`src/data/anonymize.py`) es heurística y no garantiza quitar todo dato personal.
- Los pesos entrenados con datos del bufete no se publican: un modelo con fine-tuning puede memorizar texto de entrenamiento.
- El modelo de valor usa parámetros estimados con el bufete o marcados como supuesto en `value_params.yaml`.
