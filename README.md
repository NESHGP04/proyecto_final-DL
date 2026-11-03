# proyecto_final-DL

Clasificación y enrutamiento automático de casos entrantes de un bufete de abogados en Guatemala, con modelos de deep learning propios, transfer learning y fine-tuning, y un modelo de valor que traduce métricas técnicas en valor de negocio. CC3092 Deep Learning y Sistemas Inteligentes, UVG.

**Estado:** en desarrollo. Antes de trabajar lee [docs/CONTEXTO.md](docs/CONTEXTO.md); las tareas están en [docs/TAREAS_NES.md](docs/TAREAS_NES.md) y [docs/TAREAS_CAMI.md](docs/TAREAS_CAMI.md).

## Instalación

```bash
python -m venv venv-final
source venv-final/bin/activate
pip install -r requirements.txt
```

## Pruebas

```bash
pytest
```

## Modelos

| ID | Modelo |
|---|---|
| M0 | TF-IDF + SVM lineal (referencia) |
| M1 | Modelo propio desde cero (embeddings + CNN 1D) |
| M2 | BETO congelado + cabeza (modelo base de transfer learning) |
| M3 | BETO con fine-tuning |
| M4 | M3 + adaptación de dominio (opcional) |

```bash
python -m src.models.m0_tfidf_svm --split val
```

## Datos

Los datos del bufete son confidenciales y no están en este repo (`data/` está en `.gitignore`). El formato esperado está descrito en [docs/CONTEXTO.md](docs/CONTEXTO.md) sección 7. `src/data/synthetic.py` genera datos sintéticos solo para probar el código.

## Reproducir resultados

Pendiente (Nes, CP6): instrucciones completas para reproducir las métricas del informe.
