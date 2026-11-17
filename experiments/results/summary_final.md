# Resumen final (test)

Modelos: m0, m1, m2, m3. Semillas: [42, 43, 44]. Casos evaluados: 2692.

| Modelo | F1 macro [IC 95 %] | Exactitud | F1 macro entre semillas |
|---|---|---|---|
| m0 | 0.785 [0.770, 0.800] | 0.785 | una semilla |
| m1 | 0.740 [0.726, 0.755] | 0.735 | 0.736 ± 0.003 (n=3) |
| m2 | 0.695 [0.679, 0.711] | 0.694 | 0.691 ± 0.003 (n=3) |
| m3 | 0.797 [0.782, 0.811] | 0.795 | 0.795 ± 0.001 (n=3) |

Recall por clase

| Modelo | laboral | civil | mercantil | penal | familia | otros |
|---|---|---|---|---|---|---|
| m0 | 0.77 | 0.62 | 0.78 | 0.84 | 0.98 | 0.71 |
| m1 | 0.69 | 0.69 | 0.65 | 0.73 | 0.99 | 0.67 |
| m2 | 0.70 | 0.51 | 0.72 | 0.77 | 0.87 | 0.61 |
| m3 | 0.79 | 0.69 | 0.80 | 0.85 | 0.96 | 0.69 |

Valor mensual con los parámetros de value_params.yaml (τ elegido con val).

| Modelo | τ | Valor mensual |
|---|---|---|
| m0 | 0.66 | 1,144.8 |
| m1 | 0.83 | 921.5 |
| m2 | 0.85 | 621.6 |
| m3 | 0.90 | 1,149.5 |

Calibración por temperatura (T se ajusta con val; no cambia la clase predicha ni el F1). ECE = error de calibración (menor es mejor); el valor usa las probabilidades calibradas y su propio τ.

| Modelo | T | ECE sin calibrar | ECE calibrado | τ calibrado | Valor mensual calibrado |
|---|---|---|---|---|---|
| m0 | 0.79 | 0.092 | 0.026 | 0.76 | 1,117.6 |
| m1 | 1.25 | 0.035 | 0.037 | 0.73 | 926.7 |
| m2 | 1.22 | 0.047 | 0.024 | 0.75 | 625.7 |
| m3 | 1.41 | 0.083 | 0.018 | 0.77 | 1,147.8 |

Deltas de F1 macro (siguiente − anterior), IC pareado 95 %

| Comparación | Delta | IC | ¿Cruza 0? |
|---|---|---|---|
| m2 → m3 | +0.102 | [+0.084, +0.116] | no |
| m1 → m3 | +0.056 | [+0.040, +0.072] | no |
| m0 → m1 | -0.045 | [-0.060, -0.030] | no |

Costos

| Modelo | trainable_params | train_seconds | total_train_seconds | head_train_seconds | embed_seconds | inference_ms_per_text | peak_memory_mb |
|---|---|---|---|---|---|---|---|
| m0 |  | 5.57 |  |  |  | 0.177 | 688 |
| m1 | 2715706 | 949.04 |  |  |  | 1.728 | 1253 |
| m2 | 4614 |  |  | 6.75 | 216.6 | 12.05 | 760 |
| m3 | 109855494 | 1154.77 |  |  |  | 2.08 | 3249 |

total_train_seconds (solo M4) = adaptación MLM + fine-tuning.

Nota: el valor usa un solo C_op para todos los modelos; ajustarlo por versión en value_params.yaml si se quiere comparar costo de cómputo.
