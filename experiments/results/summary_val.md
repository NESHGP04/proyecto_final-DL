# Resumen de validación

Modelos: m0, m1, m2. Semillas: [42, 43, 44]. Casos evaluados: 2688.

| Modelo | F1 macro [IC 95 %] | Exactitud | F1 macro entre semillas |
|---|---|---|---|
| m0 | 0.781 [0.765, 0.796] | 0.781 | una semilla |
| m1 | 0.747 [0.730, 0.763] | 0.741 | 0.744 ± 0.002 (n=3) |
| m2 | 0.694 [0.678, 0.711] | 0.695 | 0.692 ± 0.001 (n=3) |

Recall por clase

| Modelo | laboral | civil | mercantil | penal | familia | otros |
|---|---|---|---|---|---|---|
| m0 | 0.73 | 0.62 | 0.79 | 0.85 | 0.96 | 0.73 |
| m1 | 0.69 | 0.69 | 0.66 | 0.77 | 0.97 | 0.66 |
| m2 | 0.68 | 0.49 | 0.76 | 0.78 | 0.87 | 0.60 |

Valor mensual con los parámetros de value_params.yaml (τ elegido con esta misma partición).

| Modelo | τ | Valor mensual |
|---|---|---|
| m0 | 0.66 | 967.3 |
| m1 | 0.83 | 868.3 |
| m2 | 0.85 | 629.4 |

Deltas de F1 macro (siguiente − anterior), IC pareado 95 %

| Comparación | Delta | IC | ¿Cruza 0? |
|---|---|---|---|
| m0 → m1 | -0.034 | [-0.048, -0.020] | no |

Costos

| Modelo | trainable_params | train_seconds | head_train_seconds | embed_seconds | inference_ms_per_text | peak_memory_mb |
|---|---|---|---|---|---|---|
| m0 |  |  |  |  |  |  |
| m1 | 2715706 | 594.82 |  |  | 0.596 |  |
| m2 | 4614 |  | 7.12 | 745.93 | 48.79 | 3300 |

Nota: el valor usa un solo C_op para todos los modelos; ajustarlo por versión en value_params.yaml si se quiere comparar costo de cómputo.
