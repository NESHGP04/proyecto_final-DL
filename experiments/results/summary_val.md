# Resumen de validación

Modelos: m0, m1, m2, m3, m4. Semillas: [42, 43, 44]. Casos evaluados: 2688.

| Modelo | F1 macro [IC 95 %] | Exactitud | F1 macro entre semillas |
|---|---|---|---|
| m0 | 0.781 [0.765, 0.796] | 0.781 | una semilla |
| m1 | 0.747 [0.730, 0.763] | 0.741 | 0.744 ± 0.002 (n=3) |
| m2 | 0.694 [0.678, 0.711] | 0.695 | 0.692 ± 0.001 (n=3) |
| m3 | 0.797 [0.782, 0.811] | 0.795 | 0.798 ± 0.002 (n=3) |
| m4 | 0.808 [0.793, 0.822] | 0.806 | 0.808 ± 0.000 (n=3) |

Recall por clase

| Modelo | laboral | civil | mercantil | penal | familia | otros |
|---|---|---|---|---|---|---|
| m0 | 0.73 | 0.62 | 0.79 | 0.85 | 0.96 | 0.73 |
| m1 | 0.69 | 0.69 | 0.66 | 0.77 | 0.97 | 0.66 |
| m2 | 0.68 | 0.49 | 0.76 | 0.78 | 0.87 | 0.60 |
| m3 | 0.76 | 0.64 | 0.84 | 0.82 | 0.96 | 0.74 |
| m4 | 0.78 | 0.67 | 0.81 | 0.85 | 0.98 | 0.75 |

Valor mensual con los parámetros de value_params.yaml (τ elegido con esta misma partición).

| Modelo | τ | Valor mensual |
|---|---|---|
| m0 | 0.66 | 967.3 |
| m1 | 0.83 | 868.3 |
| m2 | 0.85 | 629.4 |
| m3 | 0.90 | 1,159.6 |
| m4 | 0.97 | 1,234.6 |

Calibración por temperatura (T se ajusta con val; no cambia la clase predicha ni el F1). ECE = error de calibración (menor es mejor); el valor usa las probabilidades calibradas y su propio τ.

| Modelo | T | ECE sin calibrar | ECE calibrado | τ calibrado | Valor mensual calibrado |
|---|---|---|---|---|---|
| m0 | 0.79 | 0.082 | 0.016 | 0.76 | 950.2 |
| m1 | 1.25 | 0.045 | 0.045 | 0.73 | 866.1 |
| m2 | 1.22 | 0.041 | 0.025 | 0.75 | 621.4 |
| m3 | 1.41 | 0.084 | 0.018 | 0.77 | 1,164.4 |
| m4 | 1.71 | 0.114 | 0.032 | 0.84 | 1,257.9 |

Deltas de F1 macro (siguiente − anterior), IC pareado 95 %

| Comparación | Delta | IC | ¿Cruza 0? |
|---|---|---|---|
| m2 → m3 | +0.102 | [+0.086, +0.118] | no |
| m3 → m4 | +0.011 | [-0.000, +0.021] | sí |
| m1 → m3 | +0.050 | [+0.034, +0.066] | no |
| m0 → m1 | -0.034 | [-0.048, -0.020] | no |

Costos

| Modelo | trainable_params | train_seconds | total_train_seconds | head_train_seconds | embed_seconds | inference_ms_per_text | peak_memory_mb |
|---|---|---|---|---|---|---|---|
| m0 |  |  |  |  |  |  |  |
| m1 | 2715706 | 594.82 |  |  |  | 0.596 |  |
| m2 | 4614 |  |  | 6.51 | 745.93 | 48.79 | 3300 |
| m3 | 109855494 | 1002.72 |  |  |  | 2.19 | 3261 |
| m4 | 109855494 | 977.27 | 4314.76 |  |  | 1.93 | 4568 |

total_train_seconds (solo M4) = adaptación MLM + fine-tuning.

Nota: el valor usa un solo C_op para todos los modelos; ajustarlo por versión en value_params.yaml si se quiere comparar costo de cómputo.
