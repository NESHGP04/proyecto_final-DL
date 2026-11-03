"""M2: BETO congelado + cabeza lineal. MODELO BASE de transfer learning. Dueño: Nes (CP1).

El encoder no se entrena: solo la cabeza de clasificación sobre [CLS] o la media de tokens.
Registrar costo (tiempo, memoria). Escribe experiments/results/m2_<split>_preds.csv.
"""


def run(cfg, eval_split="val", final=False, seed=None, train_df=None):
    raise NotImplementedError("Nes, CP1")
