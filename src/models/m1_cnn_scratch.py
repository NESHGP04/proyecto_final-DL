"""M1: modelo propio entrenado desde cero (embeddings + CNN 1D, o BiLSTM con atención). Dueño: Nes (CP2).

Sin preentrenamiento. Pendiente (Nes): tokenizador propio, embeddings entrenados, CNN con filtros de
varios anchos (config.yaml, models.m1), pérdida con pesos por clase, curvas train/val, 3 semillas.
Debe terminar escribiendo experiments/results/m1_<split>_preds.csv con src.evaluation.preds.write_preds.
"""


def run(cfg, eval_split="val", final=False, seed=None, train_df=None):
    raise NotImplementedError("Nes, CP2")
