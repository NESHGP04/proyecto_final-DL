"""M3: BETO con fine-tuning completo. Técnica principal. Dueño: Nes (CP3).

Descongelar el encoder; lr bajo (config.yaml, models.m3), warmup, tasa menor en capas inferiores,
early stopping por F1 macro en val. Documentar delta M2 -> M3 con
src.evaluation.metrics.paired_bootstrap_delta, curvas y costo. Escribe m3_<split>_preds.csv.
"""


def run(cfg, eval_split="val", final=False, seed=None, train_df=None):
    raise NotImplementedError("Nes, CP3")
