"""Curvas de aprendizaje (10/25/50/100 % de train). Dueño: Nes (CP4)."""
import numpy as np

FRACTIONS = (0.10, 0.25, 0.50, 1.00)


def stratified_subsample(df, frac, seed=42, label_col="area_legal"):
    """Submuestra estratificada de df; deja al menos 1 caso por clase."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in df.groupby(label_col):
        n = max(1, int(round(len(group) * frac)))
        keep.extend(rng.choice(group.index.to_numpy(), size=n, replace=False))
    return df.loc[sorted(keep)]


def run_learning_curve(model_runner, cfg, fractions=FRACTIONS):
    """Entrena `model_runner` (M1, M2 o M3) con cada fracción de train y evalúa en val.

    Pendiente (Nes): llamar a model_runner(cfg, train_df=subsample), guardar F1 macro por fracción
    y semilla, y graficar F1 vs. tamaño de datos.
    """
    raise NotImplementedError("Nes, CP4")
