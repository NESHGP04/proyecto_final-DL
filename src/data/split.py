"""División train/val/test estratificada por área. Dueña: Cami (CP2)."""
import numpy as np
import pandas as pd


def _counts(n, ratios):
    """Cuántos casos van a val y a test de una clase de tamaño n (el resto va a train)."""
    n_val = int(round(n * ratios["val"]))
    n_test = int(round(n * ratios["test"]))
    if n >= 3:  # que val y test tengan al menos un caso de la clase
        n_val, n_test = max(n_val, 1), max(n_test, 1)
    n_val, n_test = min(n_val, n), min(n_test, n - n_val)
    return n_val, n_test


def split_dataframe(df, ratios, seed=42, temporal=False, label_col="area_legal", date_col="fecha"):
    """Devuelve una copia de df con la columna `split` en {train, val, test}.

    Estratificado por `label_col`. Con temporal=True, dentro de cada clase test es lo más reciente
    y val lo siguiente, para imitar el uso real (requiere fechas).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = "train"
    for _, group in out.groupby(label_col):
        idx = group.index.to_numpy()
        if temporal:
            order = pd.to_datetime(group[date_col]).sort_values(kind="stable").index.to_numpy()
            idx = order[::-1]  # más reciente primero
        else:
            idx = rng.permutation(idx)
        n_val, n_test = _counts(len(idx), ratios)
        out.loc[idx[:n_test], "split"] = "test"
        out.loc[idx[n_test:n_test + n_val], "split"] = "val"
    return out


def save_splits(df, out_dir):
    """Escribe train.csv, val.csv y test.csv en out_dir (UTF-8)."""
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        part.to_csv(f"{out_dir}/{name}.csv", index=False, encoding="utf-8")
