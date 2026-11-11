"""Curvas de aprendizaje (10/25/50/100 % de train) para M1, M2 y M3.

Para cada fracción y semilla se entrena el modelo con una submuestra estratificada de train y se
mide el F1 macro en val. Muestra cuándo el transfer learning ayuda (pocos datos) y cuándo deja de
ayudar. Con M3 cada punto es un fine-tuning completo: conviene 1 semilla, o correrlo en Colab.

Los puntos ya hechos se reutilizan (se puede reanudar), y el 100 % reusa la corrida principal del modelo.

Uso: python -m src.evaluation.learning_curve --model m1 --seeds 42 43 44
     python -m src.evaluation.learning_curve --plot-only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, path_of
from src.data.dataset import load_split
from src.evaluation.metrics import compute_metrics
from src.evaluation.preds import read_preds

FRACTIONS = (0.10, 0.25, 0.50, 1.00)
CSV_NAME = "learning_curve.csv"


def stratified_subsample(df, frac, seed=42, label_col="area_legal"):
    """Submuestra estratificada de df; deja al menos 1 caso por clase."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in df.groupby(label_col):
        n = max(1, int(round(len(group) * frac)))
        keep.extend(rng.choice(group.index.to_numpy(), size=n, replace=False))
    return df.loc[sorted(keep)]


def _existing(results, runs_dir, model_name, tag, frac, seed, main_seed, force):
    """Predicciones ya hechas que se pueden reutilizar: este mismo punto (si se cortó la corrida) o,
    para el 100 %, la corrida principal del modelo (así la curva coincide con la tabla del informe)."""
    if force:
        return None
    candidates = [runs_dir / f"{tag}_val_preds.csv"]
    if frac >= 1.0:
        main = model_name if seed == main_seed else f"{model_name}_s{seed}"
        candidates.append(results / f"{main}_val_preds.csv")
    return next((str(c) for c in candidates if c.exists()), None)


def run_learning_curve(model_runner, cfg, model_name, fractions=FRACTIONS, seeds=None, data_dir=None,
                       out_dir=None, force=False):
    """Entrena `model_runner` con cada fracción de train y evalúa en val. Devuelve un DataFrame con las filas de este modelo.

    model_runner(cfg, "val", seed=, train_df=, data_dir=, out_dir=, tag=) debe escribir las
    predicciones de val (M1, M2 y M3 lo cumplen). Los resultados se guardan en
    experiments/results/learning_curve.csv (se reemplazan solo las filas de este modelo).
    """
    seeds = list(seeds) if seeds else [cfg["seed"]]
    results = Path(out_dir or path_of(cfg, "results"))
    runs_dir = results / f"lc_{model_name}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(cfg, "train", data_dir=data_dir)
    labels = list(cfg["areas"])

    rows = []
    for frac in fractions:
        for seed in seeds:
            sub = train if frac >= 1.0 else stratified_subsample(train, frac, seed=seed)
            tag = f"{model_name}_f{int(round(frac * 100))}_s{seed}"
            path = _existing(results, runs_dir, model_name, tag, frac, seed, cfg["seed"], force)
            if path is None:
                path = model_runner(cfg, "val", seed=seed, train_df=sub, data_dir=data_dir, out_dir=runs_dir, tag=tag)
            preds = read_preds(path)
            m = compute_metrics(preds["y_true"], preds["y_pred"], labels)
            rows.append({"model": model_name, "fraction": frac, "n_train": len(sub), "seed": seed,
                         "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"]})
    this_model = pd.DataFrame(rows)
    csv = results / CSV_NAME
    merged = this_model
    if csv.exists():
        old = pd.read_csv(csv)
        merged = pd.concat([old[old["model"] != model_name], this_model], ignore_index=True)
    merged.to_csv(csv, index=False)
    return this_model


def merge_curves(csv, other):
    """Incorpora al CSV las filas de los modelos que trae `other` (reemplaza las de esos modelos)."""
    new = pd.read_csv(other)
    base = pd.read_csv(csv) if Path(csv).exists() else new.iloc[0:0]
    merged = pd.concat([base[~base["model"].isin(new["model"])], new], ignore_index=True)
    merged.to_csv(csv, index=False)
    return merged


def plot_learning_curves(df, path):
    """F1 macro en val vs. número de casos de entrenamiento: media ± desvío entre semillas por modelo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for model, g in df.groupby("model"):
        agg = g.groupby("n_train")["val_macro_f1"].agg(["mean", "std"]).fillna(0).reset_index()
        ax.plot(agg["n_train"], agg["mean"], marker="o", label=model)
        ax.fill_between(agg["n_train"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.2)
    ax.set(xlabel="casos de entrenamiento (escala log)", ylabel="F1 macro en val", title="Curva de aprendizaje")
    ax.set_xscale("log")
    sizes = sorted(df["n_train"].unique())
    ax.set_xticks(sizes, [f"{n:,}".replace(",", ".") for n in sizes])
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    lo, hi = df["val_macro_f1"].min(), df["val_macro_f1"].max()
    ax.set_ylim(max(0, lo - 0.05), min(1.0, hi + 0.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _runner(name):
    # import tardío: evita cargar torch para quien solo usa stratified_subsample
    if name == "m1":
        from src.models.m1_cnn_scratch import run
    elif name == "m2":
        from src.models.m2_bert_frozen import run
    elif name == "m3":
        from src.models.m3_bert_finetune import run
    else:
        raise ValueError(f"modelo desconocido: {name}")
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["m1", "m2", "m3"])
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--fractions", type=float, nargs="+", help="por defecto 0.1 0.25 0.5 1.0")
    ap.add_argument("--force", action="store_true", help="repetir puntos ya hechos")
    ap.add_argument("--merge", help="otro learning_curve.csv (p. ej. el de Colab con M3) cuyas filas se incorporan")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    results = path_of(cfg, "results")
    if not args.plot_only:
        run_learning_curve(_runner(args.model), cfg, args.model, seeds=args.seeds, force=args.force,
                           fractions=tuple(args.fractions) if args.fractions else FRACTIONS)
    if args.merge:
        merge_curves(results / CSV_NAME, args.merge)
    plot_learning_curves(pd.read_csv(results / CSV_NAME), results / "learning_curve.png")
    print(results / "learning_curve.png")
