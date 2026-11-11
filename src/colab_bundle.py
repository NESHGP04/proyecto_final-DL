"""Empaqueta lo necesario para correr los modelos pesados en Colab: código, configuración, splits y
resultados previos. Genera artifacts/colab_bundle.zip (se sube a Google Drive).

Incluye: src/, configs/, requirements.txt, data/processed/train.csv y val.csv (+ splits_info.json) y
los resultados de val de M0 a M2 (predicciones y run.json), para poder compararlos con M3.
NO incluye: test.csv (cerrado), data/raw, data/interim (datos del bufete), pesos ni caches.
Si data/processed tuviera datos del bufete el empaquetado se niega: en Colab solo van datos públicos.

Uso: python -m src.colab_bundle
"""
import argparse
import zipfile
from pathlib import Path

from src.config import ROOT, load_config, path_of

TOP = "proyecto_final-DL"
CODE_DIRS = ("src", "configs")


def _files(cfg):
    """Lista de (ruta en disco, ruta dentro del zip relativa al proyecto)."""
    out = []
    for d in CODE_DIRS:
        out += [(p, p.relative_to(ROOT)) for p in sorted((ROOT / d).rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    out.append((ROOT / "requirements.txt", Path("requirements.txt")))
    processed = path_of(cfg, "processed")
    out += [(processed / n, Path("data/processed") / n) for n in ("train.csv", "val.csv", "splits_info.json")]
    results = path_of(cfg, "results")
    out += [(p, Path("experiments/results") / p.name)
            for p in sorted(results.glob("m[012]*_val_preds.csv")) + sorted(results.glob("m[012]*_val_run.json"))]
    return [(src, dst) for src, dst in out if src.exists()]


def build_bundle(out_path=None, cfg=None, allow_non_public=False):
    cfg = cfg or load_config()
    processed = path_of(cfg, "processed")
    if not allow_non_public:
        import pandas as pd
        for name in ("train.csv", "val.csv"):
            f = processed / name
            if f.exists():
                origen = pd.read_csv(f, usecols=lambda c: c in {"origen", "grupo_id"}, encoding="utf-8")
                if "origen" not in origen or set(origen["origen"].dropna()) - {"saij"}:
                    raise SystemExit(f"{name} no es solo de datos públicos (origen != saij): no se sube a Colab")
    out = Path(out_path or path_of(cfg, "artifacts") / "colab_bundle.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    files = _files(cfg)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for src, dst in files:
            z.write(src, f"{TOP}/{dst.as_posix()}")
    return out, [dst.as_posix() for _, dst in files]


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    path, names = build_bundle()
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB, {len(names)} archivos)")
