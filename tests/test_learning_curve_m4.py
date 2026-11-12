import pandas as pd
import pytest
import torch

from src.config import load_config
from src.data.split import save_splits, split_dataframe
from src.data.synthetic import make_synthetic
from src.evaluation.learning_curve import plot_learning_curves, run_learning_curve, stratified_subsample
from src.evaluation.preds import write_preds


def _data(tmp_path):
    cfg = load_config()
    df = make_synthetic(cfg["areas"], n_per_area=40)
    save_splits(split_dataframe(df, cfg["split"]["ratios"], cfg["seed"]), tmp_path)
    return cfg


def test_subsample_is_stratified_and_nested_sizes():
    df = make_synthetic(["a", "b", "c"], n_per_area=40)
    for frac in (0.1, 0.5):
        sub = stratified_subsample(df, frac, seed=1)
        assert set(sub["area_legal"]) == {"a", "b", "c"}
        assert len(sub) == 3 * round(40 * frac)
    assert len(stratified_subsample(df.head(3), 0.01, seed=1)) >= 1


def test_learning_curve_with_fake_runner_and_plot(tmp_path):
    cfg = _data(tmp_path)
    labels = list(cfg["areas"])

    def fake_runner(cfg, split, seed, train_df, data_dir, out_dir, tag):
        val = pd.read_csv(f"{data_dir}/val.csv")
        probs = [[1.0 if a == t else 0.0 for a in labels] for t in val["area_legal"]]
        wrong = len(train_df) < 50  # simula que con pocos datos el modelo falla
        if wrong:
            probs = [[1.0] + [0.0] * (len(labels) - 1)] * len(val)
        path = out_dir / f"{tag}_val_preds.csv"
        write_preds(path, val["case_id"], val["area_legal"], probs, labels)
        return str(path)

    df = run_learning_curve(fake_runner, cfg, "fake", seeds=[1, 2], data_dir=tmp_path, out_dir=tmp_path / "res")
    assert len(df) == 4 * 2 and set(df["fraction"]) == {0.1, 0.25, 0.5, 1.0}
    small, full = df[df["fraction"] == 0.1]["val_macro_f1"], df[df["fraction"] == 1.0]["val_macro_f1"]
    assert small.max() < full.min()
    # volver a correr otro modelo no borra el anterior
    second = run_learning_curve(fake_runner, cfg, "fake2", fractions=(1.0,), data_dir=tmp_path, out_dir=tmp_path / "res")
    assert set(second["model"]) == {"fake2"} and len(second) == 1  # devuelve solo las filas de este modelo
    both = pd.read_csv(tmp_path / "res" / "learning_curve.csv")
    assert set(both["model"]) == {"fake", "fake2"}
    plot_learning_curves(both, tmp_path / "lc.png")
    assert (tmp_path / "lc.png").stat().st_size > 0


def test_learning_curve_reuses_done_points_and_main_run_for_100_percent(tmp_path):
    cfg = _data(tmp_path)
    labels = list(cfg["areas"])
    calls = []

    def runner(cfg, split, seed, train_df, data_dir, out_dir, tag):
        calls.append(tag)
        val = pd.read_csv(f"{data_dir}/val.csv")
        probs = [[1.0 if a == t else 0.0 for a in labels] for t in val["area_legal"]]
        path = out_dir / f"{tag}_val_preds.csv"
        write_preds(path, val["case_id"], val["area_legal"], probs, labels)
        return str(path)

    res = tmp_path / "res"
    res.mkdir()
    val = pd.read_csv(tmp_path / "val.csv")
    probs = [[1.0 if a == t else 0.0 for a in labels] for t in val["area_legal"]]
    write_preds(res / "fake_val_preds.csv", val["case_id"], val["area_legal"], probs, labels)  # corrida principal

    seed = cfg["seed"]
    run_learning_curve(runner, cfg, "fake", fractions=(0.5, 1.0), seeds=[seed], data_dir=tmp_path, out_dir=res)
    assert calls == [f"fake_f50_s{seed}"]          # el 100 % reutiliza la corrida principal
    run_learning_curve(runner, cfg, "fake", fractions=(0.5, 1.0), seeds=[seed], data_dir=tmp_path, out_dir=res)
    assert len(calls) == 1                         # reanudar no repite lo hecho
    run_learning_curve(runner, cfg, "fake", fractions=(0.5,), seeds=[seed], data_dir=tmp_path, out_dir=res, force=True)
    assert len(calls) == 2                         # --force sí repite


def test_corpus_texts_reads_paragraphs_and_skips_short_ones(tmp_path):
    from src.models.m4_bert_dapt import corpus_texts
    cfg = _data(tmp_path)
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "a.txt").write_text("Primer párrafo con varias palabras legales aquí.\n\ncorto\n\nSegundo párrafo también largo suficiente.", encoding="utf-8")
    texts = corpus_texts(cfg, include_train=False, external_dir=ext)
    assert len(texts) == 2
    with_train = corpus_texts(cfg, include_train=True, data_dir=tmp_path, external_dir=ext)
    assert len(with_train) > 2


def test_dapt_runs_offline_on_tiny_bert_and_saves_a_loadable_encoder(tmp_path):
    from transformers import AutoModel, BertConfig, BertForMaskedLM, BertTokenizerFast
    from src.models.m4_bert_dapt import dapt

    words = "el trabajador despido salario contrato demanda juez prueba ley caso de la y".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=True)
    base = tmp_path / "base"
    BertForMaskedLM(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=32, max_position_embeddings=32)).save_pretrained(base)
    tok.save_pretrained(base)

    cfg = load_config()
    texts = ["el trabajador y el contrato de la ley"] * 6 + ["despido de la demanda y el juez"] * 6
    info = dapt(cfg, texts, tmp_path / "out", overrides={"checkpoint": str(base), "dapt_epochs": 2,
                                                       "dapt_batch_size": 4, "dapt_max_len": 16})
    assert len(info["history"]) == 2 and info["mlm_head_pretrained"] is True
    assert AutoModel.from_pretrained(tmp_path / "out").config.hidden_size == 16


def _tiny_bert(tmp_path):
    from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
    words = "el trabajador despido salario contrato demanda juez prueba ley caso de la y".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=True)
    base = tmp_path / "base"
    BertForMaskedLM(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=32, max_position_embeddings=32)).save_pretrained(base)
    tok.save_pretrained(base)
    return base


def test_dapt_resumes_from_epoch_checkpoint_and_writes_output_only_when_complete(tmp_path):
    from src.models.m4_bert_dapt import dapt

    base = _tiny_bert(tmp_path)
    cfg = load_config()
    texts = ["el trabajador y el contrato de la ley"] * 6 + ["despido de la demanda y el juez"] * 6
    ov = {"checkpoint": str(base), "dapt_epochs": 3, "dapt_batch_size": 4, "dapt_max_len": 16}
    out, ck = tmp_path / "out", tmp_path / "ck"

    def crash_after_first(epoch):
        if epoch == 1:
            raise RuntimeError("se apagó Colab")

    with pytest.raises(RuntimeError):
        dapt(cfg, texts, out, overrides=ov, ckpt_dir=ck, after_epoch=crash_after_first)
    assert (ck / "state.pt").exists() and not out.exists()   # el encoder final no queda a medias
    info = dapt(cfg, texts, out, overrides=ov, ckpt_dir=ck)
    assert info["resumed_from_epoch"] == 1 and [h["epoch"] for h in info["history"]] == [1, 2, 3]
    assert (out / "config.json").exists() and not (tmp_path / "out.tmp").exists()
    with pytest.raises(ValueError):                           # un checkpoint de otra configuración no se reutiliza
        dapt(cfg, texts, tmp_path / "out2", overrides={**ov, "dapt_lr": 1e-3}, ckpt_dir=ck)


def test_build_corpus_drops_short_duplicate_and_leaky_texts(tmp_path):
    import gzip
    from src.data.dapt_corpus import build_corpus

    def txt(i):
        return " ".join(f"palabra{(i * 7 + j) % 40}x{i}" for j in range(30))

    held = [txt(100), txt(100), txt(101), txt(101)]               # val/test (repetidos para que TF-IDF los vea)
    near = txt(100).rsplit(" ", 1)[0] + " otra"                   # casi igual a un texto de val/test
    rows = [("a", "CIVIL", txt(1)), ("b", "CIVIL", txt(2)), ("c", "PENAL", txt(3)), ("d", "PENAL", txt(3)),   # duplicado
            ("e", "PENAL", "muy corto"), ("f", "CIVIL", txt(101)), ("g", "CIVIL", near)]
    src = tmp_path / "s.csv.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        pd.DataFrame(rows, columns=["id", "materia", "texto"]).to_csv(f, index=False)
    out, info = build_corpus(src, exclude_texts=held, near_dup_texts=held, n=10, seed=1)
    assert set(out["id"]) == {"a", "b", "c"}
    assert info["descartados_casi_duplicados_de_val_test"] == 1


def test_merge_curves_replaces_only_models_in_other_file(tmp_path):
    from src.evaluation.learning_curve import merge_curves
    row = {"fraction": 1.0, "n_train": 10, "seed": 1, "val_accuracy": 0.5}
    pd.DataFrame([{"model": "m1", "val_macro_f1": 0.5, **row}, {"model": "m3", "val_macro_f1": 0.1, **row}]).to_csv(
        tmp_path / "a.csv", index=False)
    pd.DataFrame([{"model": "m3", "val_macro_f1": 0.9, **row}]).to_csv(tmp_path / "b.csv", index=False)
    merged = merge_curves(tmp_path / "a.csv", tmp_path / "b.csv")
    assert dict(zip(merged["model"], merged["val_macro_f1"])) == {"m1": 0.5, "m3": 0.9}
    assert len(merged) == 2
