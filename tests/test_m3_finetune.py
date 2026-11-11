import json

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel

from src.config import load_config
from src.evaluation.compare import compare_models, to_markdown
from src.models.m3_bert_finetune import BertClassifier, build_param_groups, fit, predict_logits


def tiny_classifier(n_classes=3):
    cfg = BertConfig(vocab_size=60, hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=32)
    return BertClassifier(BertModel(cfg), n_classes, "mean", 0.0)


def test_param_groups_cover_each_param_once_and_lower_layers_learn_slower():
    model = tiny_classifier()
    groups = build_param_groups(model, lr=1e-3, layer_decay=0.5, weight_decay=0.01)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == len(list(model.parameters()))

    def lr_of(param):
        return next(g["lr"] for g in groups if any(p is param for p in g["params"]))

    layers = model.encoder.encoder.layer
    assert lr_of(model.head.weight) == pytest.approx(1e-3)
    assert lr_of(layers[2].output.dense.weight) == pytest.approx(1e-3 * 0.5)
    assert lr_of(layers[0].output.dense.weight) == pytest.approx(1e-3 * 0.5 ** 3)
    assert lr_of(model.encoder.embeddings.word_embeddings.weight) == pytest.approx(1e-3 * 0.5 ** 4)


def test_no_weight_decay_on_bias_and_layernorm():
    model = tiny_classifier()
    groups = build_param_groups(model, 1e-3, 0.9, 0.01)
    for g in groups:
        names = [n for n, p in model.named_parameters() if any(p is q for q in g["params"])]
        flags = {n.endswith("bias") or "LayerNorm" in n for n in names}
        assert len(flags) == 1 and flags.pop() == (g["weight_decay"] == 0.0)


def _toy(n_per=30, seq=8):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), n_per)
    ids = torch.as_tensor(np.stack([rng.integers(2 + 15 * c, 2 + 15 * (c + 1), seq) for c in y]))
    return ids, torch.ones_like(ids), y


def test_fit_learns_and_unfreezes_the_encoder():
    ids, mask, y = _toy()
    mc = {**load_config()["models"]["m3"], "lr": 3e-3, "epochs": 8, "batch_size": 8, "patience": 8}
    model = tiny_classifier()
    before = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    tr, va = slice(None, None, 2), slice(1, None, 2)
    hist, best = fit(model, (ids[tr], mask[tr]), y[tr], (ids[va], mask[va]), y[va], mc, "cpu", seed=1)
    assert max(h["val_macro_f1"] for h in hist) > 0.9 and 1 <= best <= len(hist)
    assert not torch.equal(before, model.encoder.embeddings.word_embeddings.weight)  # el encoder sí cambió
    assert predict_logits(model, ids[va], mask[va], "cpu").shape == (len(y[va]), 3)


def test_compare_models_reports_delta_and_checks_same_cases():
    labels = ["a", "b"]
    y = ["a", "b"] * 30
    good = pd.DataFrame({"case_id": range(60), "y_true": y, "y_pred": y})
    bad = good.assign(y_pred=["a"] * 60)
    res = compare_models(bad, good, labels, "M2", "M3", n_boot=200)
    assert res["delta_macro_f1"]["delta"] > 0 and res["delta_macro_f1"]["significant"]
    assert "M2 → M3" in to_markdown(res)
    with pytest.raises(ValueError):
        compare_models(good, good.iloc[:50], labels)


# ---------- checkpoints y reanudación ----------
class _Crash(Exception):
    pass


def _mc(**kw):
    return {**load_config()["models"]["m3"], "lr": 3e-3, "epochs": 4, "batch_size": 8, "patience": 10, **kw}


def _seeded_tiny():
    torch.manual_seed(0)
    return tiny_classifier()


def _fit(model, mc, ckpt=None, after=None, info=None):
    ids, mask, y = _toy()
    tr, va = slice(None, None, 2), slice(1, None, 2)
    return fit(model, (ids[tr], mask[tr]), y[tr], (ids[va], mask[va]), y[va], mc, "cpu", seed=1,
               ckpt_dir=ckpt, after_epoch=after, info=info)


def _same(hist_a, hist_b):
    keys = ("epoch", "train_loss", "val_loss", "val_macro_f1")
    return all(abs(a[k] - b[k]) < 1e-5 for a, b in zip(hist_a, hist_b) for k in keys) and len(hist_a) == len(hist_b)


def test_resume_after_crash_matches_uninterrupted_run(tmp_path):
    mc = _mc()
    model_a = _seeded_tiny()
    hist_a, best_a = _fit(model_a, mc, tmp_path / "a")

    def boom(epoch):
        if epoch == 2:
            raise _Crash

    with pytest.raises(_Crash):
        _fit(_seeded_tiny(), mc, tmp_path / "b", after=boom)
    assert (tmp_path / "b" / "state.pt").exists() and not list((tmp_path / "b").glob("*.tmp"))
    status = json.loads((tmp_path / "b" / "status.json").read_text(encoding="utf-8"))
    assert status["epoca"] == 2 and status["terminado"] is False

    info, model_c = {}, _seeded_tiny()
    hist_c, best_c = _fit(model_c, mc, tmp_path / "b", info=info)
    assert info["resumed_from_epoch"] == 2 and best_c == best_a and _same(hist_a, hist_c)
    for (k, a), (_, c) in zip(model_a.state_dict().items(), model_c.state_dict().items()):
        assert torch.allclose(a, c, atol=1e-5), k


def test_resume_of_a_finished_checkpoint_does_not_train_again(tmp_path):
    mc = _mc(epochs=3)
    hist, best = _fit(_seeded_tiny(), mc, tmp_path / "x")
    info = {}
    hist2, best2 = _fit(_seeded_tiny(), mc, tmp_path / "x", info=info)
    assert info["resumed_from_epoch"] == 3 and _same(hist, hist2) and best == best2


def test_checkpoint_from_another_configuration_is_rejected(tmp_path):
    _fit(_seeded_tiny(), _mc(epochs=2), tmp_path / "x")
    with pytest.raises(ValueError, match="otra configuración"):
        _fit(_seeded_tiny(), _mc(epochs=2, lr=1e-3), tmp_path / "x")


def test_run_skips_a_finished_run_without_loading_the_model(tmp_path):
    from src.data.make_splits import make_splits  # noqa: F401  (solo para dejar claro que usa splits en disco)
    from src.data.split import save_splits, split_dataframe
    from src.data.synthetic import make_synthetic
    from src.models.m3_bert_finetune import run

    cfg = load_config()
    data = tmp_path / "data"
    data.mkdir()
    df = make_synthetic(cfg["areas"], n_per_area=10)
    save_splits(split_dataframe(df, cfg["split"]["ratios"], cfg["seed"]), data)
    n_train = len(pd.read_csv(data / "train.csv"))
    res = tmp_path / "res"
    res.mkdir()
    (res / "m3_val_preds.csv").write_text("case_id\n", encoding="utf-8")
    (res / "m3_val_run.json").write_text(json.dumps({"n_train": n_train, "seed": cfg["seed"]}), encoding="utf-8")
    assert run(cfg, "val", data_dir=data, out_dir=res) == str(res / "m3_val_preds.csv")  # no tocó el modelo
