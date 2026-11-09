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
