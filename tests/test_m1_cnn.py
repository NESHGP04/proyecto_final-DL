import numpy as np
import torch

from src.config import load_config
from src.evaluation.curves import plot_history
from src.models.m1_cnn_scratch import PAD, UNK, TextCNN, build_vocab, encode, tokenize, train_model


def test_tokenize_lowercases_and_keeps_accents():
    assert tokenize("Despido, INDEMNIZACIÓN [PERSONA] 2025!") == ["despido", "indemnización", "persona", "2025"]


def test_vocab_uses_min_freq_and_reserves_special_ids():
    vocab = build_vocab(["casa casa perro", "casa gato"], min_freq=2)
    assert list(vocab) == ["casa"] and vocab["casa"] >= 2
    ids = encode(["casa perro", ""], vocab, max_len=4)
    assert ids.shape == (2, 4)
    assert ids[0].tolist() == [vocab["casa"], UNK, PAD, PAD] and ids[1].tolist() == [PAD] * 4


def test_encode_truncates_to_max_len():
    ids = encode(["a b c d e f"], {"a": 2}, max_len=3)
    assert ids.shape == (1, 3)


def test_cnn_output_shape_and_padding_is_ignored_by_embedding():
    model = TextCNN(vocab_size=50, n_classes=4, embed_dim=8, filters=5, kernel_sizes=(3, 4), dropout=0.0)
    assert model(torch.randint(2, 50, (6, 12))).shape == (6, 4)
    assert torch.all(model.embed.weight[PAD] == 0)


def _toy():
    rng = np.random.default_rng(0)
    words = {0: ["alfa", "beta", "gamma"], 1: ["delta", "epsilon", "zeta"], 2: ["eta", "theta", "iota"]}
    y = np.repeat(np.arange(3), 40)
    texts = [" ".join(rng.choice(words[c], 6)) for c in y]
    vocab = build_vocab(texts, min_freq=1)
    return encode(texts, vocab, 8), y, len(vocab) + 2


def test_train_model_learns_toy_task_and_is_reproducible():
    mc = {**load_config()["models"]["m1"], "embed_dim": 16, "filters": 8, "epochs": 15, "lr": 5e-3}
    ids, y, vs = _toy()
    model, hist, best = train_model(ids[::2], y[::2], ids[1::2], y[1::2], vs, 3, mc, seed=1)
    assert max(h["val_macro_f1"] for h in hist) > 0.9 and 1 <= best <= len(hist)
    assert {"train_loss", "val_loss", "train_macro_f1", "val_macro_f1"} <= set(hist[0])
    _, hist2, _ = train_model(ids[::2], y[::2], ids[1::2], y[1::2], vs, 3, mc, seed=1)
    assert hist == hist2


def test_early_stopping_stops_before_max_epochs():
    mc = {**load_config()["models"]["m1"], "embed_dim": 16, "filters": 8, "epochs": 40, "lr": 5e-3, "patience": 2}
    ids, y, vs = _toy()
    _, hist, _ = train_model(ids[::2], y[::2], ids[1::2], y[1::2], vs, 3, mc, seed=1)
    assert len(hist) < 40


def test_plot_history_writes_png(tmp_path):
    hist = [{"epoch": i, "train_loss": 1 / i, "val_loss": 1 / i, "train_macro_f1": 0.5, "val_macro_f1": 0.4}
            for i in range(1, 4)]
    plot_history(hist, tmp_path / "c.png", "t")
    assert (tmp_path / "c.png").stat().st_size > 0
