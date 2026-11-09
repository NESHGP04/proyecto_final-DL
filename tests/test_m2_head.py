import numpy as np
import torch

from src.models.m2_bert_frozen import pool, train_head


def test_pool_mean_ignores_padding_and_cls_takes_first():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert pool(hidden, mask, "mean").tolist() == [[2.0, 2.0]]
    assert pool(hidden, mask, "cls").tolist() == [[1.0, 1.0]]


def test_head_learns_separable_clusters():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 3, (3, 16))
    y = np.repeat(np.arange(3), 40)
    X = (centers[y] + rng.normal(0, 1, (len(y), 16))).astype(np.float32)
    head, history, best = train_head(X[::2], y[::2], X[1::2], y[1::2], 3, lr=0.05, epochs=15, seed=1)
    pred = head.predict_proba(X[1::2]).argmax(1)
    assert (pred == y[1::2]).mean() > 0.9
    assert 1 <= best <= 15 and len(history) == 15


def test_head_is_reproducible():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(2), 30)
    X = rng.normal(0, 1, (len(y), 8)).astype(np.float32) + y[:, None]
    a = train_head(X, y, X, y, 2, epochs=5, seed=3)[1]
    b = train_head(X, y, X, y, 2, epochs=5, seed=3)[1]
    assert a == b
