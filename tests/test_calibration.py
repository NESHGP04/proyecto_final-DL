import json

import numpy as np
import pytest
import torch

from src.evaluation.calibration import (apply_temperature, crossfit_ece, expected_calibration_error, fit_temperature,
                                        report)


def _overconfident(n=3000, k=4, seed=0):
    """Logits verdaderos con T=1; el modelo reporta logits x3 (sobreconfiado)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k)) * 1.5
    p_true = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=p) for p in p_true])
    return apply_temperature(p_true, 1 / 3), y


def test_temperature_recovers_overconfidence_and_keeps_predictions():
    probs, y = _overconfident()
    t = fit_temperature(probs, y)
    assert 2.4 < t < 3.6                                   # el sesgo introducido fue x3
    cal = apply_temperature(probs, t)
    assert (cal.argmax(1) == probs.argmax(1)).all()        # no cambia la clase predicha
    assert expected_calibration_error(cal, y) < expected_calibration_error(probs, y) / 3
    assert crossfit_ece(probs, y) < expected_calibration_error(probs, y) / 2
    assert np.allclose(apply_temperature(probs, 1.0), probs, atol=1e-9)


def test_report_on_preds_dataframe():
    import pandas as pd
    probs, y = _overconfident(n=600, k=3)
    labels = ["a", "b", "c"]
    df = pd.DataFrame(probs, columns=[f"prob_{lab}" for lab in labels])
    df["y_true"] = [labels[i] for i in y]
    rep = report(df, labels)
    assert rep["temperature"] > 1 and rep["ece_despues"] < rep["ece_antes"] and rep["n"] == 600


def test_predictor_divides_logits_by_temperature(tmp_path):
    from app.predictor import Predictor, load_temperature

    class Tok:
        def __call__(self, texts, **kw):
            n = len(texts)
            return type("E", (dict,), {"to": lambda self, d: self})(input_ids=torch.zeros(n, 2, dtype=torch.long),
                                                                    attention_mask=torch.ones(n, 2, dtype=torch.long))

    class Model(torch.nn.Module):
        def forward(self, ids, mask):
            return torch.tensor([[3.0, 0.0, 0.0]]).repeat(len(ids), 1)

    base = Predictor(Model(), Tok(), ["a", "b", "c"]).predict_proba(["x"])
    soft = Predictor(Model(), Tok(), ["a", "b", "c"], temperature=2.0).predict_proba(["x"])
    assert soft.max() < base.max() and soft.argmax() == base.argmax()
    assert np.allclose(soft.numpy(), apply_temperature(base.numpy(), 2.0), atol=1e-6)

    cfg = {"paths": {"artifacts": str(tmp_path), "root": str(tmp_path)}}
    assert load_temperature(cfg, "m3") == 1.0
    (tmp_path / "m3_temperature.json").write_text(json.dumps({"temperature": 1.4}), encoding="utf-8")
    assert load_temperature(cfg, "m3") == pytest.approx(1.4)
