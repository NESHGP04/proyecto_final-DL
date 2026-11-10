import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import BertConfig, BertModel, BertTokenizerFast

from app.api import create_app
from app.benchmark import benchmark
from src.evaluation.errors import (calibration_table, confusion_matrix_df, error_report, group_metrics,
                                   low_confidence_report, top_confusions)
from src.explain.explain import (aggregate_by_class, explain_text, integrated_gradients, merge_wordpieces,
                                 overlap, top_words)
from src.models.m3_bert_finetune import BertClassifier

LABELS = ["a", "b", "c"]


def _preds():
    return pd.DataFrame({
        "case_id": [f"C{i}" for i in range(8)],
        "y_true": ["a", "a", "a", "b", "b", "c", "c", "c"],
        "y_pred": ["a", "a", "b", "b", "b", "c", "a", "c"],
        "conf":   [0.9, 0.8, 0.5, 0.95, 0.7, 0.9, 0.4, 0.85],
    })


# ---------- errores ----------
def test_confusion_and_top_confusions():
    cm = confusion_matrix_df(_preds(), LABELS)
    assert cm.loc["a", "b"] == 1 and cm.loc["c", "a"] == 1 and cm.values.sum() == 8
    assert top_confusions(cm, 1)[0][2] == 1


def test_group_metrics_and_low_confidence():
    meta = pd.DataFrame({"case_id": [f"C{i}" for i in range(8)],
                         "canal": ["correo"] * 4 + ["fisico"] * 4})
    g = group_metrics(_preds(), meta, LABELS, "canal", min_n=5)
    assert set(g["canal"]) == {"correo", "fisico"} and g["pocos_casos"].all()
    low = low_confidence_report(_preds(), 0.7)
    assert low["n_low"] == 2 and low["accuracy_low"] == 0.0 and low["accuracy_high"] > 0.8


def test_error_report_writes_files(tmp_path):
    meta = pd.DataFrame({"case_id": _preds()["case_id"], "canal": "correo", "fuente_texto": "nativo"})
    rep = error_report(_preds(), meta, LABELS, 0.7, tmp_path, "mX")
    assert (tmp_path / "mX_confusion.png").exists() and (tmp_path / "mX_por_canal.csv").exists()
    assert len(calibration_table(_preds())) >= 1 and rep["baja_confianza"]["n_low"] == 2


# ---------- explicabilidad ----------
def test_merge_wordpieces_and_helpers():
    pairs = merge_wordpieces(["[CLS]", "des", "##pido", "juez", "[SEP]"], [9, 0.2, 0.3, -0.1, 9])
    assert pairs == [("despido", pytest.approx(0.5)), ("juez", pytest.approx(-0.1))]
    assert top_words(pairs, 5) == ["despido"]
    assert overlap(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3) and overlap([], []) == 1.0
    agg = aggregate_by_class([("x", [("w", 1.0)]), ("x", [("w", 0.5)]), ("x", [("z", 1.0)])], min_count=2)
    assert agg == {"x": [("w", 0.75, 2)]}


def _tiny(tmp_path):
    words = "el trabajador despido salario contrato demanda juez prueba ley caso de la y".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"))
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=32)
    return tok, BertClassifier(BertModel(cfg), 3, "mean", 0.0).eval()


def test_integrated_gradients_completeness_and_forward_embeds_matches_forward(tmp_path):
    tok, model = _tiny(tmp_path)
    enc = tok("el trabajador de la demanda y el juez", return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        direct = model(ids, mask)
        via = model.forward_embeds(model.encoder.embeddings.word_embeddings(ids), mask)
    assert torch.allclose(direct, via, atol=1e-5)
    attr, err = integrated_gradients(model.forward_embeds, model.encoder.embeddings.word_embeddings,
                                     ids, mask, target=1, steps=64, pad_id=tok.pad_token_id)
    assert attr.shape == (ids.shape[1],) and err < 0.15


def test_explain_text_returns_words_normalized(tmp_path):
    tok, model = _tiny(tmp_path)
    cls, pairs, err = explain_text(model, tok, "despido del trabajador y salario", steps=16)
    assert 0 <= cls < 3
    assert [w for w, _ in pairs] == ["despido", "trabajador", "y", "salario"]  # "del" no está en el vocabulario ([UNK])
    assert max(abs(v) for _, v in pairs) == pytest.approx(1.0)


# ---------- API ----------
class FakePredictor:
    areas = ["laboral", "civil"]

    def classify(self, texts):
        return [{"area_legal": "laboral" if "despido" in t else "civil",
                 "confianza": 0.9 if "despido" in t else 0.5,
                 "probabilidades": {"laboral": 0.9, "civil": 0.1}} for t in texts]


def test_api_routes_review_and_logging_without_text(tmp_path):
    log = tmp_path / "log.csv"
    client = TestClient(create_app(FakePredictor(), threshold=0.7, max_chars=50, log_path=log))
    assert client.get("/health").json()["estado"] == "ok"
    r = client.post("/classify", json={"texto": "despido injustificado ayer"}).json()
    assert r["area_legal"] == "laboral" and r["accion"] == "enrutar" and r["latencia_ms"] >= 0
    r2 = client.post("/classify", json={"texto": "consulta general sobre un tema"}).json()
    assert r2["accion"] == "revision"
    batch = client.post("/classify_batch", json={"textos": ["despido", "otro"]}).json()
    assert [x["accion"] for x in batch["resultados"]] == ["enrutar", "revision"]
    assert client.post("/classify", json={"texto": "   "}).status_code == 422
    assert client.post("/classify", json={"texto": "x" * 51}).status_code == 413
    assert client.post("/classify_batch", json={"textos": []}).status_code == 422
    stats = client.get("/stats").json()
    assert stats["predicciones"] == 4 and stats["tasa_revision"] == 0.5
    logged = log.read_text(encoding="utf-8")
    assert "despido" not in logged and "laboral" in logged and logged.count("\n") == 5


def test_benchmark_reports_times_and_cost_only_if_given():
    res = benchmark(lambda texts: [None] * len(texts), ["a", "b"], n_single=10, batch_size=4)
    assert res["latencia_ms_p95"] >= res["latencia_ms_p50"] >= 0 and "costo_por_1000_predicciones" not in res
    res = benchmark(lambda texts: [None] * len(texts), ["a"], n_single=5, batch_size=2, cost_per_hour=1.0)
    assert res["costo_por_1000_predicciones"] > 0
