from src.evaluation.metrics import bootstrap_ci, compute_metrics, paired_bootstrap_delta

LABELS = ["a", "b"]
Y = ["a", "a", "a", "b", "b", "b"] * 10


def test_perfect_predictions():
    m = compute_metrics(Y, Y, LABELS)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["per_class"]["a"]["support"] == 30


def test_ci_contains_point_estimate():
    pred = list(Y)
    pred[0], pred[3] = "b", "a"
    point = compute_metrics(Y, pred, LABELS)["macro_f1"]
    low, high = bootstrap_ci(Y, pred, LABELS, n_boot=300)
    assert low <= point <= high


def test_paired_delta_positive_when_b_is_better():
    bad = ["a"] * len(Y)
    delta, low, high = paired_bootstrap_delta(Y, bad, Y, LABELS, n_boot=300)
    assert delta > 0 and low > 0
