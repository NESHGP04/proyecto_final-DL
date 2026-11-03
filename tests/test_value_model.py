import pandas as pd

from src.value.value_model import coverage_precision, optimal_tau, sensitivity, value, value_curve

PARAMS = {"N": 100.0, "t_m": 6.0, "c_h": 60.0, "e_0": 0.10, "c_err": 50.0, "C_op": 10.0}


def _preds():
    return pd.DataFrame({
        "y_true": ["a"] * 4 + ["b"] * 4,
        "y_pred": ["a", "a", "a", "b", "b", "b", "b", "b"],
        "conf":   [0.9, 0.9, 0.9, 0.4, 0.9, 0.9, 0.9, 0.6],
    })


def test_coverage_precision():
    kappa, p = coverage_precision(_preds(), 0.8)
    assert kappa == 6 / 8
    assert p == 1.0


def test_value_matches_formula_by_hand():
    # tau=0.8: kappa=6/8, p=1. saved = 100*0.75*(6/60)*60 = 450; error_delta = 100*0.75*(0-0.10)*50 = -375
    assert abs(value(_preds(), 0.8, PARAMS) - (450 + 375 - 10)) < 1e-9


def test_no_auto_cases_costs_only_operation():
    assert value(_preds(), 1.1, PARAMS) == -10.0


def test_curve_and_optimal_tau():
    curve = value_curve(_preds(), PARAMS)
    tau, best = optimal_tau(curve)
    assert 0.0 <= tau <= 1.0 and best == curve["value"].max()


def test_sensitivity_shape():
    s = sensitivity(_preds(), PARAMS, 0.8)
    assert set(s["param"]) == set(PARAMS) and len(s) == len(PARAMS) * 5
