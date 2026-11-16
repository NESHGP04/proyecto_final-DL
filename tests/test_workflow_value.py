import math

import pandas as pd

from src.value.value_model import value, workflow_components, workflow_curve


def _preds():
    return pd.DataFrame({"y_true": ["a", "a"], "y_pred": ["a", "b"], "conf": [.9, .5]})


def _params(**changes):
    return dict(dict(N=10., t_m=9.5, t_a=4.325, t_r=0., c_h=1., e_0=0., c_err=0., C_op=0.), **changes)


def test_workflow_matches_original_without_assisted_or_review_time():
    params = _params(t_a=0., t_r=0., c_h=100., c_err=50., C_op=20., e_0=.03)
    for tau in (0., .7, 1.):
        assert math.isclose(workflow_components(_preds(), tau, params)["value"], value(_preds(), tau, params))


def test_time_capacity_at_full_coverage():
    result = workflow_components(_preds(), 0, _params())
    assert math.isclose(result["net_minutes"], 51.75)
    assert math.isclose(result["operation_budget"], .8625)


def test_rejected_cases_have_review_overhead_even_when_none_accepted():
    result = workflow_components(_preds(), 1, _params(t_r=1., c_h=60., C_op=3.))
    assert result["kappa"] == 0 and result["net_error_exposure"] == 0
    assert result["net_minutes"] == -10 and result["value"] == -13


def test_partial_coverage_and_error_cost():
    params = _params(t_r=1., c_h=60., c_err=10.)
    result = workflow_components(_preds(), .7, params)
    assert math.isclose(result["net_minutes"], 20.875)
    assert result["net_error_exposure"] == 0
    assert math.isclose(workflow_components(_preds(), 0, params)["value"], 1.75)


def test_negative_savings_and_invalid_params():
    assert workflow_components(_preds(), 0, _params(t_a=12.))["value"] < 0
    for changes in ({"t_r": -1}, {"c_h": float("nan")}, {"e_0": 1.2}):
        try:
            workflow_components(_preds(), .7, _params(**changes))
        except ValueError:
            pass
        else:
            raise AssertionError("Debe rechazar parámetros inválidos")


def test_curve_includes_components_and_101_thresholds():
    curve = workflow_curve(_preds(), _params())
    assert len(curve)==101 and {"net_minutes", "net_error_exposure", "operation_budget", "value"} <= set(curve)
