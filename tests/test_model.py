import numpy as np
from src.model import compute_metrics, make_model


def test_metrics_identical_forecast():
    y = np.array([100.0, 200.0, 300.0])
    m = compute_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0


def test_metrics_rmse_positive():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])
    m = compute_metrics(y_true, y_pred)
    assert m["rmse"] > 0
    assert m["mae"] > 0


def test_make_model_unknown():
    import pytest
    with pytest.raises(ValueError):
        make_model("nonexistent_model")