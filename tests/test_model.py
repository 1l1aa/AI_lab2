"""Обязательные тесты (раздел 7 ТЗ ЛР№2). Часть 2: модели и метрики."""
import numpy as np
import pandas as pd
import pytest
from src.model import compute_metrics, build_estimator


def test_predictions_count_and_finiteness():
    """Пункт 3 ТЗ: число прогнозов = числу строк, все значения конечны."""
    rng = np.random.RandomState(0)
    n = 30
    X = pd.DataFrame({
        "Species": rng.choice(["Bream", "Perch", "Roach"], n),
        "Volume": rng.uniform(100, 3000, n),
        "Length3": rng.uniform(10, 50, n),
        "Height": rng.uniform(3, 18, n),
        "Width": rng.uniform(2, 8, n),
    })
    y = rng.uniform(10, 1000, n)

    cfg = {
        "model": "linear_regression",
        "features": ["Volume", "Length3", "Height", "Width"],
        "categorical": True,
        "scaling": True,
        "log_target": False,
    }
    pipe = build_estimator(cfg, seed=42).fit(X, y)
    y_pred = pipe.predict(X)

    assert len(y_pred) == n
    assert np.all(np.isfinite(y_pred))


def test_metrics_match_manual_calculation():
    """Пункт 4 ТЗ: метрики на фиксированном примере = ручному расчёту."""
    # y_true = [100, 200, 300]
    # y_pred = [110, 190, 310]
    # MSE = ((−10)² + 10² + (−10)²) / 3 = 100
    # MAE = (10 + 10 + 10) / 3 = 10
    # ȳ = 200, SS_res = 100+100+100 = 300, SS_tot = 10000+0+10000 = 20000
    # R² = 1 − 300/20000 = 0.985
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    m = compute_metrics(y_true, y_pred)
    assert m["mse"] == pytest.approx(100.0)
    assert m["mae"] == pytest.approx(10.0)
    assert m["r2"] == pytest.approx(0.985)

    # Граничный случай: идеальный прогноз
    m_ideal = compute_metrics(y_true, y_true)
    assert m_ideal["mse"] == 0.0
    assert m_ideal["mae"] == 0.0
    assert m_ideal["r2"] == 1.0


def test_saved_pipeline_reproduces_predictions(tmp_path):
    """Пункт 5 ТЗ: сохранённая цепочка после загрузки воспроизводит прогноз."""
    import joblib

    rng = np.random.RandomState(42)
    n = 40
    X = pd.DataFrame({
        "Species": rng.choice(["Bream", "Perch"], n),
        "Volume": rng.uniform(100, 3000, n),
        "Length3": rng.uniform(10, 50, n),
        "Height": rng.uniform(3, 18, n),
        "Width": rng.uniform(2, 8, n),
    })
    y = rng.uniform(10, 1000, n)

    cfg = {
        "model": "ridge",
        "features": ["Volume", "Length3", "Height", "Width"],
        "categorical": True,
        "scaling": True,
        "log_target": False,
        "tune_alpha": False,
    }
    pipe = build_estimator(cfg, seed=42).fit(X, y)
    preds_before = pipe.predict(X)

    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    pipe_loaded = joblib.load(path)
    preds_after = pipe_loaded.predict(X)

    assert np.allclose(preds_before, preds_after, atol=1e-6)