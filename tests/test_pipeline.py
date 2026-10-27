import numpy as np
import pandas as pd
import joblib
import tempfile
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data import load_data, split_data
from src.preprocess import build_preprocessor


def test_split_no_overlap():
    """1. Индексы train и test не пересекаются"""
    df = load_data()
    X_train, X_test, _, _ = split_data(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_unknown_category():
    """2. Пайплайн принимает строку с неизвестной категорией"""
    df = load_data()
    X_train, _, y_train, _ = split_data(df)
    
    numeric_cols = ["Length3", "Height", "Width"]
    categorical_cols = ["Species"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = Pipeline([("preprocess", preprocessor), ("regressor", Ridge())])
    model.fit(X_train, y_train)

    X_new = pd.DataFrame([{"Length3": 30.0, "Height": 10.0, "Width": 5.0, "Species": "UnknownFish"}])
    pred = model.predict(X_new)
    assert len(pred) == 1


def test_prediction_shape_and_finite():
    """3. Число прогнозов равно числу строк, значения конечны"""
    df = load_data()
    X_train, X_test, y_train, _ = split_data(df)
    
    model = Pipeline([
        ("preprocess", build_preprocessor(["Length3", "Height", "Width"], ["Species"])),
        ("regressor", Ridge())
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    assert len(preds) == len(X_test)
    assert np.all(np.isfinite(preds))


def test_metrics_manual_calculation():
    """4. Метрики на зафиксированном контрольном примере совпадают с ручным расчетом"""
    # Возьмем простые числа для ручного расчета
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    
    # Ручной расчет:
    # MSE = ((100-110)^2 + (200-180)^2) / 2 = (100 + 400) / 2 = 250.0
    # MAE = (|100-110| + |200-180|) / 2 = (10 + 20) / 2 = 15.0
    # Среднее y_true = 150. R2 = 1 - ( (100+400) / ((100-150)^2 + (200-150)^2) ) = 1 - (500 / 5000) = 0.9
    
    assert np.isclose(mean_squared_error(y_true, y_pred), 250.0)
    assert np.isclose(mean_absolute_error(y_true, y_pred), 15.0)
    assert np.isclose(r2_score(y_true, y_pred), 0.9)


def test_saved_pipeline_reproducibility():
    """5. Сохраненная цепочка после загрузки воспроизводит прогноз в заданном допуске"""
    df = load_data()
    X_train, X_test, y_train, _ = split_data(df)
    
    model = Pipeline([
        ("preprocess", build_preprocessor(["Length3", "Height", "Width"], ["Species"])),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    
    # Делаем прогноз до сохранения
    pred_original = model.predict(X_test.iloc[:5])
    
    # Сохраняем и загружаем (используем joblib для скорости и надежности теста)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        joblib.dump(model, model_path)
        loaded_model = joblib.load(model_path)
        
    # Делаем прогноз после загрузки
    pred_loaded = loaded_model.predict(X_test.iloc[:5])
    
    # Проверяем, что прогнозы совпадают с точностью до 5 знаков (допуск)
    np.testing.assert_allclose(pred_original, pred_loaded, rtol=1e-5)