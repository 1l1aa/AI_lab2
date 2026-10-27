import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor

from src.data import load_data, split_data
from src.preprocess import build_preprocessor

def test_split_no_overlap():
    """Проверка: индексы train и test не пересекаются"""
    df = load_data()
    X_train, X_test, _, _ = split_data(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))

def test_unknown_category():
    """Проверка: пайплайн не падает при неизвестной категории"""
    df = load_data()
    X_train, _, y_train, _ = split_data(df)
    
    numeric_cols = ["Length3", "Height", "Width"]
    categorical_cols = ["Species"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = Pipeline([("preprocess", preprocessor), ("regressor", Ridge())])
    model.fit(X_train, y_train)

    # Создаем "строку" с неизвестным видом
    X_new = pd.DataFrame([{"Length3": 30.0, "Height": 10.0, "Width": 5.0, "Species": "UnknownFish"}])
    pred = model.predict(X_new)
    assert len(pred) == 1

def test_prediction_shape_and_finite():
    """Проверка: число прогнозов равно числу строк, значения конечны"""
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