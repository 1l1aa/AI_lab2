"""Фабрика моделей и метрики."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mse":  float(mean_squared_error(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def _wrap_target(model, target_transform: str | None):
    if target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=model, func=np.log1p, inverse_func=np.expm1
        )
    return model


def make_model(name: str, seed: int = 42,
               target_transform: str | None = None,
               groups=None):
    """Создаёт модель по имени.
    
    groups: если передано — используется для GroupKFold во вложенном CV.
    """
    # Создаем генератор сплитов на основе переданных групп
    if groups is not None:
        cv = list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))
    else:
        cv = GroupKFold(n_splits=5)

    if name == "dummy_mean":
        m = DummyRegressor(strategy="mean")
    elif name == "dummy_median":
        m = DummyRegressor(strategy="median")
    elif name == "linear":
        m = LinearRegression()
    elif name == "ridge_cv":
        m = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv)
    elif name == "elastic_cv":
        m = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            alphas=np.logspace(-3, 2, 20),
            cv=cv, max_iter=10_000, random_state=seed,
        )
    elif name == "rf":
        m = RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=1, # n_jobs=1 для стабильности в Jupyter
        )
    elif name == "hgb":
        m = HistGradientBoostingRegressor(random_state=seed)
    elif name == "hgb_search":
        base = HistGradientBoostingRegressor(random_state=seed)
        param_dist = {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [None, 3, 5],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [0.0, 1.0],
        }
        m = RandomizedSearchCV(
            base, param_dist, n_iter=20, cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=seed, n_jobs=1, refit=True, # n_jobs=1
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return _wrap_target(m, target_transform)