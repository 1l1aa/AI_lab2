"""Фабрика моделей, метрики, обёртка log-target."""
from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_model(name: str, seed: int = 42, **kwargs):
    if name == "dummy_regressor":
        return DummyRegressor(strategy=kwargs.get("strategy", "mean"))
    if name == "linear_regression":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "random_forest":
        p = kwargs.get("rf_params") or {}
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 1),
            random_state=p.get("random_state", seed),
            n_jobs=p.get("n_jobs", -1),
        )
    raise ValueError(f"Неизвестная модель: {name}")


def build_estimator(exp_cfg: dict, seed: int) -> Pipeline:
    from src.features import build_preprocessor

    numeric = exp_cfg["features"]
    use_cat = bool(exp_cfg.get("categorical", False))
    use_scaling = bool(exp_cfg.get("scaling", False))
    log_target = bool(exp_cfg.get("log_target", False))

    pre = build_preprocessor(numeric, use_cat, use_scaling)

    extra = {k: v for k, v in exp_cfg.items() if k not in {"model", "name"}}
    model = make_model(exp_cfg["model"], seed=seed, **extra)

    if exp_cfg["model"] == "ridge" and exp_cfg.get("tune_alpha", False):
        regressor = GridSearchCV(
            Ridge(random_state=seed),
            param_grid={"alpha": exp_cfg.get("alpha_grid", [1.0])},
            cv=5,
            scoring="neg_mean_squared_error",
        )
    else:
        regressor = model

    if log_target:
        return Pipeline([
            ("pre", pre),
            ("ttr", TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=np.expm1,
            )),
        ])

    return Pipeline([("pre", pre), ("model", regressor)])