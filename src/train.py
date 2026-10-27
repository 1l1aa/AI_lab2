import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import sklearn
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV

from src.data import load_data, split_data
from src.preprocess import build_preprocessor

RANDOM_STATE = 42


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def log_experiment(run_name, model, X_train, X_test, y_train, y_test, params, cv=5):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("python_version", sys.version.split()[0])
        mlflow.set_tag("sklearn_version", sklearn.__version__)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
        cv_mse_mean = -cv_scores.mean()
        cv_mse_std = cv_scores.std()
        mlflow.log_metrics({"cv_mse_mean": cv_mse_mean, "cv_mse_std": cv_mse_std})
        print(f"{run_name} CV MSE: {cv_mse_mean:.2f} +/- {cv_mse_std:.2f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        print(f"{run_name} test: {metrics}")

        # Сохраняем таблицу прогнозов
        pred_df = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_pred,
            "residual": y_test.values - y_pred,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pred_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name)
        os.unlink(f.name)

        # --- ГЕНЕРАЦИЯ И ЛОГИРОВАНИЕ ГРАФИКОВ ---
        os.makedirs("reports", exist_ok=True)
        
        # 1. График остатков
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=y_pred, y=y_test.values - y_pred)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Residuals - {run_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Residual (True - Pred)")
        plt.tight_layout()
        res_path = f"reports/{run_name}_residuals.png"
        plt.savefig(res_path)
        plt.close()
        mlflow.log_artifact(res_path)

        # 2. График Прогноз vs Истина
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=y_test.values, y=y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f"Predictions vs Actual - {run_name}")
        plt.xlabel("Actual Weight")
        plt.ylabel("Predicted Weight")
        plt.tight_layout()
        pred_path = f"reports/{run_name}_pred_vs_true.png"
        plt.savefig(pred_path)
        plt.close()
        mlflow.log_artifact(pred_path)

        # --- СОХРАНЕНИЕ МОДЕЛИ ---
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )
        return metrics


def run_e0(X_train, X_test, y_train, y_test):
    X_dummy_train = np.zeros((len(X_train), 1))
    X_dummy_test = np.zeros((len(X_test), 1))
    for strategy in ["mean", "median"]:
        model = DummyRegressor(strategy=strategy)
        log_experiment(
            f"E0-dummy-{strategy}",
            model, X_dummy_train, X_dummy_test, y_train, y_test,
            {"model": f"DummyRegressor({strategy})"},
        )


def run_e1(X_train, X_test, y_train, y_test):
    numeric_cols = ["Length3", "Height", "Width"]
    categorical_cols = ["Species"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, use_log_features=True)
    regressor = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)
    model = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])
    log_experiment(
        "E1-linear-log-target",
        model, X_train, X_test, y_train, y_test,
        {"model": "LinearRegression", "target": "log1p", "features": "log1p"},
    )


def run_e2(X_train, X_test, y_train, y_test):
    numeric_cols = ["Length3", "Height", "Width"]
    categorical_cols = ["Species"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, use_log_features=True)
    regressor = TransformedTargetRegressor(regressor=Ridge(), func=np.log1p, inverse_func=np.expm1)
    model = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])

    param_grid = {"regressor__regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_alpha = grid.best_params_["regressor__regressor__alpha"]
    print(f"E2 best alpha: {best_alpha}")

    log_experiment(
        "E2-ridge-log-target",
        best_model, X_train, X_test, y_train, y_test,
        {"model": "Ridge", "alpha": best_alpha, "target": "log1p", "features": "log1p"},
    )


def run_e3(X_train, X_test, y_train, y_test):
    numeric_cols = ["Length3", "Height", "Width"]
    categorical_cols = ["Species"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, use_log_features=False)
    regressor = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, max_iter=200, learning_rate=0.1, max_depth=5
    )
    model = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])
    log_experiment(
        "E3-hgb-raw",
        model, X_train, X_test, y_train, y_test,
        {"model": "HistGradientBoostingRegressor", "max_iter": 200, "max_depth": 5},
    )


def run_e4(X_train, X_test, y_train, y_test):
    X_train_e4 = X_train.copy()
    X_test_e4 = X_test.copy()
    X_train_e4["Vproxy"] = X_train_e4["Length3"] * X_train_e4["Height"] * X_train_e4["Width"]
    X_test_e4["Vproxy"] = X_test_e4["Length3"] * X_test_e4["Height"] * X_test_e4["Width"]

    numeric_cols = ["Length3", "Height", "Width", "Vproxy"]
    categorical_cols = ["Species"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, use_log_features=True)
    regressor = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)
    model = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])
    log_experiment(
        "E4-linear-vproxy",
        model, X_train_e4, X_test_e4, y_train, y_test,
        {"model": "LinearRegression", "features": "log1p + Vproxy", "target": "log1p"},
    )


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    run_e0(X_train, X_test, y_train, y_test)
    run_e1(X_train, X_test, y_train, y_test)
    run_e2(X_train, X_test, y_train, y_test)
    run_e3(X_train, X_test, y_train, y_test)
    run_e4(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()