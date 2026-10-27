import numpy as np
import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample

from src.data import load_data, split_data
from src.preprocess import build_preprocessor

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Обучаем модель
    numeric_cols = ["Length3", "Height", "Width"]
    categorical_cols = ["Species"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, use_log_features=True)
    regressor = TransformedTargetRegressor(regressor=Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1)
    model = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Бутстрэп (1000 итераций)
    n_iterations = 1000
    mse_scores, mae_scores, r2_scores = [], [], []

    print(f"Запуск Bootstrap ({n_iterations} итераций)...")
    for _ in range(n_iterations):
        # Случайная выборка с возвращением (индексы)
        indices = resample(np.arange(len(y_test)))
        y_true_boot = y_test.iloc[indices]
        y_pred_boot = y_pred[indices]

        mse_scores.append(mean_squared_error(y_true_boot, y_pred_boot))
        mae_scores.append(mean_absolute_error(y_true_boot, y_pred_boot))
        r2_scores.append(r2_score(y_true_boot, y_pred_boot))

    # 95% доверительные интервалы (2.5% и 97.5% перцентили)
    metrics_ci = {
        "mse_ci_lower": np.percentile(mse_scores, 2.5),
        "mse_ci_upper": np.percentile(mse_scores, 97.5),
        "mae_ci_lower": np.percentile(mae_scores, 2.5),
        "mae_ci_upper": np.percentile(mae_scores, 97.5),
        "r2_ci_lower": np.percentile(r2_scores, 2.5),
        "r2_ci_upper": np.percentile(r2_scores, 97.5),
    }

    print("\n--- 95% Доверительные интервалы (Bootstrap) ---")
    print(f"MSE: [{metrics_ci['mse_ci_lower']:.2f}, {metrics_ci['mse_ci_upper']:.2f}]")
    print(f"MAE: [{metrics_ci['mae_ci_lower']:.2f}, {metrics_ci['mae_ci_upper']:.2f}]")
    print(f"R2:  [{metrics_ci['r2_ci_lower']:.4f}, {metrics_ci['r2_ci_upper']:.4f}]")

    # Логируем в MLflow
    with mlflow.start_run(run_name="E5-bootstrap-confidence-intervals"):
        mlflow.log_params({"n_iterations": n_iterations, "model": "Ridge (best)"})
        mlflow.log_metrics(metrics_ci)
        mlflow.set_tag("task", "Additional: Bootstrap CI")

if __name__ == "__main__":
    main()