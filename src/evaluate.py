import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
import os

from src.data import load_data, split_data
from src.preprocess import build_preprocessor

def main():
    os.makedirs("reports", exist_ok=True)
    
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Обучаем лучшую модель (Ridge из E2)
    numeric_cols = ["Length3", "Height", "Width"]
    categorical_cols = ["Species"]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, use_log_features=True)
    regressor = TransformedTargetRegressor(regressor=Ridge(alpha=0.1), func=np.log1p, inverse_func=np.expm1)
    model = Pipeline([("preprocess", preprocessor), ("regressor", regressor)])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Сохраняем результаты
    results = X_test.copy()
    results["y_true"] = y_test
    results["y_pred"] = y_pred
    results["residual"] = y_test - y_pred
    results["abs_error"] = np.abs(results["residual"])

    with mlflow.start_run(run_name="Error-analysis"):
        # 1. График остатков
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x="y_pred", y="residual", data=results)
        plt.axhline(0, color="red", linestyle="--")
        plt.title("График остатков (Ridge)")
        plt.xlabel("Предсказанный вес")
        plt.ylabel("Остаток (y_true - y_pred)")
        plt.tight_layout()
        plt.savefig("reports/residuals_plot.png")
        mlflow.log_artifact("reports/residuals_plot.png")
        plt.close()
        print("График остатков сохранен в reports/residuals_plot.png и в MLflow")

        # 2. График Прогноз vs Истина
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x="y_true", y="y_pred", data=results)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title("Зависимость прогноза от истинного значения")
        plt.xlabel("Истинный вес")
        plt.ylabel("Предсказанный вес")
        plt.tight_layout()
        plt.savefig("reports/prediction_vs_true.png")
        mlflow.log_artifact("reports/prediction_vs_true.png")
        plt.close()
        print("График прогнозов сохранен в reports/prediction_vs_true.png и в MLflow")

        # 3. Топ-5 крупнейших ошибок
        top_errors = results.nlargest(5, "abs_error")
        top_errors.to_csv("reports/top_errors.csv", index=False)
        mlflow.log_artifact("reports/top_errors.csv")
        print("\n--- Топ-5 крупнейших ошибок ---")
        print(top_errors[["Species", "Length3", "y_true", "y_pred", "abs_error"]])

        # 4. MAE по видам
        mae_by_species = results.groupby("Species")["abs_error"].mean().sort_values(ascending=False)
        print("\n--- MAE по видам рыб ---")
        print(mae_by_species)

        # 5. Проверка отрицательных предсказаний
        neg_preds = results[results["y_pred"] < 0]
        print(f"\nКоличество отрицательных предсказаний: {len(neg_preds)}")

if __name__ == "__main__":
    main()