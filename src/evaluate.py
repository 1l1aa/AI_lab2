"""Финальная оценка лучшей модели на test. Открывать test ОДИН РАЗ.

Запуск:
    python -m src.evaluate --config configs/default.yaml
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.features import load_data, split_train_val_test, split_xy, add_volume
from src.model import compute_metrics


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])

    # 1. Загружаем данные
    df = load_data(cfg["data"]["path"], cfg["data"]["drop_zero_weight"])
    df = add_volume(df)
    
    # 2. Восстанавливаем то же самое разбиение
    _, _, test_df = split_train_val_test(
        df,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        seed=seed,
        decimals=cfg["split"]["group_round_decimals"],
    )

    # 3. Загружаем лучшую модель
    best_id = json.loads(Path("models/best_model_metrics.json").read_text())["best_experiment"]
    exp_cfg = cfg["experiments"][best_id]
    model = joblib.load("models/best_model.joblib")

    # 4. Готовим тестовые признаки
    X_test, y_test = split_xy(
        test_df,
        numeric_features=exp_cfg["features"],
        use_categorical=bool(exp_cfg.get("categorical", False)),
    )

    # 5. Предсказание и метрики
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.to_numpy(), y_pred)
    
    print(f"=== Оценка на тесте ({best_id}: {exp_cfg['name']}) ===")
    print(f"  Test RMSE: {metrics['rmse']:.2f} г")
    print(f"  Test MAE : {metrics['mae']:.2f} г")
    print(f"  Test R^2 : {metrics['r2']:.4f}")

    # 6. Сохраняем отчет в JSON
    Path("reports").mkdir(exist_ok=True)
    with open("reports/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_experiment": best_id,
            "name": exp_cfg["name"],
            **metrics
        }, f, ensure_ascii=False, indent=2)

    # 7. Строим графики
    FIG = Path("reports/figures")
    FIG.mkdir(parents=True, exist_ok=True)

    # График 1: Предсказание vs Факт
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="идеал y=x")
    ax.set_xlabel("Факт, г")
    ax.set_ylabel("Предсказание, г")
    ax.set_title(f"Test: {best_id} — предсказание vs факт")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "02_pred_vs_fact_test.png", dpi=150)
    plt.close()

    # График 2: Остатки
    residuals = y_test.to_numpy() - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Предсказание, г")
    ax.set_ylabel("Остаток (факт − предсказание), г")
    ax.set_title("Test residuals")
    plt.tight_layout()
    plt.savefig(FIG / "03_residuals_test.png", dpi=150)
    plt.close()

    print("\nГрафики сохранены в reports/figures/")
    print("Метрики сохранены в reports/test_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)