"""Обучение моделей E0–E4 + диагностика E1 + логирование в MLflow.

Запуск:
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.features import (
    add_volume, build_preprocessor, load_data, split_train_val_test, split_xy
)
from src.model import compute_metrics, make_model


def run_experiment(
    exp_id: str,
    exp_cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> tuple[Pipeline, dict]:
    """Один эксперимент E0..E4. Метрики считаются на val и test."""
    numeric_features = exp_cfg["features"]
    use_cat = bool(exp_cfg.get("categorical", False))
    use_scaling = bool(exp_cfg.get("scaling", False))
    needs_groups = bool(exp_cfg.get("needs_groups", False))

    X_train, y_train = split_xy(train_df, numeric_features, use_cat)
    X_val,   y_val   = split_xy(val_df,   numeric_features, use_cat)
    X_test,  y_test  = split_xy(test_df,  numeric_features, use_cat)

    pre = build_preprocessor(numeric_features, use_cat, use_scaling)
    
    # Передаем группы для кросс-валидации (если нужно)
    model = make_model(
        exp_cfg["model"],
        seed=seed,
        target_transform=exp_cfg.get("target_transform"),
        groups=train_df["group"].to_numpy() if needs_groups else None
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    pipe.fit(X_train, y_train)

    y_val_pred  = pipe.predict(X_val)
    y_test_pred = pipe.predict(X_test)

    m_val  = compute_metrics(y_val.to_numpy(),  y_val_pred)
    m_test = compute_metrics(y_test.to_numpy(), y_test_pred)

    metrics = {f"val_{k}": v for k, v in m_val.items()}
    metrics.update({f"test_{k}": v for k, v in m_test.items()})
    return pipe, metrics


def diagnose_linear(
    pipe: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_features,
    out_dir: Path = Path("reports"),
) -> dict:
    """Диагностика для E1: остатки, VIF, корреляции."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True)

    X_val, y_val = split_xy(val_df, numeric_features, use_cat=False)
    y_pred = pipe.predict(X_val)
    res = y_val.to_numpy() - y_pred

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(y_pred, res, s=8)
    ax[0].axhline(0.0, color="red")
    ax[0].set_xlabel("fitted")
    ax[0].set_ylabel("residual")
    ax[0].set_title("Residuals vs Fitted")
    ax[1].hist(res, bins=30)
    ax[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "e1_residuals.png", dpi=120)
    plt.close(fig)

    corr = train_df[numeric_features].corr()

    vif = {}
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Xnum = train_df[numeric_features].dropna().astype(float)
        for i, c in enumerate(numeric_features):
            vif[c] = float(variance_inflation_factor(Xnum.values, i))
    except Exception as e:
        vif = {"error": str(e)}

    skew = float(train_df["Weight"].skew())
    return {"vif": vif, "skew_target": skew, "corr": corr.round(3).to_dict()}


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])

    df = load_data(cfg["data"]["path"], cfg["data"]["drop_zero_weight"])
    df = add_volume(df)

    train_df, val_df, test_df = split_train_val_test(
        df,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        seed=seed,
        decimals=cfg["split"].get("group_round_decimals", 2),
    )
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results: dict[str, dict] = {}
    trained: dict[str, Pipeline] = {}

    # Список типов, которым MLflow должен доверять при сохранении моделей
    trusted_types = [
        "numpy.dtype",
        "sklearn._loss._loss.CyHalfSquaredError",
        "sklearn._loss.link.IdentityLink",
        "sklearn._loss.link.Interval",
        "sklearn._loss.loss.HalfSquaredError",
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        "sklearn.metrics._regression.mean_squared_error",
        "sklearn.metrics._scorer._PredictScorer",
    ]

    for exp_id, exp_cfg in cfg["experiments"].items():
        with mlflow.start_run(run_name=f"{exp_id}_{exp_cfg['name']}"):
            pipe, metrics = run_experiment(
                exp_id, exp_cfg, train_df, val_df, test_df, seed
            )
            trained[exp_id] = pipe
            results[exp_id] = {"name": exp_cfg["name"], **metrics}

            mlflow.log_params({
                "exp_id": exp_id,
                "model": exp_cfg["model"],
                "features": ",".join(exp_cfg["features"]),
                "categorical": exp_cfg.get("categorical", False),
                "scaling": exp_cfg.get("scaling", False),
                "target_transform": exp_cfg.get("target_transform", "none"),
                "seed": seed,
            })
            mlflow.log_metrics(metrics)

            if exp_cfg["model"] == "linear":
                diag = diagnose_linear(pipe, train_df, val_df, exp_cfg["features"])
                mlflow.log_dict(diag, "diagnostics_e1.json")
                for name, v in diag.get("vif", {}).items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"vif_{name}", v)
                mlflow.log_metric("skew_target", diag["skew_target"])

            if cfg["mlflow"].get("log_models", True):
                try:
                    mlflow.sklearn.log_model(
                        pipe, 
                        name="model", 
                        skops_trusted_types=trusted_types
                    )
                except TypeError:
                    # Для старых версий MLflow
                    mlflow.sklearn.log_model(pipe, artifact_path="model")

            print(f"{exp_id}: {metrics}")

    best_id = min(results, key=lambda k: results[k]["val_rmse"])
    print(f"\nЛучшая модель по val_rmse: {best_id} "
          f"({results[best_id]['name']}) "
          f"| test_rmse={results[best_id]['test_rmse']:.4f}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(trained[best_id], "models/best_model.joblib")

    with open("models/best_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_experiment": best_id, "results": results},
            f, ensure_ascii=False, indent=2,
        )

    summary = pd.DataFrame(
        [
            {
                "exp_id": k,
                "name": v["name"],
                "val_rmse": v["val_rmse"],
                "test_rmse": v["test_rmse"],
                "test_mae": v["test_mae"],
                "test_r2": v["test_r2"],
            }
            for k, v in results.items()
        ]
    ).sort_values("val_rmse")
    summary.to_csv("models/summary.csv", index=False)
    print("\nСводка (сортировка по val_rmse):")
    print(summary.to_string(index=False))
    print("\nСохранено: models/best_model.joblib, models/summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)