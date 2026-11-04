"""Обучение E0–E4 + логирование в MLflow.

Запуск:
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import cross_validate

from src.features import add_volume, load_data, split_train_val_test, split_xy
from src.model import build_estimator, compute_metrics


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _plot_pred_vs_fact(y_true, y_pred, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="k")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", label="идеал y=x")
    ax.set_xlabel("Факт, г")
    ax.set_ylabel("Прогноз, г")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_residuals(y_pred, residuals, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Прогноз, г")
    ax.set_ylabel("Остаток (факт − прогноз), г")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _cv_metrics(pipe, X, y):
    """5-fold CV. При пустом X (E0) возвращает NaN — не критично."""
    if X.shape[1] == 0:
        return {k: float("nan") for k in
                ["cv_mse", "cv_mse_std", "cv_mae", "cv_mae_std",
                 "cv_r2", "cv_r2_std"]}
    res = cross_validate(
        pipe, X, y, cv=5,
        scoring={"mse": "neg_mean_squared_error",
                 "mae": "neg_mean_absolute_error",
                 "r2": "r2"},
        error_score=np.nan,
    )
    return {
        "cv_mse": float(-np.nanmean(res["test_mse"])),
        "cv_mse_std": float(np.nanstd(-res["test_mse"])),
        "cv_mae": float(-np.nanmean(res["test_mae"])),
        "cv_mae_std": float(np.nanstd(-res["test_mae"])),
        "cv_r2": float(np.nanmean(res["test_r2"])),
        "cv_r2_std": float(np.nanstd(res["test_r2"])),
    }


def run_experiment(exp_id, exp_cfg, train_df, val_df, test_df, seed, fig_dir):
    numeric = exp_cfg["features"]
    use_cat = bool(exp_cfg.get("categorical", False))

    X_train, y_train = split_xy(train_df, numeric, use_cat)
    X_val, y_val = split_xy(val_df, numeric, use_cat)
    X_test, y_test = split_xy(test_df, numeric, use_cat)

    # Для E0 (features=[]) — пустой DataFrame с правильными индексами
    if X_train.shape[1] == 0:
        X_train = pd.DataFrame(index=X_train.index)
    if X_val.shape[1] == 0:
        X_val = pd.DataFrame(index=X_val.index)
    if X_test.shape[1] == 0:
        X_test = pd.DataFrame(index=X_test.index)

    pipe = build_estimator(exp_cfg, seed=seed)
    cv_metrics = _cv_metrics(pipe, X_train, y_train)

    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    val_m = compute_metrics(y_val.to_numpy(), y_val_pred)

    y_test_pred = pipe.predict(X_test)
    test_m = compute_metrics(y_test.to_numpy(), y_test_pred)

    fig_pf = fig_dir / f"{exp_id}_pred_vs_fact.png"
    fig_res = fig_dir / f"{exp_id}_residuals.png"
    _plot_pred_vs_fact(y_test.to_numpy(), y_test_pred, fig_pf,
                       f"{exp_id}: прогноз vs факт (test)")
    _plot_residuals(y_test_pred, y_test.to_numpy() - y_test_pred, fig_res,
                    f"{exp_id}: остатки (test)")

    pred_table = X_test.copy().reset_index(drop=True)
    pred_table["y_true"] = y_test.to_numpy()
    pred_table["y_pred"] = y_test_pred
    pred_table["residual"] = pred_table["y_true"] - pred_table["y_pred"]
    pred_csv = fig_dir.parent / f"{exp_id}_test_predictions.csv"
    pred_table.to_csv(pred_csv, index=False)

    return {
        "pipe": pipe,
        **cv_metrics,
        "val": val_m,
        "test": test_m,
        "artifacts": [str(fig_pf), str(fig_res), str(pred_csv)],
    }


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
        decimals=cfg["split"]["group_round_decimals"],
    )
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results = {}
    trained = {}
    commit = _git_commit()

    for exp_id, exp_cfg in cfg["experiments"].items():
        with mlflow.start_run(run_name=f"{exp_id}_{exp_cfg['name']}"):
            out = run_experiment(
                exp_id, exp_cfg, train_df, val_df, test_df, seed, fig_dir
            )
            trained[exp_id] = out["pipe"]
            results[exp_id] = {
                "name": exp_cfg["name"],
                **{k: out[k] for k in
                   ["cv_mse", "cv_mse_std", "cv_mae", "cv_mae_std",
                    "cv_r2", "cv_r2_std"]},
                "test": out["test"],
                "val": out["val"],
            }

            mlflow.set_tags({
                "git_commit": commit,
                "data_version": cfg["data"].get("version", "unknown"),
                "seed": str(seed),
                "python": ".".join(map(str, os.sys.version_info[:3])),
            })
            mlflow.log_params({
                "exp_id": exp_id,
                "model": exp_cfg["model"],
                "features": ",".join(exp_cfg["features"]) or "none",
                "categorical": exp_cfg.get("categorical", False),
                "scaling": exp_cfg.get("scaling", False),
                "log_target": exp_cfg.get("log_target", False),
                "tune_alpha": exp_cfg.get("tune_alpha", False),
                "seed": seed,
                "test_size": cfg["split"]["test_size"],
                "val_size": cfg["split"]["val_size"],
            })

            mlflow.log_metrics({
                "cv_mse": out["cv_mse"], "cv_mse_std": out["cv_mse_std"],
                "cv_mae": out["cv_mae"], "cv_mae_std": out["cv_mae_std"],
                "cv_r2": out["cv_r2"], "cv_r2_std": out["cv_r2_std"],
                "val_mse": out["val"]["mse"], "val_mae": out["val"]["mae"],
                "val_r2": out["val"]["r2"],
                "test_mse": out["test"]["mse"], "test_mae": out["test"]["mae"],
                "test_r2": out["test"]["r2"], "test_rmse": out["test"]["rmse"],
            })

            for art in out["artifacts"]:
                mlflow.log_artifact(art)

            if cfg["mlflow"].get("log_models", True):
                try:
                    mlflow.sklearn.log_model(out["pipe"], artifact_path="model")
                except Exception as e:
                    print(f"  [warn] log_model failed for {exp_id}: {e}")

            print(f"{exp_id}: test_mse={out['test']['mse']:.2f}, "
                  f"test_r2={out['test']['r2']:.4f}")

    valid = {k: v for k, v in results.items() if not np.isnan(v["cv_mse"])}
    if valid:
        best_id = min(valid, key=lambda k: valid[k]["cv_mse"])
    else:
        best_id = list(results)[-1]
    print(f"\nЛучшая модель по CV MSE: {best_id} ({results[best_id]['name']})")

    joblib.dump(trained[best_id], "models/best_model.joblib")
    with open("models/best_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_experiment": best_id, **results}, f,
                  ensure_ascii=False, indent=2)
    print("Сохранено: models/best_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)