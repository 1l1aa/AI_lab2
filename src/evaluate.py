"""Финальная оценка лучшей модели на test + анализ ошибок."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features import load_data, split_train_val_test, split_xy, add_volume
from src.model import compute_metrics


def analyze_errors(test_df, y_true, y_pred, out_dir: Path):
    df = test_df.copy().reset_index(drop=True)
    df["y_true"] = np.asarray(y_true)
    df["y_pred"] = np.asarray(y_pred)
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()

    cols_top = [c for c in
                ["Species", "y_true", "y_pred", "abs_err",
                 "Length3", "Height", "Width", "Volume"]
                if c in df.columns]
    top5 = df.nlargest(5, "abs_err")[cols_top]
    top5.to_csv(out_dir / "error_top5.csv", index=False)
    print("\n=== Топ-5 ошибок ===")
    print(top5.to_string(index=False))

    if "Species" in df.columns:
        mae_by_species = (
            df.groupby("Species")["abs_err"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "MAE", "count": "n"})
            .sort_values("MAE", ascending=False)
        )
        mae_by_species.to_csv(out_dir / "error_mae_by_species.csv")
        print("\n=== MAE по видам ===")
        print(mae_by_species.to_string())

    bins = [0, 200, 500, 1000, np.inf]
    labels = ["<200", "200-500", "500-1000", ">1000"]
    df["weight_bin"] = pd.cut(df["y_true"], bins=bins, labels=labels, right=False)
    mae_by_bin = (
        df.groupby("weight_bin", observed=True)["abs_err"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "MAE", "count": "n"})
    )
    mae_by_bin.to_csv(out_dir / "error_mae_by_weight_range.csv")
    print("\n=== MAE по диапазонам массы (г) ===")
    print(mae_by_bin.to_string())

    neg = df[df["y_pred"] <= 0]
    neg.to_csv(out_dir / "error_non_positive_predictions.csv", index=False)
    print(f"\n=== Отрицательные/нулевые прогнозы: {len(neg)} ===")


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])

    df = load_data(cfg["data"]["path"], cfg["data"]["drop_zero_weight"])
    df = add_volume(df)
    _, _, test_df = split_train_val_test(
        df,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        seed=seed,
        decimals=cfg["split"]["group_round_decimals"],
    )

    meta = json.loads(Path("models/best_model_metrics.json").read_text())
    best_id = meta["best_experiment"]
    exp_cfg = cfg["experiments"][best_id]
    model = joblib.load("models/best_model.joblib")

    X_test, y_test = split_xy(
        test_df,
        numeric_features=exp_cfg["features"],
        use_categorical=bool(exp_cfg.get("categorical", False)),
    )
    if X_test.shape[1] == 0:
        X_test = pd.DataFrame(index=X_test.index)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.to_numpy(), y_pred)

    print(f"=== Оценка на тесте ({best_id}: {exp_cfg['name']}) ===")
    for k, v in metrics.items():
        print(f"  test_{k}: {v:.4f}")

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    with open(reports / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_experiment": best_id, "name": exp_cfg["name"],
                   **metrics}, f, ensure_ascii=False, indent=2)

    analyze_errors(test_df, y_test.to_numpy(), y_pred, reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)