"""Bootstrap доверительные интервалы для лучшей модели.

Запуск: python -m src.bootstrap_ci
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.utils import resample

from src.features import load_data, split_train_val_test, split_xy, add_volume
from src.model import compute_metrics


def main(config_path: str = "configs/default.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])

    df = load_data(cfg["data"]["path"], cfg["data"]["drop_zero_weight"])
    df = add_volume(df)
    _, _, test_df = split_train_val_test(
        df, cfg["split"]["test_size"], cfg["split"]["val_size"],
        seed, cfg["split"]["group_round_decimals"],
    )

    best_id = json.loads(
        Path("models/best_model_metrics.json").read_text()
    )["best_experiment"]
    exp_cfg = cfg["experiments"][best_id]
    model = joblib.load("models/best_model.joblib")

    X_test, y_test = split_xy(
        test_df, exp_cfg["features"],
        bool(exp_cfg.get("categorical", False)),
    )
    if X_test.shape[1] == 0:
        X_test = pd.DataFrame(index=X_test.index)
    y_test = y_test.to_numpy()

    N_ITERATIONS = 1000
    rmse_scores, r2_scores = [], []
    np.random.seed(seed)

    for _ in range(N_ITERATIONS):
        X_boot, y_boot = resample(X_test, y_test, random_state=None)
        y_pred = model.predict(X_boot)
        m = compute_metrics(y_boot, y_pred)
        rmse_scores.append(m["rmse"])
        r2_scores.append(m["r2"])

    rmse_scores = np.array(rmse_scores)
    r2_scores = np.array(r2_scores)
    rmse_ci = np.percentile(rmse_scores, [2.5, 97.5])
    r2_ci = np.percentile(r2_scores, [2.5, 97.5])

    print(f"Bootstrap 95% CI (N={N_ITERATIONS}):")
    print(f"  RMSE: [{rmse_ci[0]:.2f}; {rmse_ci[1]:.2f}] г")
    print(f"  R²:   [{r2_ci[0]:.4f}; {r2_ci[1]:.4f}]")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_iterations": N_ITERATIONS,
            "rmse_ci_95": rmse_ci.tolist(),
            "r2_ci_95": r2_ci.tolist(),
        }, f, ensure_ascii=False, indent=2)

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rmse_scores, bins=30, color="skyblue", edgecolor="black")
    axes[0].axvline(rmse_ci[0], color="red", linestyle="--")
    axes[0].axvline(rmse_ci[1], color="red", linestyle="--")
    axes[0].set_title("Bootstrap RMSE")
    axes[0].set_xlabel("RMSE, г")
    axes[1].hist(r2_scores, bins=30, color="lightgreen", edgecolor="black")
    axes[1].axvline(r2_ci[0], color="red", linestyle="--")
    axes[1].axvline(r2_ci[1], color="red", linestyle="--")
    axes[1].set_title("Bootstrap R²")
    axes[1].set_xlabel("R²")
    fig.tight_layout()
    fig.savefig(fig_dir / "06_bootstrap_ci.png", dpi=150)
    plt.close(fig)
    print("Сохранено: reports/figures/06_bootstrap_ci.png, reports/bootstrap_ci.json")


if __name__ == "__main__":
    main()