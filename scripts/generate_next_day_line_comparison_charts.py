from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"


def plot_regression_line() -> None:
    df = pd.read_csv(TABLES / "regression_results.csv")
    metrics = ["mae", "rmse", "r2"]
    long_df = df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")

    plt.figure(figsize=(10, 5))
    for model in long_df["model"].unique():
        part = long_df[long_df["model"] == model]
        plt.plot(part["metric"], part["value"], marker="o", linewidth=1.8, label=model)
    plt.title("Next-day temperature regression model comparison")
    plt.xlabel("Evaluation metric")
    plt.ylabel("Metric value")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "regression_model_comparison_line.png", dpi=160)
    plt.close()


def plot_classification_line() -> None:
    df = pd.read_csv(TABLES / "rain_classification_results.csv")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    long_df = df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")

    plt.figure(figsize=(11, 5))
    for model in long_df["model"].unique():
        part = long_df[long_df["model"] == model]
        plt.plot(part["metric"], part["value"], marker="o", linewidth=1.8, label=model)
    plt.title("Next-day rain classification model comparison")
    plt.xlabel("Evaluation metric")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "classification_model_comparison_line.png", dpi=160)
    plt.close()


def main() -> None:
    plot_regression_line()
    plot_classification_line()


if __name__ == "__main__":
    main()
