"""Plotting utilities for experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_bias_advantage(bias_df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4.5))
    ax = sns.lineplot(
        data=bias_df,
        x="bias",
        y="advantage_over_truth",
        hue="variant",
        marker="o",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Deviation Advantage vs. Audit/Sanction Variants")
    ax.set_ylabel("Utility advantage over truthful")
    ax.set_xlabel("Bias added to reported probability")
    ax.legend(title="Variant")
    plt.tight_layout()
    plt.savefig(output_dir / "bias_advantage.png", dpi=200)
    plt.close()


def plot_audit_penalty_heatmap(grid_df: pd.DataFrame, output_dir: Path) -> None:
    pivot = grid_df.pivot_table(
        values="advantage_over_truth",
        index="penalty_magnitude",
        columns="audit_probability",
        aggfunc="mean",
    )
    sns.set_theme(style="white")
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
    )
    ax.set_title("Deviator Advantage under Audit Frequency & Penalty Magnitude")
    ax.set_xlabel("Audit probability")
    ax.set_ylabel("Penalty magnitude")
    plt.tight_layout()
    plt.savefig(output_dir / "audit_penalty_heatmap.png", dpi=200)
    plt.close()


def plot_profile_rewards(profile_df: pd.DataFrame, output_dir: Path) -> None:
    subset = profile_df[profile_df["profile"].isin(["truthful_vs_truthful", "best_deviation_vs_truthful"])]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(
        data=subset,
        x="variant",
        y="mean_reward",
        hue="profile",
    )
    ax.set_title("Truthful vs. Best Deviation Rewards by Variant")
    ax.set_ylabel("Mean reward per interaction")
    ax.set_xlabel("Environment variant")
    plt.tight_layout()
    plt.savefig(output_dir / "profile_rewards.png", dpi=200)
    plt.close()


def main() -> None:
    raw_dir = Path("results/raw")
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    bias_df = pd.read_csv(raw_dir / "bias_sweep.csv")
    profile_df = pd.read_csv(raw_dir / "profiles.csv")
    grid_df = pd.read_csv(raw_dir / "audit_penalty_grid.csv")

    plot_bias_advantage(bias_df, plot_dir)
    plot_audit_penalty_heatmap(grid_df, plot_dir)
    plot_profile_rewards(profile_df, plot_dir)


if __name__ == "__main__":
    main()
