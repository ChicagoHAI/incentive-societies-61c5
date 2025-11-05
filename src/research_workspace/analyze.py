"""Statistical analysis of simulation outputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .experiments import _variant_configs
from .simulation import SimulationConfig, Strategy, simulate_profile


def _strategy_from_profile_row(row: pd.Series) -> Strategy:
    return Strategy(
        name=f"bias_{row['strategy_bias']:+.2f}",
        bias=float(row["strategy_bias"]),
        comply_with_safety=bool(row.get("strategy_comply", True)),
    )


def evaluate_variant(
    label: str,
    config: SimulationConfig,
    truthful: Strategy,
    deviating: Strategy,
    repeats: int,
    trials_per_repeat: int,
    seed: int,
) -> Dict[str, object]:
    truthful_rewards = []
    deviating_rewards = []

    for idx in range(repeats):
        base_seed = seed + idx * 31
        truth_rng = np.random.default_rng(base_seed)
        dev_rng = np.random.default_rng(base_seed + 13)

        truth_result = simulate_profile(replace(config, n_trials=trials_per_repeat), [truthful, truthful], truth_rng)
        dev_result = simulate_profile(replace(config, n_trials=trials_per_repeat), [deviating, truthful], dev_rng)

        truthful_rewards.append(truth_result["agent_0"]["mean_reward"])
        deviating_rewards.append(dev_result["agent_0"]["mean_reward"])

    truth_arr = np.array(truthful_rewards)
    dev_arr = np.array(deviating_rewards)
    diff = dev_arr - truth_arr

    t_stat, p_value = stats.ttest_1samp(diff, popmean=0.0)
    ci_low, ci_high = np.percentile(diff, [2.5, 97.5])

    return {
        "variant": label,
        "mean_truth_reward": float(np.mean(truth_arr)),
        "mean_deviation_reward": float(np.mean(dev_arr)),
        "mean_advantage": float(np.mean(diff)),
        "std_advantage": float(np.std(diff, ddof=1)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "repeats": repeats,
        "trials_per_repeat": trials_per_repeat,
    }


def run_analysis(
    repeats: int = 200,
    trials_per_repeat: int = 20_000,
    seed: int = 99,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    truthful = Strategy(name="truthful", bias=0.0, comply_with_safety=True)
    profile_df = pd.read_csv(Path("results/raw/profiles.csv"))
    best_profiles = profile_df[profile_df["profile"] == "best_deviation_vs_truthful"].set_index("variant")

    records = []
    diff_store: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for label, config in _variant_configs():
        best_row = best_profiles.loc[label]
        deviating = _strategy_from_profile_row(best_row)
        stats_row = evaluate_variant(
            label=label,
            config=config,
            truthful=truthful,
            deviating=deviating,
            repeats=repeats,
            trials_per_repeat=trials_per_repeat,
            seed=seed,
        )
        records.append(stats_row)

    stats_df = pd.DataFrame.from_records(records)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(output_dir / "advantage_stats.csv", index=False)
        with open(output_dir / "analysis_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "repeats": repeats,
                    "trials_per_repeat": trials_per_repeat,
                    "seed": seed,
                },
                f,
                indent=2,
            )

    return stats_df


def main() -> None:
    stats_df = run_analysis(output_dir=Path("results/analysis"))
    print(stats_df)


if __name__ == "__main__":
    main()
