"""Experiment orchestration for incentive-compatible signaling study."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .simulation import SimulationConfig, Strategy, simulate_profile, sweep_bias_grid


def _base_config() -> SimulationConfig:
    """Default environment shared across variants."""

    return SimulationConfig(
        n_trials=100_000,
        beta_alpha=2.0,
        beta_beta=3.0,
        scoring_weight=1.0,
        action_threshold=0.62,
        influence_bonus=0.6,
        tool_success_reward=1.8,
        tool_failure_cost=-2.2,
        audit_probability=0.0,
        penalty_magnitude=0.0,
        penalty_tolerance=0.05,
        safety_enabled=False,
        safety_lower=0.45,
        safety_upper=0.60,
        safety_penalty=2.5,
    )


def _variant_configs() -> List[Tuple[str, SimulationConfig]]:
    base = _base_config()
    return [
        ("baseline", base),
        (
            "sanction",
            replace(
                base,
                audit_probability=0.3,
                penalty_magnitude=3.5,
            ),
        ),
        (
            "sanction_safety",
            replace(
                base,
                audit_probability=0.3,
                penalty_magnitude=3.5,
                safety_enabled=True,
            ),
        ),
    ]


def _audit_penalty_grid() -> List[Dict[str, float]]:
    audit_probs = [0.0, 0.1, 0.3, 0.5]
    penalties = [0.0, 1.5, 3.5, 5.0]
    return [{"audit_probability": ap, "penalty_magnitude": pm} for ap, pm in product(audit_probs, penalties)]


def run_experiments(seed: int = 7, output_dir: Path | None = None) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    truthful = Strategy(name="truthful", bias=0.0, comply_with_safety=True)
    variants = _variant_configs()

    profile_records: List[Dict[str, object]] = []
    bias_sweep_frames: List[pd.DataFrame] = []

    for label, config in variants:
        # Simulate truthful vs truthful profile for reference.
        truthful_rng = np.random.default_rng(rng.integers(0, 1_000_000))
        profile_truth = simulate_profile(config, [truthful, truthful], truthful_rng)
        profile_records.append(
            {
                "variant": label,
                "profile": "truthful_vs_truthful",
                "agent": "agent_0",
                "mean_reward": profile_truth["agent_0"]["mean_reward"],
                "unsafe_rate": profile_truth["agent_0"]["unsafe_rate"],
                "advantage_over_truth": 0.0,
                "action_rate": profile_truth["environment"]["action_rate"],
                "config": json.dumps(profile_truth["environment"]["config"]),
            }
        )

        # Evaluate deviation incentives.
        bias_values = [-0.30, -0.15, 0.0, 0.15, 0.30]
        compliance_flags = [True, False] if config.safety_enabled else [True]
        sweep_rng = np.random.default_rng(rng.integers(0, 1_000_000))
        sweep_df = sweep_bias_grid(config, truthful, bias_values, compliance_flags, sweep_rng)
        sweep_df["variant"] = label
        bias_sweep_frames.append(sweep_df)

        best_row = sweep_df.loc[sweep_df["utility"].idxmax()]
        profile_records.append(
            {
                "variant": label,
                "profile": "best_deviation_vs_truthful",
                "agent": "agent_0",
                "mean_reward": best_row["utility"],
                "unsafe_rate": best_row["unsafe_action_rate"],
                "advantage_over_truth": best_row["advantage_over_truth"],
                "action_rate": best_row["variant_action_rate"],
                "config": json.dumps(asdict(config)),
                "strategy_bias": best_row["bias"],
                "strategy_comply": bool(best_row["comply_with_safety"]),
            }
        )

    bias_grid_results = pd.concat(bias_sweep_frames, ignore_index=True)
    profile_results = pd.DataFrame.from_records(profile_records)

    # Sensitivity grid over audit probability and penalty magnitude.
    grid_records: List[Dict[str, object]] = []
    for overrides in _audit_penalty_grid():
        grid_config = replace(_base_config(), **overrides)
        grid_rng = np.random.default_rng(rng.integers(0, 1_000_000))
        sweep = sweep_bias_grid(grid_config, truthful, [-0.3, -0.15, 0.0, 0.15, 0.3], [True], grid_rng)
        best = sweep.loc[sweep["utility"].idxmax()]
        grid_records.append(
            {
                "audit_probability": overrides["audit_probability"],
                "penalty_magnitude": overrides["penalty_magnitude"],
                "best_bias": best["bias"],
                "advantage_over_truth": best["advantage_over_truth"],
                "unsafe_action_rate": best["unsafe_action_rate"],
            }
        )

    grid_results = pd.DataFrame.from_records(grid_records)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        bias_grid_results.to_csv(output_dir / "bias_sweep.csv", index=False)
        profile_results.to_csv(output_dir / "profiles.csv", index=False)
        grid_results.to_csv(output_dir / "audit_penalty_grid.csv", index=False)

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": seed,
                    "variants": [label for label, _ in variants],
                    "bias_grid_shape": list(bias_grid_results.shape),
                    "profile_shape": list(profile_results.shape),
                },
                f,
                indent=2,
            )

    return {
        "bias_sweep": bias_grid_results,
        "profiles": profile_results,
        "audit_penalty_grid": grid_results,
    }


def main() -> None:
    output_base = Path("results/raw")
    frames = run_experiments(seed=42, output_dir=output_base)
    # Convenience printout for quick inspection.
    for name, frame in frames.items():
        print(f"{name}: shape={frame.shape}")
        print(frame.head())


if __name__ == "__main__":
    main()
