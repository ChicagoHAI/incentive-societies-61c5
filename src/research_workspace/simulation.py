"""Core simulation utilities for incentive-compatible signaling environments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the repeated signaling environment."""

    n_trials: int
    beta_alpha: float
    beta_beta: float
    scoring_weight: float
    action_threshold: float
    influence_bonus: float
    tool_success_reward: float
    tool_failure_cost: float
    audit_probability: float
    penalty_magnitude: float
    penalty_tolerance: float
    safety_enabled: bool
    safety_lower: float
    safety_upper: float
    safety_penalty: float


@dataclass(frozen=True)
class Strategy:
    """Parameterized reporting strategy."""

    name: str
    bias: float = 0.0
    comply_with_safety: bool = True
    clip_min: float = 0.01
    clip_max: float = 0.99

    def decide(
        self,
        probs: np.ndarray,
        safety_required: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return reports and abstention flags for a batch of latent probabilities."""

        abstain = safety_required & self.comply_with_safety
        reports = np.clip(probs + self.bias, self.clip_min, self.clip_max)
        # Represent abstention with NaNs to simplify downstream averaging.
        reports = np.where(abstain, np.nan, reports)
        return reports, abstain


def _draw_latent_events(config: SimulationConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    probs = rng.beta(config.beta_alpha, config.beta_beta, size=config.n_trials)
    outcomes = rng.random(config.n_trials) < probs
    return probs, outcomes.astype(np.int8)


def _aggregator_action(
    reports: np.ndarray,
    action_threshold: float,
) -> np.ndarray:
    """Decide whether to deploy the tool based on mean reported probability."""

    valid_counts = np.sum(~np.isnan(reports), axis=0)
    summed = np.nansum(reports, axis=0)
    mean_reports = np.divide(
        summed,
        valid_counts,
        out=np.full_like(summed, -np.inf, dtype=float),
        where=valid_counts > 0,
    )
    return mean_reports >= action_threshold


def _audit_penalties(
    reports: np.ndarray,
    true_probs: np.ndarray,
    audit_probability: float,
    penalty_magnitude: float,
    penalty_tolerance: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute audit penalties and hits for each agent."""

    needs_audit = ~np.isnan(reports)
    random_draws = rng.random(reports.shape)
    triggered = (random_draws < audit_probability) & needs_audit

    diff = np.abs(reports - true_probs)
    diff = np.where(np.isnan(diff), 0.0, diff)
    excessive = diff > penalty_tolerance
    penalties = np.where(triggered & excessive, penalty_magnitude * (diff - penalty_tolerance), 0.0)

    audit_hits = triggered & excessive
    return penalties, audit_hits, triggered


def simulate_profile(
    config: SimulationConfig,
    strategies: Iterable[Strategy],
    rng: np.random.Generator,
) -> Dict[str, pd.Series]:
    """Simulate a profile of strategies and return aggregate per-agent metrics."""

    strategies = list(strategies)
    probs, outcomes = _draw_latent_events(config, rng)
    safety_required = np.zeros_like(probs, dtype=bool)
    if config.safety_enabled:
        safety_required = (probs >= config.safety_lower) & (probs <= config.safety_upper)

    reports = []
    abstains = []
    for strat in strategies:
        rep, abstain = strat.decide(probs, safety_required)
        reports.append(rep)
        abstains.append(abstain)

    report_matrix = np.vstack(reports)
    abstain_matrix = np.vstack(abstains)

    action = _aggregator_action(report_matrix, config.action_threshold)
    influence_reward = config.influence_bonus * action.astype(float)

    tool_reward = np.where(outcomes, config.tool_success_reward, config.tool_failure_cost)
    tool_reward = tool_reward * action.astype(float)

    # Score rewards only when an agent submits a report.
    outcome_matrix = np.broadcast_to(outcomes, report_matrix.shape)
    scoring_reward = config.scoring_weight * (1.0 - (report_matrix - outcome_matrix) ** 2)
    scoring_reward = np.where(np.isnan(report_matrix), 0.0, scoring_reward)

    penalties, audit_hits, audit_trigger = _audit_penalties(
        report_matrix,
        probs,
        config.audit_probability,
        config.penalty_magnitude,
        config.penalty_tolerance,
        rng,
    )

    safety_violation = np.zeros_like(report_matrix, dtype=bool)
    if config.safety_enabled:
        # Unsafe if abstention required but the agent still reported.
        violation_mask = np.broadcast_to(safety_required, report_matrix.shape) & ~np.isnan(report_matrix)
        safety_violation = violation_mask

    safety_cost = np.zeros_like(report_matrix, dtype=float)
    unsafe_action = np.zeros_like(report_matrix, dtype=bool)
    if config.safety_enabled and config.safety_penalty > 0:
        unsafe_action = safety_violation & action
        safety_cost = np.where(safety_violation, config.safety_penalty, 0.0)

    total_reward = (
        scoring_reward
        + influence_reward
        + tool_reward
        - penalties
        - safety_cost
    )

    results: Dict[str, pd.Series] = {}
    for idx, strat in enumerate(strategies):
        prefix = f"agent_{idx}"
        data = {
            "strategy": strat.name,
            "mean_reward": float(np.mean(total_reward[idx])),
            "mean_scoring_reward": float(np.mean(scoring_reward[idx])),
            "mean_influence_reward": float(np.mean(influence_reward)),
            "mean_tool_reward": float(np.mean(tool_reward)),
            "mean_penalty": float(np.mean(penalties[idx])),
            "audit_rate": float(np.mean(audit_trigger[idx])),
            "audit_hits": int(np.sum(audit_hits[idx])),
            "unsafe_rate": float(np.mean(safety_violation[idx])) if config.safety_enabled else 0.0,
            "unsafe_action_rate": float(np.mean(unsafe_action[idx]))
            if config.safety_enabled
            else 0.0,
            "reports_submitted": int(np.sum(~np.isnan(report_matrix[idx]))),
        }
        results[prefix] = pd.Series(data)

    aggregate = {
        "variant_reward_mean": float(np.mean(total_reward)),
        "action_rate": float(np.mean(action)),
        "mean_prob": float(np.mean(probs)),
        "mean_outcome": float(np.mean(outcomes)),
        "config": asdict(config),
    }
    results["environment"] = pd.Series(aggregate)
    return results


def sweep_bias_grid(
    config: SimulationConfig,
    truthful_strategy: Strategy,
    bias_values: Iterable[float],
    compliance_flags: Iterable[bool],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Evaluate deviation incentives over a grid of biases and compliance choices."""

    records: List[Dict[str, object]] = []
    baseline = simulate_profile(config, [truthful_strategy, truthful_strategy], rng)
    baseline_util = baseline["agent_0"]["mean_reward"]

    for bias in bias_values:
        for comply in compliance_flags:
            name = f"bias_{bias:+.2f}{'_comply' if comply else '_defy'}"
            deviating_strategy = Strategy(name=name, bias=bias, comply_with_safety=comply)
            profile_results = simulate_profile(config, [deviating_strategy, truthful_strategy], rng)
            util = profile_results["agent_0"]["mean_reward"]
            records.append(
                {
                    "bias": bias,
                    "comply_with_safety": comply,
                    "utility": util,
                    "advantage_over_truth": util - baseline_util,
                    "audit_hits": profile_results["agent_0"]["audit_hits"],
                    "unsafe_action_rate": profile_results["agent_0"]["unsafe_action_rate"],
                    "reports_submitted": profile_results["agent_0"]["reports_submitted"],
                    "variant_action_rate": profile_results["environment"]["action_rate"],
                }
            )

    return pd.DataFrame.from_records(records)
