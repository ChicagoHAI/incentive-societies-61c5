## Research Question
Can formally engineered communication protocols with sanctioning and audit mechanisms make truthful uncertainty reporting a subgame-perfect equilibrium in repeated multi-agent interactions, while maintaining safety constraints on tool use?

## Background and Motivation
- Recent work on collaborative and federated learning (e.g., Dorner et al., 2023/2025) shows rational agents may manipulate updates when incentives allow, undermining team performance.
- Incentive-compatible data sharing schemes (Clinton et al., 2025) and inference-aided crowdsourcing incentives (Hu et al., 2018) demonstrate that proper scoring rules paired with discrepancy penalties can encourage honesty but typically assume exogenous auditors.
- Assurance audit frameworks (Lam et al., 2024) emphasize structured, criteria-based audits yet do not formalize equilibrium properties under repeated interactions.
- Safety proposals for agentic LLMs highlight the need for abstention zones and predictive constraints; however, canonical references (e.g., Predictive Safety Networks) lack publicly available implementations, leaving a gap in integrating safety with incentive design.
- Establishing provable (or empirically supported) truthfulness guarantees in multi-agent epistemics could inform trustworthy AI deployments in high-stakes settings.

## Hypothesis Decomposition
1. **H1 (Truthful Equilibrium)**: In a repeated signaling game with proper scoring-rule rewards and probabilistic audits that levy long-run penalties, truthful uncertainty reports constitute a subgame-perfect equilibrium.
2. **H2 (Baseline Deviation)**: Without sanction or audit hooks, agents face incentives to exaggerate confidence, yielding higher expected rewards than truthful reporting.
3. **H3 (Safety Constraint Integration)**: Adding abstention requirements (analogous to predictive safety networks) maintains equilibrium truthfulness while reducing unsafe tool activations, provided penalties are calibrated.

Independent variables:
- Presence/absence of sanctions (penalty magnitude, audit frequency).
- Communication protocol parameters (message granularity, required uncertainty reports).
- Safety constraint toggles (mandatory abstention zone thresholds).

Dependent variables:
- Truthfulness metrics (expected Brier score gap vs. ground truth).
- Incentive alignment metrics (expected cumulative utility per agent).
- Safety metrics (rate of abstention violations, unsafe tool invocations).

Success criteria:
- Demonstrate truthful strategy yields higher (or equal) expected utility than deviations in sanction-enabled environment.
- Show statistically significant incentive gap favoring misreporting in baseline environment.
- Confirm safety constraints do not break truthfulness equilibrium while reducing unsafe actions.

## Proposed Methodology

### Approach
Use synthetic repeated multi-agent signaling games to model agents sharing probabilistic claims about hidden world states. Implement parameterized agent strategies (truthful, overconfident, underconfident, adaptive best-response) and compare payoffs under different environment designs (baseline vs. sanction/audit vs. safety-augmented). Analytical equilibrium checks will be complemented by empirical reinforcement-learning or best-response simulations.

### Experimental Steps
1. **Formalize Game Environment**: Define per-round payoff structure combining proper scoring rules, communication costs, and sanction penalties (Monte Carlo simulated for tractability).
2. **Baseline Analysis**: Compute expected utilities for candidate strategy profiles without sanctions; verify deviations outperform truthful reports.
3. **Sanction Mechanism Evaluation**: Introduce audit sampling and long-run penalty accumulation; run dynamic programming / iterative best-response to assess equilibrium stability.
4. **Safety Constraint Layer**: Add abstention threshold and tool activation constraints; evaluate with same agents to gauge impact on truthfulness and safety metrics.
5. **Sensitivity & Robustness**: Sweep audit frequency, penalty magnitude, and abstention bandwidth to map parameter regimes where truthfulness is stable.
6. **Statistical Testing & Visualization**: Aggregate simulation results, compute means/variances, conduct paired tests, and visualize payoff landscapes.

### Baselines
- **No Sanction**: Proper scoring-rule reward only (log score / Brier score). Tests whether truthfulness holds without penalties.
- **Static Penalty**: Immediate penalty for detected lies without cumulative tracking (to see if long-run penalties matter).
- **Random Strategy**: Agents reporting random beliefs to confirm evaluation pipeline.

### Evaluation Metrics
- **Truthfulness Error**: Mean absolute deviation between reported and true probabilities; Brier score difference relative to truthful reporting.
- **Utility Gap**: Expected cumulative utility difference between truthful strategy and best deviating strategy.
- **Audit Efficiency**: Fraction of audits detecting misreports; time to penalize deceptive agents.
- **Safety Violations**: Count/rate of abstention breaches or unsafe actions.

### Statistical Analysis Plan
- Perform Monte Carlo simulations (≥10k episodes per condition) to estimate means and standard errors.
- Use paired t-tests or Wilcoxon signed-rank tests (if non-normal) comparing truthful vs. deviating utilities under each mechanism.
- Apply bootstrap confidence intervals (95%) for key metrics.
- Conduct regression/sensitivity analysis to relate audit frequency to utility gap.

## Expected Outcomes
- **Support H1**: Sanctioned environments should yield truthful reporting as dominant/best-response strategy with non-positive deviation gains.
- **Support H2**: Baseline environment should display positive deviation gains for overconfident policies.
- **Support H3**: Safety layer should reduce unsafe actions with minimal impact on truthfulness metrics if penalties tuned correctly.
- Negative or ambiguous results will highlight parameter regimes where mechanisms fail or create perverse incentives.

## Timeline and Milestones
- **0.5 hr**: Finalize formal model, parameter ranges, and simulation scaffolding.
- **1.0 hr**: Implement baseline environment and agent strategies; validate against analytical expectations.
- **1.0 hr**: Add sanction/audit mechanics; run core comparative simulations.
- **0.5 hr**: Integrate safety/abstention layer and rerun key experiments.
- **0.5 hr**: Sensitivity sweeps & statistical analysis.
- **0.5 hr**: Documentation (REPORT.md, README) and visualization assembly.
- Buffer (~20%): 0.4 hr reserved for debugging or reruns.

## Potential Challenges
- **Analytical complexity**: Proving subgame perfection may be intractable; rely on empirical best-response approximations and document limitation.
- **Simulation stability**: Ensuring convergence of adaptive strategies (e.g., Q-learning) on CPU might be slow; plan for simpler best-response enumeration if needed.
- **Parameter explosion**: Large sweep space could exceed time budget; prioritize most informative ranges (e.g., audit probability ∈ {0, 0.1, 0.3, 0.5}).
- **Safety layer calibration**: Setting abstention penalties too high could trivialize agent behavior; will iteratively adjust while monitoring metrics.

## Success Criteria
- Complete end-to-end experiments demonstrating mechanisms and documenting parameter regimes.
- Provide statistical evidence (confidence intervals, hypothesis tests) supporting or refuting sub-hypotheses.
- Deliver reproducible code, datasets (synthetic generation scripts), and comprehensive REPORT.md aligning with methodology requirements.
