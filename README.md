## Incentive-Compatible Societies Workspace

- Synthetic signaling-game study evaluating how audits, sanctions, and safety abstention zones affect truthful probability reporting in multi-agent settings.
- Baseline environment incentivizes +0.15 overconfidence, yielding +0.044 utility advantage and +9 pp tool activation; sanction + safety mechanisms shrink the advantage to ≈0 with p>0.2.
- Audit probability ≥0.3 paired with penalty ≥3.5 is sufficient to flip deviation incentives; safety penalties eliminate gains from ignoring abstention rules.

### Reproducing the Experiments
1. Create and activate the virtual environment (already provisioned):
   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. Install dependencies (if fresh clone):
   ```bash
   uv sync
   ```
3. Run core simulations and generate artifacts:
   ```bash
   python -m research_workspace.experiments
   python -m research_workspace.visualize
   python -m research_workspace.analyze
   ```

### Project Structure
- `src/research_workspace/simulation.py` – Vectorized environment simulator and strategy grid utilities.
- `src/research_workspace/experiments.py` – Orchestrates variants, saves raw metrics, and parameter sweeps.
- `src/research_workspace/visualize.py` – Produces comparative plots in `results/plots/`.
- `src/research_workspace/analyze.py` – Bootstrap-style statistical validation of deviation advantages.
- `results/` – Raw CSV outputs, plots, and analysis summaries (see `REPORT.md` for interpretation).

Full methodology, results, and references are detailed in `REPORT.md`.
