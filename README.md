# RL Assignment 1 — Multi-Armed Bandit with Costs

Implementation of ε-Greedy and UCB bandit algorithms on a 5-arm environment where each arm carries a fixed cost. Covers stationary and non-stationary settings, regret analysis, and algorithm comparison.

## Environment

| Arm | Reward Mean | Std | Cost | Net Mean |
|-----|------------|-----|------|----------|
| 0   | 0.8        | 0.1 | 0.20 | 0.60     |
| 1   | 0.6        | 0.1 | 0.10 | 0.50     |
| 2   | 0.9        | 0.1 | 0.30 | **0.60** |
| 3   | 0.4        | 0.1 | 0.05 | 0.35     |
| 4   | 0.7        | 0.1 | 0.15 | 0.55     |

Optimal arm: 0 or 2 (both net mean = 0.60).

## Algorithms

- **ε-Greedy** — explores uniformly at rate ε, exploits the current best estimate otherwise. Updates Q values with an incremental mean.
- **UCB** — selects arms via `Q + c * sqrt(log(t) / N)`, balancing exploitation with optimism in the face of uncertainty.

## Plots

| File | Description |
|------|-------------|
| `plots/plot1_epsilon_comparison.png` | Cumulative net reward for ε ∈ {0.01, 0.1, 0.5} |
| `plots/plot2_modified_costs.png` | Effect of raising arm 2's cost from 0.3 → 0.8 |
| `plots/plot3_nonstationary_costs.png` | Static vs drifting costs (drift every 100 steps) |
| `plots/plot4_ucb_vs_egreedy.png` | UCB (c=2.0) vs ε-Greedy (ε=0.01, 0.1) |
| `plots/plot5_dynamic_rewards.png` | Static vs shifting reward means (shift every 200 steps) |
| `plots/plot6_regret.png` | Cumulative regret for ε ∈ {0.01, 0.1, 0.5} |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Generate all assignment plots:

```bash
python generate_plots.py
```

Run the interactive simulation (also generates plots):

```bash
python multi_armed_bandit.py
```

## Files

```
multi_armed_bandit.py   # Bandit classes + simulation logic
generate_plots.py       # Standalone script that saves all 6 plots
plots/                  # Generated figures
requirements.txt        # Python dependencies
```
