import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

np.random.seed(42)

# ── Base parameters ──────────────────────────────────────────────────────────
REWARD_MEANS = [0.8, 0.6, 0.9, 0.4, 0.7]
REWARD_STDS  = [0.1, 0.1, 0.1, 0.1, 0.1]
COSTS        = [0.2, 0.1, 0.3, 0.05, 0.15]
NUM_ARMS     = 5
NUM_STEPS    = 1000
NET_MEANS    = [m - c for m, c in zip(REWARD_MEANS, COSTS)]  # [0.6,0.5,0.6,0.35,0.55]
OPTIMAL_NET  = max(NET_MEANS)   # 0.6


# ── Environment helpers ───────────────────────────────────────────────────────
def get_reward_and_cost(arm, means=None, stds=None, costs=None):
    means = means or REWARD_MEANS
    stds  = stds  or REWARD_STDS
    costs = costs or COSTS
    return np.random.normal(means[arm], stds[arm]), costs[arm]


# ── ε-Greedy ──────────────────────────────────────────────────────────────────
class EpsilonGreedy:
    def __init__(self, k, epsilon):
        self.k, self.epsilon = k, epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def select(self):
        return np.random.randint(self.k) if np.random.rand() < self.epsilon else np.argmax(self.Q)

    def update(self, arm, net_reward):
        self.N[arm] += 1
        self.Q[arm] += (net_reward - self.Q[arm]) / self.N[arm]


# ── UCB ───────────────────────────────────────────────────────────────────────
class UCB:
    def __init__(self, k, c=2.0):
        self.k, self.c = k, c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select(self):
        self.t += 1
        untried = np.where(self.N == 0)[0]
        if len(untried):
            return untried[0]
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.argmax(ucb)

    def update(self, arm, net_reward):
        self.N[arm] += 1
        self.Q[arm] += (net_reward - self.Q[arm]) / self.N[arm]


# ── Run helpers ───────────────────────────────────────────────────────────────
def run_epsilon_greedy(epsilon, steps=NUM_STEPS, means=None, stds=None, costs=None):
    agent = EpsilonGreedy(NUM_ARMS, epsilon)
    cum = np.zeros(steps)
    for t in range(steps):
        arm = agent.select()
        r, c = get_reward_and_cost(arm, means, stds, costs)
        net = r - c
        agent.update(arm, net)
        cum[t] = (cum[t-1] + net) if t > 0 else net
    return cum

def run_ucb(c=2.0, steps=NUM_STEPS, means=None, stds=None, costs=None):
    agent = UCB(NUM_ARMS, c)
    cum = np.zeros(steps)
    for t in range(steps):
        arm = agent.select()
        r, cost = get_reward_and_cost(arm, means, stds, costs)
        net = r - cost
        agent.update(arm, net)
        cum[t] = (cum[t-1] + net) if t > 0 else net
    return cum


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – ε-Greedy with varying epsilon
# ═══════════════════════════════════════════════════════════════════════════════
epsilons = [0.01, 0.1, 0.5]
plt.figure(figsize=(10, 6))
for eps in epsilons:
    cum = run_epsilon_greedy(eps)
    plt.plot(cum, label=f"ε = {eps}")
plt.xlabel("Steps")
plt.ylabel("Cumulative Net Reward")
plt.title("ε-Greedy: Effect of Epsilon on Cumulative Net Reward")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot1_epsilon_comparison.png", dpi=150)
plt.close()
print("Saved plot1_epsilon_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – Modified costs (arm 2 made very expensive: cost 0.3 → 0.8)
# ═══════════════════════════════════════════════════════════════════════════════
modified_costs = [0.2, 0.1, 0.8, 0.05, 0.15]   # arm 2 cost raised to 0.8
plt.figure(figsize=(10, 6))
cum_orig = run_epsilon_greedy(0.1)
cum_mod  = run_epsilon_greedy(0.1, costs=modified_costs)
plt.plot(cum_orig, label="Original costs")
plt.plot(cum_mod,  label="Arm 2 cost = 0.8 (expensive)")
plt.xlabel("Steps")
plt.ylabel("Cumulative Net Reward")
plt.title("ε-Greedy: Impact of Making One Arm Significantly More Expensive")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot2_modified_costs.png", dpi=150)
plt.close()
print("Saved plot2_modified_costs.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Non-Stationary Costs (costs drift every 100 steps)
# ═══════════════════════════════════════════════════════════════════════════════
def run_nonstationary_costs(epsilon=0.1, steps=NUM_STEPS, drift_std=0.05, drift_interval=100):
    """Costs drift randomly every drift_interval steps."""
    agent = EpsilonGreedy(NUM_ARMS, epsilon)
    costs = np.array(COSTS, dtype=float)
    cum = np.zeros(steps)
    for t in range(steps):
        if t > 0 and t % drift_interval == 0:
            costs = np.clip(costs + np.random.normal(0, drift_std, NUM_ARMS), 0, None)
        arm = agent.select()
        r = np.random.normal(REWARD_MEANS[arm], REWARD_STDS[arm])
        net = r - costs[arm]
        agent.update(arm, net)
        cum[t] = (cum[t-1] + net) if t > 0 else net
    return cum

plt.figure(figsize=(10, 6))
cum_static = run_epsilon_greedy(0.1)
cum_ns     = run_nonstationary_costs(0.1)
plt.plot(cum_static, label="Static costs (ε=0.1)")
plt.plot(cum_ns,     label="Non-stationary costs (drift every 100 steps)")
plt.xlabel("Steps")
plt.ylabel("Cumulative Net Reward")
plt.title("ε-Greedy: Static vs Non-Stationary Costs")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot3_nonstationary_costs.png", dpi=150)
plt.close()
print("Saved plot3_nonstationary_costs.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 – UCB vs ε-Greedy comparison
# ═══════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 6))
for eps in [0.01, 0.1]:
    cum = run_epsilon_greedy(eps)
    plt.plot(cum, label=f"ε-Greedy (ε={eps})")
cum_ucb = run_ucb(c=2.0)
plt.plot(cum_ucb, label="UCB (c=2.0)", linestyle="--")
plt.xlabel("Steps")
plt.ylabel("Cumulative Net Reward")
plt.title("UCB vs ε-Greedy: Cumulative Net Reward Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot4_ucb_vs_egreedy.png", dpi=150)
plt.close()
print("Saved plot4_ucb_vs_egreedy.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 – Dynamic Rewards (mean rewards shift every 200 steps)
# ═══════════════════════════════════════════════════════════════════════════════
def run_dynamic_rewards(epsilon=0.1, steps=NUM_STEPS, shift_interval=200, shift_std=0.2):
    """Mean rewards shift periodically every shift_interval steps."""
    agent = EpsilonGreedy(NUM_ARMS, epsilon)
    means = np.array(REWARD_MEANS, dtype=float)
    cum = np.zeros(steps)
    for t in range(steps):
        if t > 0 and t % shift_interval == 0:
            means = np.clip(means + np.random.normal(0, shift_std, NUM_ARMS), 0, None)
        arm = agent.select()
        r = np.random.normal(means[arm], REWARD_STDS[arm])
        net = r - COSTS[arm]
        agent.update(arm, net)
        cum[t] = (cum[t-1] + net) if t > 0 else net
    return cum

plt.figure(figsize=(10, 6))
cum_static  = run_epsilon_greedy(0.1)
cum_dynamic = run_dynamic_rewards(0.1)
plt.plot(cum_static,  label="Static rewards (ε=0.1)")
plt.plot(cum_dynamic, label="Dynamic rewards (shift every 200 steps)")
plt.xlabel("Steps")
plt.ylabel("Cumulative Net Reward")
plt.title("ε-Greedy: Static vs Dynamic Reward Distributions")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot5_dynamic_rewards.png", dpi=150)
plt.close()
print("Saved plot5_dynamic_rewards.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 – Regret over time for ε-Greedy
# ═══════════════════════════════════════════════════════════════════════════════
def run_epsilon_greedy_regret(epsilon, steps=NUM_STEPS):
    agent = EpsilonGreedy(NUM_ARMS, epsilon)
    regret = np.zeros(steps)
    cumulative_regret = 0
    for t in range(steps):
        arm = agent.select()
        r, c = get_reward_and_cost(arm)
        net = r - c
        agent.update(arm, net)
        # regret = optimal expected net - actual net received
        cumulative_regret += OPTIMAL_NET - net
        regret[t] = cumulative_regret
    return regret

plt.figure(figsize=(10, 6))
for eps in [0.01, 0.1, 0.5]:
    regret = run_epsilon_greedy_regret(eps)
    plt.plot(regret, label=f"ε = {eps}")
plt.xlabel("Steps")
plt.ylabel("Cumulative Regret")
plt.title("ε-Greedy: Cumulative Regret over Time")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot6_regret.png", dpi=150)
plt.close()
print("Saved plot6_regret.png")

print("\nAll plots saved to ./plots/")
