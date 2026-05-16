import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Environment parameters ────────────────────────────────────────────────────

REWARD_MEANS = [0.8, 0.6, 0.9, 0.4, 0.7]
REWARD_STDS  = [0.1, 0.1, 0.1, 0.1, 0.1]
COSTS        = [0.2, 0.1, 0.3, 0.05, 0.15]
NET_MEANS    = [m - c for m, c in zip(REWARD_MEANS, COSTS)]
OPTIMAL_NET  = max(NET_MEANS)   # 0.6  (arms 0 and 2 are tied)


# ── Bandit agents ─────────────────────────────────────────────────────────────

class EpsilonGreedyBandit:
    def __init__(self, epsilon=0, k=5):
        """
        Initializes the bandit with the given parameters.

        Parameters:
            epsilon (float): The exploration rate (0 ≤ ε ≤ 1).
            k (int): The number of arms (actions).
        """
        self.k = k              # Number of arms
        self.epsilon = epsilon  # Exploration rate
        self.Q = np.zeros(k)    # Estimated values of each arm
        self.N = np.zeros(k)    # Number of times each arm has been selected

    def select_action(self):
        """
        Selects an action using the ε-greedy strategy.

        Returns:
            action (int): The index of the selected arm.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)    # Explore: select a random action
        else:
            return np.argmax(self.Q)            # Exploit: select the action with the highest estimated value

    def update(self, action, reward):
        """
        Updates the estimated value of the selected action based on the received reward.

        Parameters:
            action (int): The index of the selected arm.
            reward (float): The reward received from the environment after taking the action.
        """
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]  # Update estimated value using incremental formula


class UCBBandit:
    """Upper Confidence Bound bandit."""

    def __init__(self, k=5, c=2.0):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_action(self):
        self.t += 1
        untried = np.where(self.N == 0)[0]
        if len(untried):
            return untried[0]
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


# ── Environment ───────────────────────────────────────────────────────────────

def get_reward_and_cost(arm):
    """
    Simulates the environment by returning the reward and cost for a given arm.

    Parameters:
        arm (int): The selected arm (route).

    Returns:
        reward (float): Sampled reward from the arm's reward distribution.
        cost (float): Fixed cost of the arm.
    """
    reward = np.random.normal(REWARD_MEANS[arm], REWARD_STDS[arm])
    return reward, COSTS[arm]


def get_reward_and_cost_custom(arm, means, stds, costs):
    """Like get_reward_and_cost but with caller-supplied parameters."""
    return np.random.normal(means[arm], stds[arm]), costs[arm]


# ── Algorithm runners ─────────────────────────────────────────────────────────

def epsilon_greedy_with_costs(num_arms, num_steps, epsilon):
    """
    Runs the ε-greedy algorithm while accounting for costs.

    Parameters:
        num_arms (int): Number of arms (routes).
        num_steps (int): Number of time steps to run the algorithm.
        epsilon (float): Exploration rate.

    Returns:
        cumulative_net_rewards (numpy array): Cumulative net rewards over time.
    """
    bandit = EpsilonGreedyBandit(epsilon=epsilon, k=num_arms)
    cumulative_net_rewards = np.zeros(num_steps)

    for step in range(num_steps):
        action = bandit.select_action()             # Select an arm
        reward, cost = get_reward_and_cost(action)  # Get reward and cost from the environment
        net_reward = reward - cost                  # Calculate net reward
        bandit.update(action, net_reward)           # Update the bandit's estimates
        cumulative_net_rewards[step] = cumulative_net_rewards[step-1] + net_reward if step > 0 else net_reward

    return cumulative_net_rewards


def epsilon_greedy_custom(num_arms, num_steps, epsilon, means, stds, costs):
    """ε-Greedy run with custom environment parameters."""
    bandit = EpsilonGreedyBandit(epsilon=epsilon, k=num_arms)
    cumulative = np.zeros(num_steps)
    for step in range(num_steps):
        action = bandit.select_action()
        r, c = get_reward_and_cost_custom(action, means, stds, costs)
        net = r - c
        bandit.update(action, net)
        cumulative[step] = (cumulative[step - 1] + net) if step > 0 else net
    return cumulative


def ucb_with_costs(num_arms, num_steps, c=2.0):
    """Runs the UCB algorithm with the standard cost environment."""
    bandit = UCBBandit(k=num_arms, c=c)
    cumulative = np.zeros(num_steps)
    for step in range(num_steps):
        action = bandit.select_action()
        reward, cost = get_reward_and_cost(action)
        net = reward - cost
        bandit.update(action, net)
        cumulative[step] = (cumulative[step - 1] + net) if step > 0 else net
    return cumulative


def run_nonstationary_costs(num_arms, num_steps, epsilon, drift_std=0.05, drift_interval=100):
    """ε-Greedy where costs drift by N(0, drift_std) every drift_interval steps."""
    bandit = EpsilonGreedyBandit(epsilon=epsilon, k=num_arms)
    costs = np.array(COSTS, dtype=float)
    cumulative = np.zeros(num_steps)
    for step in range(num_steps):
        if step > 0 and step % drift_interval == 0:
            costs = np.clip(costs + np.random.normal(0, drift_std, num_arms), 0, None)
        action = bandit.select_action()
        r = np.random.normal(REWARD_MEANS[action], REWARD_STDS[action])
        net = r - costs[action]
        bandit.update(action, net)
        cumulative[step] = (cumulative[step - 1] + net) if step > 0 else net
    return cumulative


def run_dynamic_rewards(num_arms, num_steps, epsilon, shift_std=0.2, shift_interval=200):
    """ε-Greedy where mean rewards shift by N(0, shift_std) every shift_interval steps."""
    bandit = EpsilonGreedyBandit(epsilon=epsilon, k=num_arms)
    means = np.array(REWARD_MEANS, dtype=float)
    cumulative = np.zeros(num_steps)
    for step in range(num_steps):
        if step > 0 and step % shift_interval == 0:
            means = np.clip(means + np.random.normal(0, shift_std, num_arms), 0, None)
        action = bandit.select_action()
        r = np.random.normal(means[action], REWARD_STDS[action])
        net = r - COSTS[action]
        bandit.update(action, net)
        cumulative[step] = (cumulative[step - 1] + net) if step > 0 else net
    return cumulative


def compute_regret(num_arms, num_steps, epsilon):
    """Returns cumulative regret array for ε-Greedy vs the optimal arm."""
    bandit = EpsilonGreedyBandit(epsilon=epsilon, k=num_arms)
    regret = np.zeros(num_steps)
    cum_regret = 0.0
    for step in range(num_steps):
        action = bandit.select_action()
        reward, cost = get_reward_and_cost(action)
        net = reward - cost
        bandit.update(action, net)
        cum_regret += OPTIMAL_NET - net
        regret[step] = cum_regret
    return regret


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cumulative_net_rewards_for_epsilons(num_arms, num_steps, epsilons):
    """
    Runs the ε-greedy algorithm for multiple epsilon values and plots the
    cumulative net rewards on the same figure.

    Parameters:
        num_arms (int): Number of arms (routes).
        num_steps (int): Number of time steps to run each simulation.
        epsilons (list[float]): Exploration rates to compare.
    """
    plt.figure(figsize=(10, 6))

    for epsilon in epsilons:
        cumulative_net_rewards = epsilon_greedy_with_costs(num_arms, num_steps, epsilon)
        plt.plot(cumulative_net_rewards, label=f"epsilon={epsilon}")

    plt.xlabel("Steps")
    plt.ylabel("Cumulative Net Reward")
    plt.title("ε-Greedy Algorithm with Costs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_assignment_plots(num_arms=5, num_steps=1000):
    """Generate and save all plots needed for the assignment answers."""
    os.makedirs("plots", exist_ok=True)
    np.random.seed(42)

    # Plot 1: ε comparison (Q2a)
    plt.figure(figsize=(10, 6))
    for eps in [0.01, 0.1, 0.5]:
        plt.plot(epsilon_greedy_with_costs(num_arms, num_steps, eps), label=f"ε = {eps}")
    plt.xlabel("Steps"); plt.ylabel("Cumulative Net Reward")
    plt.title("ε-Greedy: Effect of Epsilon on Cumulative Net Reward")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots/plot1_epsilon_comparison.png", dpi=150); plt.close()

    # Plot 2: modified costs (Q2b) — arm 2 cost raised from 0.3 to 0.8
    modified_costs = [0.2, 0.1, 0.8, 0.05, 0.15]
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_greedy_with_costs(num_arms, num_steps, 0.1), label="Original costs")
    plt.plot(
        epsilon_greedy_custom(num_arms, num_steps, 0.1, REWARD_MEANS, REWARD_STDS, modified_costs),
        label="Arm 2 cost = 0.8 (very expensive)"
    )
    plt.xlabel("Steps"); plt.ylabel("Cumulative Net Reward")
    plt.title("ε-Greedy: Impact of Making One Arm More Expensive")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots/plot2_modified_costs.png", dpi=150); plt.close()

    # Plot 3: non-stationary costs (Q3a)
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_greedy_with_costs(num_arms, num_steps, 0.1), label="Static costs (ε=0.1)")
    plt.plot(run_nonstationary_costs(num_arms, num_steps, 0.1),
             label="Non-stationary costs (drift every 100 steps)")
    plt.xlabel("Steps"); plt.ylabel("Cumulative Net Reward")
    plt.title("ε-Greedy: Static vs Non-Stationary Costs")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots/plot3_nonstationary_costs.png", dpi=150); plt.close()

    # Plot 4: UCB vs ε-Greedy (Q3b)
    plt.figure(figsize=(10, 6))
    for eps in [0.01, 0.1]:
        plt.plot(epsilon_greedy_with_costs(num_arms, num_steps, eps),
                 label=f"ε-Greedy (ε={eps})")
    plt.plot(ucb_with_costs(num_arms, num_steps, c=2.0), label="UCB (c=2.0)", linestyle="--")
    plt.xlabel("Steps"); plt.ylabel("Cumulative Net Reward")
    plt.title("UCB vs ε-Greedy: Cumulative Net Reward")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots/plot4_ucb_vs_egreedy.png", dpi=150); plt.close()

    # Plot 5: dynamic rewards (Q3c)
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_greedy_with_costs(num_arms, num_steps, 0.1), label="Static rewards (ε=0.1)")
    plt.plot(run_dynamic_rewards(num_arms, num_steps, 0.1),
             label="Dynamic rewards (shift every 200 steps)")
    plt.xlabel("Steps"); plt.ylabel("Cumulative Net Reward")
    plt.title("ε-Greedy: Static vs Dynamic Reward Distributions")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots/plot5_dynamic_rewards.png", dpi=150); plt.close()

    # Plot 6: regret (Q4)
    plt.figure(figsize=(10, 6))
    for eps in [0.01, 0.1, 0.5]:
        plt.plot(compute_regret(num_arms, num_steps, eps), label=f"ε = {eps}")
    plt.xlabel("Steps"); plt.ylabel("Cumulative Regret")
    plt.title("ε-Greedy: Cumulative Regret over Time")
    plt.legend(); plt.tight_layout()
    plt.savefig("plots/plot6_regret.png", dpi=150); plt.close()

    print("All assignment plots saved to ./plots/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    num_arms  = 5
    num_steps = 1000
    epsilons  = [0.0, 0.05, 0.1, 0.2, 0.5]

    plot_cumulative_net_rewards_for_epsilons(num_arms, num_steps, epsilons)
    generate_assignment_plots(num_arms, num_steps)
