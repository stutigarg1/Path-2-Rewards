import numpy as np
import matplotlib.pyplot as plt
from Assignment2Tools import prob_vector_generator, markov_matrix_generator

# === Optimized Policy Evaluation Function ===
def policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    np.random.seed(0)
    N = len(Swind)
    V = np.random.uniform(-10, 10, N)  # Random initialization
    delta = float('inf')
    iterations = 0
    max_iterations = 100  # Prevents infinite loop

    while (delta > theta or iterations < Kmin) and iterations < max_iterations:
        V_new = np.zeros(N)
        delta = 0

        for s in range(N):
            action_rewards = []

            for action in range(2):  # 0 = don't transmit, 1 = transmit
                if (iterations % tau != 0) and action == 1:
                    continue
                if action == 1 and B < eta:
                    continue

                reward = (action * lmbda) - (action * eta)
                expected_value = np.dot(P[s], V)

                if iterations % 20 == 0:
                    print(f"Iter {iterations}, State {s}, Action {action}, Reward {reward:.2f}, EV {expected_value:.2f}")

                action_rewards.append(reward + beta * expected_value)

            V_new[s] = min(action_rewards) if action_rewards else V[s]
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        iterations += 1

    print(f"\nPolicy evaluation completed in {iterations} iterations.")
    return V

# === System Parameters ===
Swind = np.linspace(0, 1, 15)  # Wind state space
mu_wind = 0.3
z_wind = 0.5
stddev_wind = z_wind * np.sqrt(mu_wind * (1 - mu_wind))
retention_prob = 0.9
P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)

lmbda = 0.7      # Data arrival reward
B = 10           # Battery level
eta = 2          # Energy cost to transmit
Delta = 3        # Max delay
mu_delta = 2     # Mean delay
z_delta = 0.5
stddev_delta = z_delta * np.sqrt(Delta * (Delta - mu_delta))
alpha = prob_vector_generator(np.arange(Delta + 1), mu_delta, stddev_delta)

tau = 4          # Scheduling interval
gamma = 1 / 15   # Discount rate (not directly used here)
beta = 0.8       # Discount factor
theta = 0.005    # Convergence threshold
Kmin = 20        # Minimum number of iterations

# === Run Policy Evaluation ===
V = policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)

# === Print Final Value Function ===
print("\nFinal Value Function:\n", V)

# === Plot the Value Function ===
plt.figure(figsize=(10, 6))
plt.plot(Swind, V, marker='o', color='blue', linewidth=2, linestyle='-')
plt.title("Value Function vs Wind State (Greedy Policy)")
plt.xlabel("Wind State (Swind)")
plt.ylabel("Value Function V(s)")
plt.grid(True)
plt.tight_layout()
plt.show()
