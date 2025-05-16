import numpy as np
from itertools import product
from Assignment2Tools import prob_vector_generator, markov_matrix_generator

def action_space_function(battery_level, eta):
    if battery_level < eta:
        return [0]  # Only action is to not transmit
    else:
        return [0, 1]  # Can choose to transmit or not

def compute_q_value(b, s_idx, z, a, V, P, alpha, lmbda, eta, B_max, beta):
    q_val = 0
    for sp_idx in range(len(Swind)):
        for delta in range(len(alpha)):
            b_prime = b - (eta if a == 1 else 0) + delta
            b_prime = max(0, min(B_max, b_prime))

            z_prime = 1 if a == 1 else 0

            prob_wind = P[s_idx, sp_idx]
            prob_solar = alpha[delta]
            reward = 0

            if a == 1:
                reward = lmbda * (1 - z)

            q_val += prob_wind * prob_solar * (reward + beta * V[b_prime, sp_idx, z_prime])

    return q_val

def policy_iteration(Swind, P, lmbda, B_max, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    num_b = B_max + 1
    num_s = len(Swind)
    num_z = 2

    V = np.zeros((num_b, num_s, num_z))
    policy = np.zeros((num_b, num_s, num_z), dtype=int)

    is_policy_stable = False
    iteration = 0

    while not is_policy_stable:
        # Policy Evaluation
        eval_iter = 0
        while True:
            delta = 0
            V_new = np.copy(V)
            for b, s_idx, z in product(range(num_b), range(num_s), range(num_z)):
                a = policy[b, s_idx, z]
                q_val = compute_q_value(b, s_idx, z, a, V, P, alpha, lmbda, eta, B_max, beta)
                delta = max(delta, np.abs(q_val - V[b, s_idx, z]))
                V_new[b, s_idx, z] = q_val
            V = V_new
            eval_iter += 1
            if delta < theta and eval_iter >= Kmin:
                break

        # Policy Improvement
        is_policy_stable = True
        for b, s_idx, z in product(range(num_b), range(num_s), range(num_z)):
            old_action = policy[b, s_idx, z]
            action_space = action_space_function(b, eta)
            q_vals = [compute_q_value(b, s_idx, z, a, V, P, alpha, lmbda, eta, B_max, beta) for a in action_space]
            best_action = action_space[np.argmax(q_vals)]
            policy[b, s_idx, z] = best_action
            if best_action != old_action:
                is_policy_stable = False
        iteration += 1
        print(f"Iteration {iteration} completed. Policy stable: {is_policy_stable}")

    return V, policy

# System parameters
Swind = np.linspace(0, 1, 21)
mu_wind = 0.3
z_wind = 0.5
stddev_wind = z_wind * np.sqrt(mu_wind * (1 - mu_wind))
retention_prob = 0.9
P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)

lmbda = 0.7
B_max = 10
eta = 2
Delta = 3
mu_delta = 2
z_delta = 0.5
stddev_delta = z_delta * np.sqrt(Delta * (Delta - mu_delta))
alpha = prob_vector_generator(np.arange(Delta + 1), mu_delta, stddev_delta)

tau = 4
gamma = 1 / 15
beta = 0.95
theta = 0.01
Kmin = 10

# Run policy iteration
V_optimal, policy_optimal = policy_iteration(Swind, P, lmbda, B_max, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)

print("Optimal Value Function:\n", V_optimal)
print("\nOptimal Policy:\n", policy_optimal)