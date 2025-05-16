import numpy as np
from Assignment2Tools import prob_vector_generator, markov_matrix_generator

def value_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    num_states = len(Swind)  # Number of wind speed states

    # Step 1: Initialize value function and policy
    V = np.zeros(num_states)  # Initialize V(s) = 0 for all states
    policy = np.zeros(num_states, dtype=int)  # Initialize policy (best action for each state)

    # Step 2: Iteratively update V(s) until convergence
    for _ in range(Kmin):
        V_new = np.copy(V)  # Create a copy of V to store updates

        # Loop over all states
        for s in range(num_states):
            best_value = -np.inf  # Start with a very low value
            best_action = None  # Store the best action

            # Loop over all possible actions (transmission values)
            for a in range(num_states):
                expected_value = 0

                # Loop over all possible next states
                for s_next in range(num_states):
                    R = - (Swind[s] - Swind[s_next])**2  # Reward function
                    P_sa = P[s, s_next]  # Transition probability

                    expected_value += P_sa * (R + beta * V[s_next])

                # Choose the best action
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = a  # Store the best action

            # Update value function and policy
            V_new[s] = best_value
            policy[s] = best_action

        # Check for convergence
        if np.max(np.abs(V_new - V)) < theta:
            break  # Stop iteration if values have converged

        V = V_new  # Update value function

    return V, policy  # Return optimal value function and policy


# System parameters (set to default values)
Swind = np.linspace(0, 1, 21)                      # The set of all possible normalized wind speed.
mu_wind = 0.3                                      # Mean wind speed. You can vary this between 0.2 to 0.8.
z_wind = 0.5                                       # Z-factor of the wind speed. You can vary this between 0.25 to 0.75.
                                                   # Z-factor = Standard deviation divided by mean.
                                                   # Higher the Z-factor, the more is the fluctuation in wind speed.
stddev_wind = z_wind*np.sqrt(mu_wind*(1-mu_wind))  # Standard deviation of the wind speed.
retention_prob = 0.9                               # Retention probability is the probability that the wind speed in the current and the next time slot is the same.
                                                   # You can vary the retention probability between 0.05 to 0.95.
                                                   # Higher retention probability implies lower fluctuation in wind speed.
P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)  # Markovian probability matrix governing wind speed.

lmbda = 0.7  # Probability of successful transmission.

B = 10         # Maximum battery capacity.
eta = 2        # Battery power required for one transmission.
Delta = 3      # Maximum solar power in one time slot.
mu_delta = 2   # Mean of the solar power in one time slot.
z_delta = 0.5  # Z-factor of the slower power in one time slot. You can vary this between 0.25 to 0.75.
stddev_delta = z_delta*np.sqrt(Delta*(Delta-mu_delta))  # Standard deviation of the solar power in one time slot.
alpha = prob_vector_generator(np.arange(Delta+1), mu_delta, stddev_delta)  # Probability distribution of solar power in one time slot.

tau = 4       # Number of time slots in active phase.
gamma = 1/15  # Probability of getting chance to transmit. It can vary between 0.01 to 0.99.

beta = 0.95   # Discount factor.
theta = 0.01  # Convergence criteria: Maximum allowable change in value function to allow convergence.
Kmin = 10     # Convergence criteria: Minimum number of iterations to allow convergence.


# Call value iteration function and print results
V_optimal, policy_optimal = value_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)

print("Optimal Value Function:")
print(V_optimal)

print("\nOptimal Policy:")
print(policy_optimal)