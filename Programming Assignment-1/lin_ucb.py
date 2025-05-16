import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from CloudComputing import ServerAllocationEnv

# Initialize environment
env = ServerAllocationEnv()

def LinUCBPolicy(env, alpha=1.0):
    """
    Implements the UCB Algorithm for Linear Bandits in ServerAllocationEnv.
    
    :param env: The custom ServerAllocationEnv environment.
    :param alpha: The confidence parameter for exploration-exploitation balance.
    :return: Reward history and trained policy parameters (theta).
    """

    obsv, _ = env.reset()
    Nfeatures = len(np.array(obsv).flatten())  # Extract number of features
    Nactions = 8  # Number of possible actions
    theta = np.zeros((Nactions, Nfeatures + 1))  # +1 for bias term
    A = [np.eye(Nfeatures + 1) for _ in range(Nactions)]  # Design matrices
    b = [np.zeros(Nfeatures + 1) for _ in range(Nactions)]  # Reward vectors

    # Training
    Nepisodes = env.Horizon
    reward_arr = []

    for n in range(Nepisodes):
        obsv, _ = env.reset()
        obsv = np.array(obsv).flatten()
        z = np.concatenate((obsv, [1]))  # Include bias term
        truncated = False

        while not truncated:
            ucb_values = np.zeros(Nactions)

            for action in range(Nactions):
                A_inv = np.linalg.pinv(A[action])  # Use pseudo-inverse for stability
                theta[action] = np.dot(A_inv, b[action])  # Estimate reward parameters
                uncertainty = alpha * np.sqrt(z.T @ A_inv @ z)  # Compute UCB
                ucb_values[action] = np.dot(theta[action], z) + uncertainty

            action = np.argmax(ucb_values)  # Select best action
            
            # Take action in environment
            next_obsv, reward, _, truncated, _ = env.step(action)
            reward_arr.append(reward)

            # Update policy parameters
            if not truncated:
                z_temp = z.reshape((-1, 1))  # Convert to column vector
                A[action] += np.dot(z_temp, z_temp.T)  # Update A-matrix
                b[action] += reward * z  # Update reward vector
                
                obsv = np.array(next_obsv).flatten()
                z = np.concatenate((obsv, [1]))  # Feature vector for next step

    return np.array(reward_arr), theta  # Return reward history and final policy


# Run LinUCB Algorithm
reward_arr, theta = LinUCBPolicy(env)

# Plot Reward History
plt.figure(figsize=(10, 5))
plt.plot(np.convolve(reward_arr, np.ones(500)/500, mode='valid'), label="Receding Window Avg Reward")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Receding Window Time-Averaged Reward (UCB Algorithm)")
plt.legend()
plt.show()