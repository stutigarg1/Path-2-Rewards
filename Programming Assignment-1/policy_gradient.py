import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from CloudComputing import ServerAllocationEnv

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, 1)
        self.log_std_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        return mean, std

# Function to sample an action from the policy
def sample_action(policy_net, state):
    state = torch.FloatTensor(state)
    mean, std = policy_net(state)
    action = torch.normal(mean, std)
    return action.item()

# Incremental normalization
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_stats(self):
        variance = self.M2 / self.n if self.n > 1 else 1
        return self.mean, np.sqrt(variance)

# Training loop
def train_agent(env, policy_net, optimizer, num_episodes=5000, window_size=500):
    reward_window = deque(maxlen=window_size)
    running_stats = RunningStats()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.concatenate([np.array(s) for s in state])
        done = False
        log_probs = []
        rewards = []
        
        while not done:
            # Normalize state
            state_mean, state_std = running_stats.get_stats()
            state = (state - state_mean) / (state_std + 1e-8)
            
            action = sample_action(policy_net, state)
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            
            rewards.append(reward)
            reward_window.append(reward)
            
            # Update running stats
            next_state = np.concatenate([np.array(s) for s in next_state])
            running_stats.update(next_state)
            
            state = next_state
        
        # Compute returns and update policy
        returns = np.array(rewards)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        optimizer.zero_grad()
        policy_loss = -torch.stack(log_probs) * torch.tensor(returns, dtype=torch.float32)
        policy_loss.sum().backward()
        optimizer.step()
        
        # Logging
        episode_rewards.append(np.mean(reward_window))
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(reward_window)}")
    
    # Plot receding window time-averaged reward
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Receding Window Avg Reward')
    plt.title('Policy Gradient with Continuous Action Space')
    plt.show()

# Main
if __name__ == "__main__":
    env = ServerAllocationEnv()
    input_dim = env.MaxJobs * 4
    hidden_dim = 64

    policy_net = PolicyNetwork(input_dim, hidden_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    
    train_agent(env, policy_net, optimizer)