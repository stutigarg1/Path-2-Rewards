from google.colab import files

import numpy as np
import random
import gym
from gym import spaces

# Define the custom Traffic Intersection environment
class TrafficIntersectionEnv(gym.Env):
    def __init__(self):
        super(TrafficIntersectionEnv, self).__init__()
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete([21, 21])
        
        self.arrival_probs = [0.28, 0.4]
        self.departure_prob = 0.9
        self.departure_decay = 0.1
        
    def reset(self):
        self.queue1 = 0
        self.queue2 = 0
        self.time = 0
        return np.array([self.queue1, self.queue2])
    
    def step(self, action):
        if random.random() < self.arrival_probs[0]:
            self.queue1 += 1
        if random.random() < self.arrival_probs[1]:
            self.queue2 += 1
        
        if action == 0:
            departure_prob1 = self.departure_prob
            departure_prob2 = max(0, self.departure_prob - self.time * self.departure_decay)
        else:
            departure_prob1 = max(0, self.departure_prob - self.time * self.departure_decay)
            departure_prob2 = self.departure_prob
        
        if random.random() < departure_prob1 and self.queue1 > 0:
            self.queue1 -= 1
        if random.random() < departure_prob2 and self.queue2 > 0:
            self.queue2 -= 1
        
        self.time += 1
        
        reward = -(self.queue1 + self.queue2)
        done = self.time >= 2000
        
        return np.array([self.queue1, self.queue2]), reward, done, {}
    
    def render(self):
        print(f"Time: {self.time}, Queue1: {self.queue1}, Queue2: {self.queue2}")

# Define the SARSA algorithm for training
def SARSA(env, epsilon=0.1, alpha=0.1, beta=0.997, num_episodes=2000):
    Q = np.zeros((21, 21, 2))
    policy = np.zeros((21, 21), dtype=int)
    
    for episode in range(num_episodes):
        state = env.reset()
        queue1, queue2 = state
        action = np.argmax(Q[queue1, queue2]) if random.random() > epsilon else env.action_space.sample()
        
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_queue1, next_queue2 = next_state
            next_queue1 = np.clip(next_queue1, 0, 20)
            next_queue2 = np.clip(next_queue2, 0, 20)
            
            next_action = np.argmax(Q[next_queue1, next_queue2]) if random.random() > epsilon else env.action_space.sample()
            
            Q[queue1, queue2, action] += alpha * (reward + beta * Q[next_queue1, next_queue2, next_action] - Q[queue1, queue2, action])
            
            queue1, queue2 = next_queue1, next_queue2
            action = next_action
        
        for i in range(21):
            for j in range(21):
                policy[i, j] = np.argmax(Q[i, j])
    
    np.save('policy1.npy', policy)
    return policy

# Define the Expected SARSA algorithm
def ExpectedSARSA(env, epsilon=0.1, alpha=0.1, beta=0.997, num_episodes=2000):
    Q = np.zeros((21, 21, 2))
    policy = np.zeros((21, 21), dtype=int)
    
    for episode in range(num_episodes):
        state = env.reset()
        queue1, queue2 = state
        action = np.argmax(Q[queue1, queue2]) if random.random() > epsilon else env.action_space.sample()
        
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_queue1, next_queue2 = next_state
            next_queue1 = np.clip(next_queue1, 0, 20)
            next_queue2 = np.clip(next_queue2, 0, 20)
            
            # Expected SARSA uses expectation over next actions
            expected_Q = np.mean(Q[next_queue1, next_queue2])
            
            Q[queue1, queue2, action] += alpha * (reward + beta * expected_Q - Q[queue1, queue2, action])
            
            queue1, queue2 = next_queue1, next_queue2
            action = np.argmax(Q[queue1, queue2]) if random.random() > epsilon else env.action_space.sample()
        
        for i in range(21):
            for j in range(21):
                policy[i, j] = np.argmax(Q[i, j])
    
    np.save('policy2.npy', policy)
    return policy

# Define the Value Function SARSA algorithm
def ValueFunctionSARSA(env, epsilon=0.1, alpha=0.1, beta=0.997, num_episodes=2000):
    V = np.zeros((21, 21))
    policy = np.zeros((21, 21), dtype=int)
    
    for episode in range(num_episodes):
        state = env.reset()
        queue1, queue2 = state
        done = False
        while not done:
            action = 0 if random.random() > epsilon else 1
            
            next_state, reward, done, _ = env.step(action)
            next_queue1, next_queue2 = next_state
            next_queue1 = np.clip(next_queue1, 0, 20)
            next_queue2 = np.clip(next_queue2, 0, 20)
            
            V[queue1, queue2] += alpha * (reward + beta * V[next_queue1, next_queue2] - V[queue1, queue2])
            
            queue1, queue2 = next_queue1, next_queue2
        
        for i in range(21):
            for j in range(21):
                policy[i, j] = 0  # default policy when actions are equivalent
    
    np.save('policy3.npy', policy)
    return policy

# Training and saving policies
env = TrafficIntersectionEnv()

SARSA_policy = SARSA(env)
ExpectedSARSA_policy = ExpectedSARSA(env)
ValueFunctionSARSA_policy = ValueFunctionSARSA(env)

# Download the policies
files.download('policy1.npy')
files.download('policy2.npy')
files.download('policy3.npy')

policy1 = np.load('policy1.npy')
policy2 = np.load('policy2.npy')
policy3 = np.load('policy3.npy')

import numpy as np

# Upload the file
from google.colab import files
uploaded = files.upload()

# Load the file
policy = np.load('policy1.npy')
print(policy)