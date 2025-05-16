import numpy as np
import random
import matplotlib.pyplot as plt

# Define the custom Traffic Intersection environment
class TrafficIntersectionEnv:
    def __init__(self):
        # Define the action space: 2 actions, 0 for red on road 1, 1 for green on road 1 (same for road 2)
        self.action_space = 2
        
        # Define the observation space (Queue lengths for both roads), bounded by [0, 20]
        self.observation_space = (21, 21)  # Two roads, each with a queue length of 0-20
        
        # Define transition probabilities
        self.arrival_probs = [0.28, 0.4]  # Arrival probabilities for road 1 and road 2
        self.departure_prob = 0.9         # Departure probability when green light is on
        self.departure_decay = 0.1        # Departure probability decay when light turns red
        
    def reset(self):
        self.queue1 = 0  # Initial queue length for road 1
        self.queue2 = 0  # Initial queue length for road 2
        self.time = 0    # Initialize time step
        return np.array([self.queue1, self.queue2])  # Initial state
    
    def step(self, action):
        # Action 0: Green light for road 1, Red for road 2
        # Action 1: Green light for road 2, Red for road 1
        
        # Handle arrivals for both roads
        if random.random() < self.arrival_probs[0]:
            self.queue1 += 1
        if random.random() < self.arrival_probs[1]:
            self.queue2 += 1
        
        # Handle departures based on the action (Green light allows departure)
        if action == 0:  # Green for road 1
            departure_prob1 = self.departure_prob
            departure_prob2 = max(0, self.departure_prob - self.time * self.departure_decay)
        else:  # Green for road 2
            departure_prob1 = max(0, self.departure_prob - self.time * self.departure_decay)
            departure_prob2 = self.departure_prob
        
        # Depart vehicles with the respective probabilities
        if random.random() < departure_prob1 and self.queue1 > 0:
            self.queue1 -= 1
        if random.random() < departure_prob2 and self.queue2 > 0:
            self.queue2 -= 1
        
        self.time += 1  # Update time
        
        # Define a reward function, which is the negative queue length (minimizing queue)
        reward = -(self.queue1 + self.queue2)
        
        # End the episode if the time exceeds a certain number of time slots (limit steps)
        done = self.time >= 2000  # End after 2000 steps (episodes)
        
        return np.array([self.queue1, self.queue2]), reward, done, {}
    
    def render(self):
        print(f"Time: {self.time}, Queue1: {self.queue1}, Queue2: {self.queue2}")

# Define the SARSA algorithm for training the model
def SARSA(env, epsilon=0.1, alpha=0.1, beta=0.997, num_episodes=2000):
    Q = np.zeros((21, 21, 2))  # Initialize Q-values for each state-action pair (queue lengths and actions)
    
    for episode in range(num_episodes):
        state = env.reset()
        queue1, queue2 = state
        action = np.argmax(Q[queue1, queue2]) if random.random() > epsilon else random.randint(0, 1)
        
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_queue1, next_queue2 = next_state
            
            # Ensure next_queue1 and next_queue2 are within bounds (0 to 20)
            next_queue1 = np.clip(next_queue1, 0, 20)
            next_queue2 = np.clip(next_queue2, 0, 20)
            
            next_action = np.argmax(Q[next_queue1, next_queue2]) if random.random() > epsilon else random.randint(0, 1)
            
            # Update Q-value for SARSA
            Q[queue1, queue2, action] += alpha * (reward + beta * Q[next_queue1, next_queue2, next_action] - Q[queue1, queue2, action])
            
            # Set next state and action
            queue1, queue2 = next_queue1, next_queue2
            action = next_action
    
    return Q  # Return the learned Q-table

# Define the TestPolicy function for testing the trained Q-values
def TestPolicy(env, Q):
    # Reset the environment
    state = env.reset()
    queue1, queue2 = state
    time_slots = []  # To store the time steps
    queue_lengths = []  # To store the queue lengths at each time step
    actions_taken = []  # To store the actions taken at each time step
    
    done = False
    while not done:
        # Choose action based on the learned Q-values
        action = np.argmax(Q[queue1, queue2])
        
        # Perform the action in the environment
        next_state, reward, done, _ = env.step(action)
        next_queue1, next_queue2 = next_state
        
        # Track the state, queue lengths, and actions
        time_slots.append(env.time)
        queue_lengths.append([queue1, queue2])
        actions_taken.append(action)
        
        # Update state
        queue1, queue2 = next_queue1, next_queue2
    
    # Convert queue_lengths to numpy array for easier handling
    queue_lengths = np.array(queue_lengths)
    
    # Plot the queue lengths over time slots
    plt.plot(time_slots, queue_lengths[:, 0], label="Queue 1")
    plt.plot(time_slots, queue_lengths[:, 1], label="Queue 2")
    plt.xlabel("Time Slots")
    plt.ylabel("Queue Length")
    plt.title("Queue Lengths of Both Roads Over Time")
    plt.legend()
    plt.show()
    
    # Calculate the average sum of queue lengths over the episode
    avg_queue_sum = np.mean(queue_lengths.sum(axis=1))
    
    # Display the results
    print(f"Average Sum of Queue Lengths: {avg_queue_sum}")
    print(f"Actions Taken: {actions_taken}")

# Create the environment instance
env = TrafficIntersectionEnv()

# Train the SARSA model to get the Q-values
Q = SARSA(env)

# Test the trained Q-values by running the agent in the environment
TestPolicy(env, Q)
