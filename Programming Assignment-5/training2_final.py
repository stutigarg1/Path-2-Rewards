import numpy as np
import gymnasium as gym
import pygame
# Write just ONE line of code below this comment to import DQN from stable baseline
from stable_baselines3 import DQN

def visualize_model_performance(model):
    env = gym.make('MountainCar-v0', render_mode='human')
    
    x, _ = env.reset()
    total_reward = 0
    terminated, truncated = False, False
    while not(terminated) and not(truncated):
        action, _ = model.predict(x)
        x, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    
    print('Total reward = {}'.format(total_reward))
    env.close()
    # pygame.display.quit() # Use this line when the display screen is not going away
        

class Custom_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Write your own code here to implement the modified reward.
        position, velocity = observation
        # Component 1: Position-based reward (encourage moving towards goal at +0.5)
        position_reward = 100 * (position + 0.5)  # Scale position to make it more positive as car moves right
        
        # Component 2: Speed-based reward (encourage building momentum)
        speed_reward = 10 * abs(velocity)  # Reward based on absolute velocity (speed)
        
        # Component 3: Penalty for not reaching goal (prevent infinite reward accumulation)
        time_penalty = -1  # Small penalty for each step
        
        # Combined modified reward
        modified_reward = position_reward + speed_reward + time_penalty
        
        return observation, modified_reward, terminated, truncated, info


# Initiate the mountain car environment.
env = gym.make('MountainCar-v0')

# Write just one line of code below this comment to create a modified environment using Custom_Wrapper class.
modified_env = Custom_Wrapper(env)

# Write just TWO lines of code below this comment to train a DQN model for mountain car.
model = DQN("MlpPolicy", modified_env, train_freq=4, buffer_size=10000, learning_rate=1e-3, batch_size=32, target_update_interval=500, learning_starts=500, exploration_initial_eps=1.0, exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[32, 32]), verbose=1)
model.learn(total_timesteps=50000)

# Close the mountain car environment.
env.close()

# Write just ONE line of code below to save the DQN model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL2.
# THE MODEL THAT YOU SUBMIT MUST NOT EXCEED 1 MB. ELSE ZERO FOR SECTION 4.
model.save("MODEL2")

# Write just ONE line of code below this comment to call visualize_model_performance in order to test the performance of the trained model
visualize_model_performance(model)