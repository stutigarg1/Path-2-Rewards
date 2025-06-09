import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class ModifiedRobotNavigationEnv(gym.Wrapper):
    def __init__(self, env, H):
        super().__init__(env)
        self.H = H   # Number of time slots in a mini episode.
        
        # The action space of the modified robot navigation environment.
        Delta = H*self.env.delta
        self.action_space = gym.spaces.Box(-Delta*np.ones(2), Delta*np.ones(2), dtype=np.float32)
        
    
    def conventional_policy(self, robot_position, goal_intermediate):
        # For don't move, return -1.
        # For up, return 0.
        # For down, return 1.
        # For left, return 2.
        # For right, return 3.
        
        # FREE ADVICE: To account for decimal error, it is better to give some
        # level to tolerance when comparing two values. Examples:
        # 1) Rather then writting if a>b, it is better to write if a-b>0.00001.
        # 2) Rather then writting if a<b, it is better to write if a-b<-0.00001.
        
        x_robot, y_robot = robot_position
        x_goal, y_goal = goal_intermediate
        
        tolerance = 0.00001
        
        # Check x-direction first
        if x_goal - x_robot > tolerance:
            return 3  # go right
        elif x_robot - x_goal > tolerance:
            return 2  # go left
        # If x positions are approximately equal, check y-direction
        elif y_goal - y_robot > tolerance:
            return 0  # go up
        elif y_robot - y_goal > tolerance:
            return 1  # go down
        else:
            return -1  # don't move


    def step(self, action):
        # Compute the intermediate goal
        goal_intermediate = self.env.robot_position + action        
        grid_x = int(goal_intermediate[0]/self.env.delta) + 1
        grid_y = int(goal_intermediate[1]/self.env.delta) + 1        
        grid_x = self.env.delta*(0.5 + (grid_x - 1))
        grid_y = self.env.delta*(0.5 + (grid_y - 1))
        goal_intermediate = np.array([grid_x, grid_y])
        
        # Simulate a mini-episode using the conventional policy.
        reward_miniepisode = 0
        for h in range(self.H):
            reward = -np.sqrt(np.sum((self.env.robot_position - self.env.goal)**2))
            
            a = self.conventional_policy(self.env.robot_position, goal_intermediate)  # Call conventional policy.
            
            if a!=-1: # If a==-1, then robot position does not change.
                self.env.robot_position = self.env.robot_position + self.env.action_dict[a+0]*self.env.delta
            
            self.env.trail.append(self.env.robot_position)
            
            # Check for collision
            terminated = self.env.check_collision()
            if terminated:
                reward = -10000
            
            # Check if goal is reached
            if not(terminated):
                if (self.env.goal[0] - self.env.robot_position[0])**2 + (self.env.goal[1] - self.env.robot_position[1])**2<=self.env.goal_radius**2:
                    terminated = True
                    reward = 1000
            
            reward_miniepisode+=reward
            
            self.env.t += 1
            truncated = False
            if self.env.t>self.env.Horizon:
                truncated = True
            
            if self.env.render_mode == "human":
                self.env.render()
            
            if terminated or truncated:
                break
        
        self.env.observation = np.concatenate((self.env.get_lidar_reading(),self.env.robot_position))
        
        return self.env.observation, reward_miniepisode, terminated, truncated, {}
        
  

# You can copy-paste the code for the custom callback LoggingAndSavingCallback
# that you wrote for training3.py. All you need to change is the code for
# initiating the environment during testing.
class LoggingAndSavingCallback(BaseCallback):
    def __init__(self, test_period, test_count, verbose=0):
        super().__init__(verbose)
        
        self.test_period = test_period
        self.test_count = test_count
        
        # Variables for tracking training progress
        self.current_episode_reward = 0
        self.training_rewards = []
        self.testing_rewards = []
        self.best_avg_reward = -np.inf
        self.step_count = 0
        
    def _on_step(self) -> bool:
        # Get current reward and done status
        current_reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        # Accumulate reward for current episode
        self.current_episode_reward += current_reward
        self.step_count += 1
        
        # Check if episode is done
        if done:
            # Save episode reward
            self.training_rewards.append(self.current_episode_reward)
            np.save('training_log.npy', np.array(self.training_rewards))
            
            # Reset for next episode
            self.current_episode_reward = 0
        
        # Every test_period steps, test and save model
        if self.step_count % self.test_period == 0:
            # Save latest model
            self.model.save("LATEST_MODEL")
            
            # Test the model - Create modified environment for testing
            base_env = RobotNavigationEnv()
            test_env = ModifiedRobotNavigationEnv(base_env, H=15)  # Use same H as training
            episode_rewards = []
            
            for _ in range(self.test_count):
                obs, _ = test_env.reset()
                episode_reward = 0
                terminated = False
                truncated = False
                
                while not terminated and not truncated:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = test_env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            test_env.close()
            
            # Calculate average reward
            avg_reward = np.mean(episode_rewards)
            self.testing_rewards.append(avg_reward)
            
            # Save testing log
            np.save('testing_log.npy', np.array(self.testing_rewards))
            
            # Save best model if this is the best performance
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.model.save("BEST_MODEL")
                print(f"New best average reward: {avg_reward:.2f}")
            
            print(f"Step {self.step_count}: Average test reward: {avg_reward:.2f}")
            
        return True
            
            
# Initiate the robot navigation environment. 
env = RobotNavigationEnv()
H = 15  # H is the duration of a mini-episode. My advide is to change it between 10 to 40. But you can go crazy with it!
env = ModifiedRobotNavigationEnv(env, H)


# Initiate an instance of the LoggingAndSavingCallback. Desription of test_period
# and test_count are there in _init__ function of LoggingAndSavingCallback.
test_period = 100000   # Default value. You can change it.
test_count = 3       # Default value. You can change it.
callback = LoggingAndSavingCallback(test_period, test_count)


# The code that you use to train the RL agent for the robot navigation environment
# goes below this line. The total number of lines is unlikely to be more than 10.

# Use PPO for faster training than SAC
model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=1,
    device='auto'
)

# Reduced timesteps for faster training - should complete in ~1 hour
model.learn(total_timesteps=800000, callback=callback)


# Close the robot navigation environment.
env.close()


# Write just ONE line of code below to save the model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL4.
model.save("MODEL4")