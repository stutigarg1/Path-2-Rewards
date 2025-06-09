import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class LoggingAndSavingCallback(BaseCallback):
    def __init__(self, test_period, test_count, verbose=0):
        super().__init__(verbose)
        # test_period is the number of time steps (env.step()) after which we
        # want to test the model. You also have to save the latest model every
        # test_period.
        
        # test_count is the number of episodes for which we want to test the model.
        
        self.test_period = test_period
        self.test_count = test_count
        
        # You can declare other variables here that are required to do the
        # tasks mentioned in _on_step() function.
        self.current_episode_reward = 0
        self.training_rewards = []
        self.testing_rewards = []
        self.best_avg_reward = -np.inf
        self.step_count = 0

    def _on_step(self) -> bool:
        
        # This function should do the following:
        #  1. Calculate the sum of reward of every episode and save it in a
        #     .npy file named training_log.npy during training. This MUST be
        #     done in the end of every episode. To do this, you can access the
        #     reward of the current time step using self.locals['rewards'][0]
        #     and you can check if the current episode is terminated/truncated
        #     using self.locals['dones'][0].
        #
        #  2. Every self.test_period time steps:
        #      2.a. Save the latest model as LATEST_MODEL. You can access the
        #           latest model using self.model. 
        #
        #      2.b. Test the latest model by calculating the average sum reward
        #           of the latest model over self.test_count episodes. After
        #           calculating average sum of reward, append this latest test
        #           result to a .npy file named testing_log.npy. Also, if the
        #           average sum of reward is highest till now, save the latest
        #           model as BEST_MODEL.
        #
        #           VERY IMPORTANT: While testing, you need to initiate a LOCAL
        #           robot navigation environment here. You MUST NOT use the
        #           environment that you initiated for training for testing
        #           purposes.
        
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
            
            # Test the model
            test_env = RobotNavigationEnv()
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
            
        return True # This MUST return True unless you want the training to stop.


# Initiate the robot navigation environment.
env = RobotNavigationEnv()

# Use vectorized environment for faster training (optional optimization)
env = make_vec_env(lambda: RobotNavigationEnv(), n_envs=4)

# Initiate an instance of the LoggingAndSavingCallback. Description of test_period
# and test_count are there in _init__ function of LoggingAndSavingCallback.
test_period = 50000   # Optimized for vectorized environment
test_count = 5        # Reduced for faster testing
callback = LoggingAndSavingCallback(test_period, test_count)

# The code that you use to train the RL agent for the robot navigation environment
# goes below this line. The total number of lines is unlikely to be more than 10.

model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=128,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=1,
    device='auto'
)

model.learn(total_timesteps=2000000, callback=callback)

# Close the robot navigation environment.
env.close()

# Write just ONE line of code below to save the model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL3.
model.save("MODEL3")