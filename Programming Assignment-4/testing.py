import numpy as np
import gymnasium as gym
import pygame
import tensorflow as tf
from tensorflow import keras

def get_q_values_for_all_actions(model, state, action_size):
    """Helper function to get Q-values for all actions given a state"""
    states_batch = np.tile(state.reshape(1, -1), (action_size, 1))
    actions_batch = np.eye(action_size)
    q_values = model.predict([states_batch, actions_batch], verbose=0)
    return q_values.flatten()

def choose_action(state, model):
    # This function should choose action based on the DQN model and the current state.
    # While choosing action here, exploration is not required.
    # You have to set the arguments of the function and write the required code.
    action_size = 3  # MountainCar-v0 has 3 discrete actions
    q_values = get_q_values_for_all_actions(model, state, action_size)
    return np.argmax(q_values)


# The following line load the DQN model. Write (commented) paths for both models, with and without offline data.
#model = keras.models.load_model("DQN_offline_false_fast.keras")  # Without offline data
model = keras.models.load_model("DQN_offline_true_fast.keras")   # With offline data


# The following line initializes the Mountain Car environment with render_mode
# set to 'human'.
env = gym.make("MountainCar-v0", render_mode='human')


# The following line resets the environment,
state, _ = env.reset()


end_episode = False
total_reward = 0
while not(end_episode):
    
    # The following line picks an action using choose_action() function.
    action = choose_action(state, model)
    
    
    # The following line takes that the picked action. After taking the action,
    # it gets next state,reward, terminated, truncated, and info.
    next_state, reward, terminated, truncated, _ = env.step(action)
    
    
    # The following line update the total reward
    total_reward += reward
    
    
    # The following line decides the state for the next time slot.
    state = next_state
    
    
    # The following line decides end_episode for the next time slot.
    end_episode = terminated or truncated


# The following line prints the total reward.
print("Total reward:", total_reward)


# The following line closes the environment.
env.close()

pygame.display.quit()