import numpy as np
import gymnasium as gym
from CloudComputing import ServerAllocationEnv

env = ServerAllocationEnv()

def LinGreedyPolicy(env):
    
    
    Nfeatures = 7
    Nactions = 8
    theta = np.zeros((Nactions, Nfeatures+1))          #+1 for the bias term
    A = np.zeros((Nactions, Nfeatures+1, Nfeatures+1))
    b = np.zeros((Nactions, Nfeatures+1))
    epsilon = 0.1   #constant exploration rate
                  #for better performance epsilon should be time varying (high --> low)

    #Start training the lin greedy policy
    Nepisodes = env.Horizon
    reward_arr = []
    for n in range(Nepisodes):
        t=0
        obsv, _ = env.reset()
        obsv = np.array(obsv)
        z = np.concatenate((np.array(obsv).flatten(), np.array([1]))).reshape(-1,1)

        truncated = False
        while not(truncated):
            #Decide the action to take
            v = np.random.uniform()
            if v<=epsilon:
                action = np.random.randint(Nactions)   #Exploration
            else:
                action = np.argmax(np.matmul(theta,z))  #Exploitation
                #Taking the action
                next_obsv, reward, _, truncated, _ = env.step(action)
                reward_arr.append(reward)

      #update the policy parameters
            if not(truncated):
                z_temp = z.reshape((-1,1))   #matrix
                A[action] += np.matmul(z_temp, np.transpose(z_temp))
                b[action] += reward*z_temp.reshape(-1)    #reshape to column vector
                theta[action] = np.matmul(np.linalg.inv(A[action] + 0.01*np.eye(Nfeatures+1)), b[action])

                obsv  = next_obsv
                z = np.concatenate((obsv, np.array([1])))   #feature for next time instant
                t += 1

        return np.array(reward_arr), theta   #final policy


    env = ServerAllocationEnv()
    reward_arr, theta = LinGreedyPolicy(env)

    
    
observation_space = env.observation_space
print(observation_space)

LinGreedyPolicy(env)

env.close()