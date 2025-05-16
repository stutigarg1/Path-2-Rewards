import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GymTrafficEnv(gym.Env):
    def __init__(self):

      self.max_queue= 17

      self.q1_arr_prob= 0.28
      self.q2_arr_prob= 0.4

      self.green_dep_prob= 0.9

      self.min_delta= 0
      self.max_delta= 10
      self.len_of_episode= 1800

      #max state space since (q1,q2,green signal,delta)
      self.obs_space= spaces.Multidiscrete([18, 18, 2, 11])

      #action space has 2 choices. 0=keep green 1= switch green to other road
      self.observation_space= spaces.Discrete(2) #since 2 actions
      self.reset() 

    def step(self, action):
      
      #action chosen
      if action==1 and self.delta>=10: #a=1 used for switching roads
        if self.green==0: #currently on road1
          self.green==1 #we need to switch to road2

        else: #self.green==1
          self.green=0 #we need to switch to road 1

      else: #our action is a=0 currently green
        self.delta=self.delta+1 #no switching of roads and we keep incr our timer.

      #arrival 
      if np.random.rand()<0.28: #vehicles will arrive randomly road 1
        self.q1=min(self.q1+1,17) #for road 1, it will incr for each arrival and max q1 can be 17 
         
      if np.random.rand()<0.4: #arrival prob for road 2

        self.q2=min(self.q2+1,17) #for road 2 also it will be same


      #departure
      if self.green==0: #road1 is green
        if self.q1>0:
          self.q1=self.q1-1 #car will leave

        if self.q2>0 and self.delta<=10: #implies road 2 is red. 

          p_dep=self.green_dep_prob*(1-(self.delta**2)/100) #decay formula

          if np.random.rand()<p_dep: #if our car leaves
            self.q2=self.q2-1 #we remove 1 from queue
        
      else: #road2 is green 
        if self.q2>0: 
          self.q2=self.q2-1 #car will leave
        
        if self.q1>0 and self.delta<=10: #now road 1 is red.

          p_dep=self.green_dep_prob*(1-(self.delta**2)/100) #decay formula

          if np.random.rand()<p_dep: #car will leave
            self.q1=self.q1-1 #same, we decr by 1.

    
      reward= -(self.q1+self.q2) #more penalty on larger queues

      self.time=self.time+1 #every time slot for 1 sec, till self.time<=1800 time-slots
      terminated=False #given already
      if self.time>=1800: #need to end episode after 1800 time-slots
        truncated=True
      else:
        truncated =False
      info={} #empty dictionary given


      return self.state(), reward, terminated, truncated, info

    def reset(self): 

      self.q1= np.random.randint(0,11) # while starting a new episode, we have any number of cars btw 0-10 on road 1
      self.q2= np.random.randint(0,11) #for road 2

      self.green= np.random.choice(0,1) # if 0 implies road 1 is green, else 1 implies road 2 is green

      self.delta= 0 #everthing gets refreshed so, time since last switched is also 0
      self.state() #we need to return the starting state 
        

    def render(self):
        # Used to graphics. NOT NEEDED FOR THIS ASSIGNMENT.
        pass
