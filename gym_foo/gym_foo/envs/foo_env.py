import numpy as np
import gym
from gym.spaces import Discrete
from gym.utils import seeding

#make environment in openAI gym
class Random_Walk(gym.Env):
  def __init__(self, set_seed=10):
    self.state_space = Discrete(7)
    self.action_space = Discrete(2)
    self.state=3
    self.__seed__ = self.seed(set_seed)

  def step(self, a):
    #no matter what action you take you move left/right with equal probability
    if(np.random.uniform()<0.5):
      self.state = max(0, self.state-1) #move left
    else :
      self.state = min(6, self.state+1) #move right
    
    if(self.state==0 or self.state==6):
      done=True
    else:
      done=False

    if(self.state==6):
      reward=1
    else :
      reward=0

    return self.state, reward, done

  def render(self):
    return
  
  def seed(self, __seed__):
    self.np_random, __seed__ = seeding.np_random(__seed__)
    self.action_space.seed(__seed__)
    return [__seed__]

  def reset(self):
    self.state=3
    return self.state

#make environment in openAI gym
class Bernoulli_Bandit(gym.Env):
  def __init__(self, alpha=0.1, beta=0.1, set_seed = 10):
    self.name = "Bernoulli"
    self.alpha = alpha
    self.beta = beta
    self.action_space = Discrete(2) #0 indicates left action
    self.state_space = Discrete(3)
    self.state=0
    self.q = [self.alpha, self.beta]
    self.__seed__ = self.seed(set_seed)
    
  def step(self, action):
    if(action==0):
      if(np.random.uniform() < self.alpha):
        self.state = 1
        reward = 1  
      else:
        self.state = 2
        reward = 0
      
    elif(action==1):
      if(np.random.uniform() < self.beta):
        self.state = 2
        reward = 1
      else:
        self.state = 1
        reward = 0

    return self.state, reward, True

  def render(self):
    return

  def seed(self,__seed__):
    self.np_random, __seed__ = seeding.np_random(__seed__)
    self.action_space.seed(__seed__)
    return [__seed__]
  
  def reset(self):
    self.state = 0
    return self.state

#make environment in openAI gym
class Gaussian_Bandit(gym.Env):
  def __init__(self, sigma = 1, set_seed = 10):
    self.name = "Gaussian"
    self.sigma = sigma
    self.action_space = Discrete(10) #action (a) takes you to state (a+1)
    self.state_space = Discrete(11)  #(0) is the start state
    self.state=0
    self.__seed__ = self.seed(set_seed)
    self.q = np.random.normal(0,self.sigma,size=(10,))
    
  def step(self, action):
    self.state = action+1
    reward = np.random.normal(self.q[action],self.sigma)
    # print("Gaussian bandit reward {}".format(reward))
    return self.state, reward, True

  def render(self):
    return
  
  def seed(self,__seed__):
    self.np_random, __seed__ = seeding.np_random(__seed__)
    self.action_space.seed(__seed__)
    return [__seed__]
  
  def reset(self):
    self.state = 0
    return self.state
