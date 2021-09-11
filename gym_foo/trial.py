import gym
import numpy as np

seed=100
np.random.seed(seed)

env = gym.make('gym_foo:Gaussian_Bandit-v0')
env.__init__(0.2, set_seed = seed)
print(env.q)
print(env.action_space.sample())

env = gym.make('gym_foo:Bernoulli_Bandit-v0')
env.__init__(0.2,0.3, set_seed = seed)
print(env.alpha, env.beta)

