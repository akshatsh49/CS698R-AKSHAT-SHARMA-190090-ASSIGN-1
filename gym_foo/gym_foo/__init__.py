from gym.envs.registration import register

register(
    id='Bernoulli_Bandit-v0',
    entry_point='gym_foo.envs:Bernoulli_Bandit'
)
register(
    id='Gaussian_Bandit-v0',
    entry_point='gym_foo.envs:Gaussian_Bandit'
)
register(
    id='Random_Walk-v0',
    entry_point='gym_foo.envs:Random_Walk'
)