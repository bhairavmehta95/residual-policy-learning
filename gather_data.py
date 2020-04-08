from rpl_environments import rpl_environments
import gym
import numpy as np
import random

timesteps = [0.002, 0.004, 0.01, 0.02]
for timestep in timesteps:
    multipler = timestep / timesteps[0]

    env = gym.make('ResidualMPCPush-v0')
    env.fetch_env._max_episode_steps = 100 // multipler
    env.seed(123)
    np.random.seed(123)
    random.seed(123)

    env.reset()
    env.fetch_env.sim.model.opt.timestep = timestep
    x = 0
    while x != 3:
        s, r, d, _ = env.step(np.zeros(env.action_space.shape))
        env.render()
        if d: 
            env.reset()
            x += 1
    
    env.close()
