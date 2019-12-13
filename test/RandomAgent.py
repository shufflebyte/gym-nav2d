import gym
import gym_nav2d
import numpy as np
env = gym.make('nav2d-v0')
env.reset()
for _ in range(10):
    env.render()
    act = env.action_space.sample()
    act = (np.array([315]), np.array([10]))
    obs, rew, done, info = env.step(act)     # take a random action
    print(info)
    # env.step(1)
env.close()
