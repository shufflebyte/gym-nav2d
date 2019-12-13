import gym
import gym_nav2d
import numpy as np

DEBUG = False

for e in range(100):
    env = gym.make('nav2d-v0')
    obs = env.reset()
    for i in range(100):
        act = env.action_space.sample()
        # act = (np.array([315]), np.array([10]))   # this is, where we need the obs to feed it to our agent
        obs, rew, done, info = env.step(act)     # take a random action
        env.render(mode='human')
        # print(env.render(mode='ansi'))
        # print(env.render(mode='rgb_array')) # NotImplementedError ;-)
        if DEBUG:
            print(info)
        if done:
            i=100
    env.close()
