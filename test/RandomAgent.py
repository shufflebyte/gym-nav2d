import gym
import gym_nav2d
import numpy as np
import time

DEBUG = False

for e in range(100):
    env = gym.make('nav2d-v0')
    obs = env.reset()
    cumulated_reward = 0
    i = 0
    done = False
    while not done and i < 10:
        act = env.action_space.sample()
        # act = (np.array([315]), np.array([10]))   # this is, where we need the obs to feed it to our agent
        obs, rew, done, info = env.step(act)     # take a random action
        env.render(mode='human')
        # print(env.render(mode='ansi'))
        # print(env.render(mode='rgb_array')) # NotImplementedError ;-)
        cumulated_reward += rew
        if DEBUG:
            print(info)
    if DEBUG and done:
        time.sleep(3)
    print("episode ended with cumulated rew", cumulated_reward, "and done:", done)
    env.close()
