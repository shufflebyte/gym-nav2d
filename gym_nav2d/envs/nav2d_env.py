import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Nav2dEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print("init")

    def step(self, action):
        print("step")

    def reset(self):
        print("reset")

    def render(self, mode='human'):
        # print("render human")
        pass

    def close(self):
        print("close")
