import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import math

random.seed(1337)


class Nav2dEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # action space: change direction in degree (discrete), run into this direction (Box)
        self.action_space = spaces.Tuple((spaces.Box(low=0, high=359, shape=(1,)),
                                         spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)))
        # how near should agent come to goal?
        self.eps = 10
        # observation: distance to goal
        self.len_court_x = 255
        self.len_court_y = 255
        self.low_state = 0.
        self.high_state = math.sqrt(self.len_court_y*self.len_court_y + self.len_court_x*self.len_court_x)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=(1,))

        # agent stuff
        self.agent_x = 0    # make this random later
        self.agent_y = 0    # make this random later

        # goal stuff
        self.goal_x = 50    # make this random later
        self.goal_y = 50    # make this random later

        self.reset()

    def _reward(self):
        # 100 - distance
        return 100 - abs(math.sqrt(pow((self.goal_x-self.agent_x), 2) + pow((self.goal_y-self.agent_y), 2)))

    def step(self, action):
        angle = action[0][0]
        angle_rad = angle/360*2*math.pi
        step_size = action[1][0]
        # calculate new agent state
        if 0 < angle <= 90:
            self.agent_x = self.agent_x - math.cos(angle_rad) * step_size
            self.agent_y = self.agent_y + math.sin(angle_rad) * step_size
        elif 90 < angle <= 180:
            self.agent_x = self.agent_x + math.cos(angle_rad) * step_size
            self.agent_y = self.agent_y - math.sin(angle_rad) * step_size
        elif 180 < angle <= 270:
            self.agent_x = self.agent_x - math.cos(angle_rad) * step_size
            self.agent_y = self.agent_y + math.sin(angle_rad) * step_size
        elif 270 < angle <= 360:
            self.agent_x = self.agent_x + math.cos(angle_rad) * step_size
            self.agent_y = self.agent_y - math.sin(angle_rad) * step_size

        # borders
        if self.agent_x < 0:
            self.agent_x = 0
        if self.agent_x > self.len_court_x:
            self.agent_x = self.len_court_x
        if self.agent_y < 0:
            self.agent_y = 0
        if self.agent_y > self.len_court_y:
            self.agent_y = self.len_court_y

        # calulate new observation
        obs = abs(math.sqrt(pow((self.goal_x-self.agent_x), 2) +
                            pow((self.goal_y-self.agent_y), 2)))

        rew = self._reward()
        done = bool(obs <= self.eps)
        info = "Debug:" + "act:" + str(action[0][0])+"," + str(action[1][0]) + ", obs:" + str(obs) + ", rew:" + str(rew) + ", agent pos: (" + str(self.agent_x) + "," + str(self.agent_y) + ")", "goal pos: (" + str(self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)
        return obs, rew, done, info

    def reset(self):
        # set initial state randomly
        print("reset")

    def render(self, mode='human'):
        # print("render human")
        pass

    def close(self):
        print("close")
