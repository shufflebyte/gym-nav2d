from gym import error, spaces, utils
import numpy as np
import math

from gym_nav2d.envs.nav2d_env import Nav2dEnv


class Nav2dVeryEasyEnv(Nav2dEnv):
    # this is a list of supported rendering modes!
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        Nav2dEnv.__init__(self)

    def reset(self):
        # Fixed start point and fixed goal point
        self.count_actions = 0
        self.positions = []
        self.agent_x = 10
        self.agent_y = 10
        self.goal_x = 200
        self.goal_y = 200
        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale,
                  self.goal_y*self.scale)
        obs = self._observation()
        return self._normalize_observation(obs)
