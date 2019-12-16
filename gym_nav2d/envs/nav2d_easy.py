from gym import error, spaces, utils
import numpy as np
import math

from gym_nav2d.envs.nav2d_env import Nav2dEnv


class Nav2dEasyEnv(Nav2dEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        Nav2dEnv.__init__(self)
        self.high_state = math.sqrt(math.sqrt(pow(self.len_court_x, 2) + pow(self.len_court_y, 2)))
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]),
                                            np.array([self.len_court_x, self.len_court_y, self.len_court_x,
                                                      self.len_court_y, self.high_state]), dtype=np.float32)

    def reset(self):
        # Changing start point and fixed goal point
        self.count_actions = 0
        self.positions = []
        self.agent_x = self.np_random.uniform(low=0, high=self.len_court_x)
        self.agent_y = self.np_random.uniform(low=0, high=self.len_court_y)
        self.goal_x = 127
        self.goal_y = 127
        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale,
                  self.goal_y*self.scale)
        return self._observation()
