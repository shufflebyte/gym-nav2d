from gym import error, spaces, utils
import numpy as np
import math

from gym_nav2d.envs.nav2d_env import Nav2dEnv


class Nav2dVeryHardEnv(Nav2dEnv):
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        Nav2dEnv.__init__(self)
        self.high_state = math.sqrt(math.sqrt(pow(self.len_court_x, 2) + pow(self.len_court_y, 2)))
        self.observation_space = spaces.Box(np.array([0]),
                                            np.array([self.high_state]), dtype=np.float32)

    def _observation(self):
        # distance to the goal
        return np.array([self._distance()])

    def reset(self):
        # Changing start point and fixed goal point
        self.count_actions = 0
        self.positions = []
        self.agent_x = self.np_random.uniform(low=0, high=self.len_court_x)
        self.agent_y = self.np_random.uniform(low=0, high=self.len_court_y)
        self.goal_x = self.np_random.uniform(low=0, high=self.len_court_x)
        self.goal_y = self.np_random.uniform(low=0, high=self.len_court_y)
        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale,
                  self.goal_y*self.scale)
        return self._observation()

    def step(self, action):
        self.count_actions += 1
        self._calculate_position(action)

        # calulate new observation
        obs = self._observation()

        # done for rewarding
        done = bool(obs[0] <= self.eps)
        rew = 0
        if not done:
            rew += self._step_reward()
        else:
            rew += self._reward_goal_reached()

        # break if more than max_steps actions taken
        done = bool(obs[0] <= self.eps or self.count_actions >= self.max_steps)

        info = "Debug:" + "actions performed:" + str(self.count_actions) + ", act:" + str(action[0]) + "," + str(
            action[1]) + ", dist:" + str(obs[0]) + ", rew:" + str(
            rew) + ", agent pos: (" + str(self.agent_x) + "," + str(self.agent_y) + ")", "goal pos: (" + str(
            self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])
        return obs, rew, done, info
