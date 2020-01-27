import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math


class Nav2dEnv(gym.Env):
    # this is a list of supported rendering modes!
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        self.debug = False
        # define the environment and the observations
        self.len_court_x = 255              # the size of the environment
        self.len_court_y = 255              # the size of the environment

        self.obs_low_state = np.array([-1, -1, -1, -1, 0]) # x_agent,y_agent, x_goal, y_goal, distance
        self.obs_high_state = np.array([1, 1, 1, 1, 1])
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        self.max_steps = 100
        self.max_step_size = 10
        # action space: change direction in rad (discrete), run into this direction (Box)
        self.action_angle_low = -1
        self.action_angle_high = 1
        self.action_step_low = -1
        self.action_step_high = 1
        self.action_space = spaces.Box(np.array([self.action_angle_low, self.action_step_low]),
                                       np.array([self.action_angle_high, self.action_step_high]), dtype=np.float32)

        self.count_actions = 0  # count actions for rewarding
        self.eps = 5  # distance to goal, that has to be reached to solve env
        self.np_random = None  # random generator

        # agent
        self.agent_x = 0
        self.agent_y = 0
        self.positions = []                 # track agent positions for drawing

        # the goal
        self.goal_x = 0
        self.goal_y = 0

        # rendering
        self.screen_height = 600
        self.screen_width = 600
        self.viewer = None                  # viewer for render()
        self.agent_trans = None             # Transform-object of the moving agent
        self.track_way = None               # polyline object to draw the tracked way
        self.scale = self.screen_width/self.len_court_x

        # set a seed and reset the environment
        self.seed()
        self.reset()

    def _distance(self):
        return math.sqrt(pow((self.goal_x - self.agent_x), 2) + pow(self.goal_y - self.agent_y, 2))

    # todo: think about a good reward fct that lets the agents learn to go to the goal by
    #  extra rewarding reaching the goal and learning to do this by few steps as possible
    def _reward_goal_reached(self):
        # 1000 - (distance)/10 - (sum of actions)
        return 1000

    def _step_reward(self):
        return - self._distance()/10 - 1

    def _observation(self):
        return np.array([self.agent_x, self.agent_y, self.goal_x, self.goal_y, self._distance()])

    def _normalize_observation(self, obs):
        normalized_obs = []
        for i in range(0, 4):
            normalized_obs.append(obs[i]/255*2-1)
        normalized_obs.append(obs[-1]/360.62)
        return np.array(normalized_obs)

    def _calculate_position(self, action):
        angle = (action[0] + 1) * math.pi + math.pi / 2
        if angle > 2 * math.pi:
            angle -= 2 * math.pi
        step_size = (action[1] + 1) / 2 * self.max_step_size
        # calculate new agent state
        self.agent_x = self.agent_x + math.cos(angle) * step_size
        self.agent_y = self.agent_y + math.sin(angle) * step_size

        # borders
        if self.agent_x < 0:
            self.agent_x = 0
        if self.agent_x > self.len_court_x:
            self.agent_x = self.len_court_x
        if self.agent_y < 0:
            self.agent_y = 0
        if self.agent_y > self.len_court_y:
            self.agent_y = self.len_court_y

    def step(self, action):
        self.count_actions += 1
        self._calculate_position(action)
        # calulate new observation
        obs = self._observation()

        # done for rewarding
        done = bool(obs[4] <= self.eps)
        rew = 0
        if not done:
            rew += self._step_reward()
        else:
            rew += self._reward_goal_reached()

        # break if more than max_steps actions taken
        done = bool(obs[4] <= self.eps or self.count_actions >= self.max_steps)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])

        normalized_obs = self._normalize_observation(obs)

        info = "Debug:" + "actions performed:" + str(self.count_actions) + ", act:" + str(action[0]) + "," + str(action[1]) + ", dist:" + str(normalized_obs[4]) + ", rew:" + str(
            rew) + ", agent pos: (" + str(self.agent_x) + "," + str(self.agent_y) + ")", "goal pos: (" + str(
            self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)

        return normalized_obs, rew, done, info

    def reset(self):
        self.count_actions = 0
        self.positions = []
        # set initial state randomly
        # self.agent_x = self.np_random.uniform(low=0, high=self.len_court_x)
        # self.agent_y = self.np_random.uniform(low=0, high=self.len_court_y)
        self.agent_x = 10
        self.agent_y = 240
        # self.goal_x = self.np_random.uniform(low=0, high=self.len_court_x)
        # self.goal_y = self.np_random.uniform(low=0, high=self.len_court_x)
        self.goal_x = 125
        self.goal_y = 125
        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale, self.goal_y*self.scale)

        obs = self._observation()
        return self._normalize_observation(obs)

    def render(self, mode='human'):
        if mode == 'ansi':
            return self._observation()
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            #track the way, the agent has gone
            self.track_way = rendering.make_polyline(np.dot(self.positions, self.scale))
            self.track_way.set_linewidth(4)
            self.viewer.add_geom(self.track_way)

            # draw the agent
            car = rendering.make_circle(5)
            self.agent_trans = rendering.Transform()
            car.add_attr(self.agent_trans)
            car.set_color(0, 0, 255)
            self.viewer.add_geom(car)

            goal = rendering.make_circle(5)
            goal.add_attr(rendering.Transform(translation=(self.goal_x*self.scale, self.goal_y*self.scale)))
            goal.set_color(255, 0, 0)
            self.viewer.add_geom(goal)

            self.agent_trans.set_translation(self.agent_x * self.scale, self.agent_y * self.scale)

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        elif mode == "rgb_array":
            super(Nav2dEnv, self).render(mode=mode)
        else:
            super(Nav2dEnv, self).render(mode=mode)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
