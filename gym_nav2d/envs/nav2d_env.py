import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
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
        self.obs_low_state = 0.             # define Box of observation
        self.high_state = math.sqrt(math.sqrt(pow(self.len_court_x, 2) + pow(self.len_court_y, 2)))
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]),  # x_agent, y_agent, x_goal, y_goal, distance
                    np.array([self.len_court_x, self.len_court_y, self.len_court_x, self.len_court_y, self.high_state]),
                                            dtype=np.float32)

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

    def step(self, action):
        self.count_actions += 1
        angle = (action[0] + 1) * math.pi
        step_size = (action[1] + 1) / 2 * self.max_step_size
        # calculate new agent state
        if 0 <= angle <= math.pi/2:
            self.agent_x = self.agent_x - math.cos(angle) * step_size
            self.agent_y = self.agent_y + math.sin(angle) * step_size
        elif math.pi/2 < angle <= math.pi:
            self.agent_x = self.agent_x + math.cos(angle) * step_size
            self.agent_y = self.agent_y - math.sin(angle) * step_size
        elif math.pi < angle <= math.pi*1.5:
            self.agent_x = self.agent_x - math.cos(angle) * step_size
            self.agent_y = self.agent_y + math.sin(angle) * step_size
        elif math.pi*1.5 < angle <= math.pi*2:
            self.agent_x = self.agent_x + math.cos(angle) * step_size
            self.agent_y = self.agent_y - math.sin(angle) * step_size

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

        info = "Debug:" + "actions performed:" + str(self.count_actions) + ", act:" + str(action[0]) + "," + str(action[1]) + ", dist:" + str(obs[4]) + ", rew:" + str(
            rew) + ", agent pos: (" + str(self.agent_x) + "," + str(self.agent_y) + ")", "goal pos: (" + str(
            self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)

        #track, where agent was
        self.positions.append([self.agent_x, self.agent_y])

        return obs, rew, done, info

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
        return self._observation()

    def render(self, mode='human'):
        if mode == 'ansi':
            return self._observation()
        elif mode == 'human':
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
