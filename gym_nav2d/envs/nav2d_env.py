import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import random
import math

random.seed(1337)


class Nav2dEnv(gym.Env):
    # this is a list of supported rendering modes!
    metadata = {'render.modes': ['human', 'ansi', 'rgb_array'],
                'video.frames_per_second': 30}

    def __init__(self):
        # action space: change direction in degree (discrete), run into this direction (Box)
        self.action_space = spaces.Tuple((spaces.Box(low=0, high=359, shape=(1,)),
                                          spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)))
        # how near should agent come to goal?
        self.eps = 0.1
        self.np_random = None
        self.viewer = None
        self.cartrans = None
        self.track_way = None
        # observation: distance to goal
        self.len_court_x = 255
        self.len_court_y = 255
        self.low_state = 0.
        self.high_state = math.sqrt(self.len_court_y * self.len_court_y + self.len_court_x * self.len_court_x)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=(1,))

        # agent stuff
        self.agent_x = 0
        self.agent_y = 0
        # track agent positions for drawing
        self.positions = []

        # goal stuff
        self.goal_x = 0
        self.goal_y = 0

        # rendering
        self.screen_height = 600
        self.screen_width = 600

        self.seed()
        self.reset()

    def _reward(self):
        # 100 - distance
        return 100 - abs(math.sqrt(pow((self.goal_x - self.agent_x), 2) + pow((self.goal_y - self.agent_y), 2)))

    def _observation(self):
        return abs(math.sqrt(pow((self.goal_x - self.agent_x), 2) + pow((self.goal_y - self.agent_y), 2)))

    def step(self, action):
        angle = action[0][0]
        angle_rad = angle / 360 * 2 * math.pi
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
        obs = self._observation()

        rew = self._reward()
        done = bool(obs <= self.eps or len(self.positions)>100)
        info = "Debug:" + "act:" + str(action[0][0]) + "," + str(action[1][0]) + ", obs:" + str(obs) + ", rew:" + str(
            rew) + ", agent pos: (" + str(self.agent_x) + "," + str(self.agent_y) + ")", "goal pos: (" + str(
            self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)

        #track, where agent was
        self.positions.append((self.agent_x, self.agent_y))

        return obs, rew, done, info

    def reset(self):
        # set initial state randomly
        self.agent_x = self.np_random.uniform(low=0, high=self.len_court_x)
        self.agent_y = self.np_random.uniform(low=0, high=self.len_court_y)
        self.goal_x = self.np_random.uniform(low=0, high=self.len_court_x)
        self.goal_y = self.np_random.uniform(low=0, high=self.len_court_x)
        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        return self._observation()

    def render(self, mode='human'):
        if mode == 'ansi':
            return self._observation()
        elif mode == 'human':
            scale = self.screen_width / self.len_court_x
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            #track the way, the agent has gone
            self.track_way = rendering.make_polyline(np.dot(self.positions, scale))
            self.track_way.set_linewidth(4)
            self.viewer.add_geom(self.track_way)

            # draw the agent
            car = rendering.make_circle(5)
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            car.set_color(0, 0, 255)
            self.viewer.add_geom(car)

            goal = rendering.make_circle(5)
            goal.add_attr(rendering.Transform(translation=(self.goal_x, self.goal_y)))
            goal.set_color(255, 0, 0)
            self.viewer.add_geom(goal)

            self.cartrans.set_translation(self.agent_x*scale, self.agent_y*scale)

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
