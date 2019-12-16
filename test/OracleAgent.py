import gym
#import gym_nav2d
import time
import math
import numpy as np

DEBUG = False

for e in range(100):
    # env = gym.make('gym_nav2d:nav2d-v0')
    # env = gym.make('gym_nav2d:nav2dVeryEasy-v0')
    # env = gym.make('gym_nav2d:nav2dEasy-v0')
    # env = gym.make('gym_nav2d:nav2dHard-v0')
    env = gym.make('gym_nav2d:nav2dVeryHard-v0')
    obs = env.reset()
    cumulated_reward = 0

    i = 0
    done = False
    while not done and i <= 50:
        i += 1

        agent_x = env.agent_x
        agent_y = env.agent_y

        goal_x = env.goal_x
        goal_y = env.goal_y

        distance = env._distance()
        adjacent = (goal_y - agent_y)
        disjacent = (goal_x - agent_x)
        print(math.atan2(adjacent, disjacent))
        angle = math.atan2(adjacent, disjacent) + math.pi*1.5

        # act = env.action_space.sample()
        if distance > 10:
            dist = 10
        else:
            dist = distance

        if angle > 2*math.pi:
            #print(angle)
            angle -= 2*math.pi

        angle_grad = angle/(2*math.pi)*360
        angle_a = angle/(2*math.pi)*2-1
        dist_a = dist/10*2-1
        act = np.array([angle_a, dist_a])
        obs, rew, done, info = env.step(act)     # take a random action
        env.render(mode='human')

        cumulated_reward += rew
        if DEBUG:
            print(info)
    if DEBUG and done:
        time.sleep(3)
    print("episode ended with cumulated rew", cumulated_reward, "and done:", done)
    env.close()
