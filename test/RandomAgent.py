import gym
import gym_nav2d
env = gym.make('nav2d-v0')
env.reset()
for _ in range(10):
    env.render()
    # env.step(env.action_space.sample()) # take a random action
    env.step(1)
env.close()
