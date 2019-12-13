from gym.envs.registration import register

register(
    id='nav2d-v0',
    entry_point='gym_nav2d.envs:Nav2dEnv',
)