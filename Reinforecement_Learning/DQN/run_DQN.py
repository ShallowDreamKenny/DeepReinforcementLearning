import gym
env = gym.make('MountainCar-v0', render_mode="rgb_array")
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()