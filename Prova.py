import gym

env = gym.make("MountainCar-v0")
print(env.observation_space.shape[0])