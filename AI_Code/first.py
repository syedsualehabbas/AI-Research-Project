import gymnasium as gym
env = gym.make("HalfCheetah-v5")
obs, info = env.reset()
print(obs.shape)