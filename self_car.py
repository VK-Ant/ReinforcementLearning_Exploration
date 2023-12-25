import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import imageio

print("ALL packages set")

# Load the environment
env = gym.make("BipedalWalker-v3", hardcore=True,render_mode="rgb_array")
'''
episodes = 5

for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    images = []  # To store frames for creating the video

    while not done:
        frame = env.render()
        images.append(frame)
        action = env.action_space.sample()
        n_state, reward, done, info, prob = env.step(action)
        score += reward

    # Save the frames as an MP4 file
    imageio.mimsave(f'episode_biwalker_{episode}.mp4', images)

    print("Episode:{}, Score: {}".format(episode, score))

env.close()
'''

print(env.action_space.sample())
