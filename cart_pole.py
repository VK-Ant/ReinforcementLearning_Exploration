#import necessary library
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

print("ALL packages set")

#load the environment
import gym

env = gym.make(
    "LunarLander-v2",render_mode="rgb_array"
)

episodes = 20

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info,prob = env.step(action)
        score += reward
    print("Episode:{}, Score: {}".format(episode,score))

env.close()