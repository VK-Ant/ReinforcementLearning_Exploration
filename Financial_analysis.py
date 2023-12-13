'''
MarketWatch GME Data prediction using Reinforcement learing

Data links: https://www.marketwatch.com/investing/stock/gme/download-data?startDate=11/1/2019&endDate=03/12/2021

Steps:

1. Read the data
2. Initialize the Environment
3. Build stock environment and Train the data
4. Evaluating and refer the output

credit: https://youtu.be/D9sU1hLT0QY?si=Xfv0LXIXfX1s2wYh

All are exploration in reinforcement learning try it yourself........

'''

#import necessary library
# Gym stuff
import gym
import gym_anytrading
# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Read the data
df = pd.read_csv('data/gmedata.csv')
print(df.head())

df['Date'] = pd.to_datetime(df['Date'])
#print(df.dtypes)

df.set_index('Date', inplace=True)
#print(df.head())

#RL Development
#1. Initialize the Environments
env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
print("Envirnoment signals in our data: ",env.signal_features)

#2. Build Environments
print(env.action_space)
state = env.reset()
while True: 
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done: 
        print("info", info)
        break

#Display the graph        
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

#3. Building Environment and Training
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env = DummyVecEnv([env_maker])

#Models (Stable_Baseline)
model = A2C('MlpLstmPolicy', env, verbose=1) 
model.learn(total_timesteps=1000)

#4. Evaluation

env = gym.make('stocks-v0', df=df, frame_bound=(90,110), window_size=5)
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()