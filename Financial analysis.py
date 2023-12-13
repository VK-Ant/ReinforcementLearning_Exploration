# 1. Bring in Marketwatch GME Data 
#https://www.marketwatch.com/investing/stock/gme/download-data?startDate=11/1/2019&endDate=03/12/2021



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

#Initialize the Environments
env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
print("Envirnoment signals in our data: ",env.signal_features)

#Build Environments
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