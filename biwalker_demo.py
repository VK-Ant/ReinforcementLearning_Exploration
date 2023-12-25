import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import os

print("ALL packages set")

# Load the environment
env = gym.make("BipedalWalker-v3", hardcore=True,render_mode="rgb_array")


'''
Test the environment and save the episodics

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
print(env.observation_space.sample())

#Train Model

log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=4000) 

#saved model
ppo_path = os.path.join('Training', 'Saved_Models', 'PPO_Driving_model')
model.save(ppo_path)

#evaluate test
#evaluate_policy(model, env, n_eval_episodes=10, render=True)

import cv2

width = 512
height = 512

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose a different codec if needed
video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))  # Adjust width and height according to your environment

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info,probs = env.step(action)

    # Render the environment
    frame = env.render(mode='rgb_array')

    # Save the frame to the video
    video_writer.write(frame)

    # Display the frame (optional)
    cv2.imshow('Environment', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and close the environment
video_writer.release()
env.close()
cv2.destroyAllWindows()









