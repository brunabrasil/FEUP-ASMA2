import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

# Directories for saving models and logs
models_dir = "models/A2C_SpaceInvaders"

# Load the latest model from the checkpoint
latest_model = models_dir + "/90000"

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

model = A2C.load(latest_model, env=env)

# Function to run the model and render the game
def enjoy_game(env, model, episodes=5):
    for episode in range(episodes):
        obs, _ = env.reset()  # Extract observation
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, info = env.step(action)  # Unpack the extra value
            total_reward += rewards
            env.render()
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Run and render the game
enjoy_game(env, model)

# Close the environment properly
env.close()
