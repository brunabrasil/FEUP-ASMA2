import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

class FreewayRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FreewayRewardWrapper, self).__init__(env)
        self.previous_state = None
        self.lanes_rewarded = []
        self.last_lane = None

    def reset(self, **kwargs):
        self.previous_state = self.env.reset(**kwargs)
        self.lanes_rewarded = []
        self.last_lane = None
        return self.previous_state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Extracting the y position of the chicken (assuming y_position is calculated correctly)
        y_position = state[97:104, 54:59, 0].sum()
        
        # Calculate the lane index (assuming 10 lanes)
        lane_height = state.shape[0] / 10
        lane_index = int(y_position // lane_height)  # Calculate the current lane index

        # Calculate the reward
        custom_reward = 0
        if lane_index not in self.lanes_rewarded:
            custom_reward += 1  # Passed a lane for the first time
            self.lanes_rewarded.append(lane_index)
        if terminated:
            self.lanes_rewarded = []  # Reset the lanes rewarded
            custom_reward += 20  # Reached the end of the freeway

        self.last_lane = lane_index
        self.previous_state = state
        return state, custom_reward, terminated, truncated, info

# Directories for saving models and logs
models_dir = "models/A2C_freeway"

# Load the latest model from the checkpoint
latest_model = models_dir + "/30000"

# Create the environment
env = make_vec_env(lambda: FreewayRewardWrapper(gym.make("ALE/Freeway-v5", render_mode = "human")), n_envs=1)

# Load the model
model = A2C.load(latest_model, env=env)

# Function to run the model and render the game
def enjoy_game(env, model, episodes=5):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            total_reward += rewards
            env.render()
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Run and render the game
enjoy_game(env, model)

# Close the environment properly
env.close()