import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import os
import numpy as np

class FreewayRewardWrapper(Wrapper):
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

# Parameters for A2C
a2c_params = {
    'tensorboard_log': "./a2c_freeway_tensorboard/"
}

# Directories for saving models and logs
models_dir = "models/A2C_freeway"
version = 10000

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create custom environment
env = make_vec_env(lambda: FreewayRewardWrapper(gym.make("ALE/Freeway-v5")), n_envs=4)

# Load the model from the checkpoint
model_path = f"{models_dir}/{version}"
if os.path.exists(model_path + ".zip"):
    model = A2C.load(model_path, env=env)
    print(f"Loaded model from {model_path}")
else:
    model = A2C('CnnPolicy', env, **a2c_params, verbose=1)
    print("Training new model from scratch")

# Number of timesteps to train in each iteration
TIMESTEPS = 10000

# Continue training for additional timesteps
for i in range(int(version/10000), 200):  # Adjust the range to continue training
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

print("Training complete.")
