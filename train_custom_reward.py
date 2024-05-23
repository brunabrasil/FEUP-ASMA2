import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import os
import numpy as np

# Parameters for A2C
a2c_params = {
    'tensorboard_log': "./a2c_freeway_default_tensorboard/"
}

# Directories for saving models and logs
models_dir = "models/A2C_freeway_default"
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
