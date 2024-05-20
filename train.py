import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import os

# Parameters for A2C
a2c_params = {
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0,
    'ent_coef': 0.01,
    'vf_coef': 0.25,
    'tensorboard_log': "./a2c_frogger_tensorboard/"
}

# Directories for saving models and logs
models_dir = "models/A2C_Params"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create environment
env = make_vec_env("ALE/Frogger-v5", n_envs=4)

# Load the model from the checkpoint
model_path = f"{models_dir}/300000"
# if os.path.exists(model_path):
model = A2C.load(model_path, env=env)
print(f"Loaded model from {model_path}")
# else:
#     model = A2C('CnnPolicy', env, **a2c_params, verbose=1)
#     print("Training new model from scratch")

# Number of timesteps to train in each iteration
TIMESTEPS = 10000

# Continue training for additional timesteps
for i in range(31, 61):  # Adjust the range to continue training
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

print("Training complete.")
