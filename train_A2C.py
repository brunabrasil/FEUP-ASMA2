import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import os

a2c_params = {
    'learning_rate': 0.0007,  # Learning rate
    'n_steps': 5,  # Number of steps to run for each environment per update
    'gamma': 0.99,  # Discount factor
    'gae_lambda': 1.0,  # GAE lambda parameter
    'ent_coef': 0.01,  # Entropy coefficient
    'vf_coef': 0.5,  # Value function coefficient in the loss function
    'max_grad_norm': 0.5,  # Maximum norm for the gradient clipping
    'tensorboard_log': "./a2c_SpaceInvaders_tensorboard/"
}

models_dir = "models/A2C_SpaceInvaders"
version = 10000

env = make_vec_env("ALE/SpaceInvaders-v5", n_envs= 4)
env.reset()

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model_path = f"{models_dir}/{version}"
if os.path.exists(model_path + ".zip"):
    model = A2C.load(model_path, env=env)
    print(f"Loaded model from {model_path}")
else:
    model = A2C('CnnPolicy', env, **a2c_params, verbose=1)
    print("Training new model from scratch")



TIMESTEPS = 10000
for i in range(int(version/10000), 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")