import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import os

a2c_params = {
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0,
    'ent_coef': 0.01,
    'vf_coef': 0.25,
    'tensorboard_log': "./a2c_frogger_tensorboard/"
}

models_dir = "models/A2C_Params"
logdir = "logs_A2C_Params"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_vec_env("ALE/Frogger-v5", n_envs= 4)
env.reset()

model = A2C('CnnPolicy', env, **a2c_params, verbose=1)

TIMESTEPS = 10000
for i in range(1,31):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")