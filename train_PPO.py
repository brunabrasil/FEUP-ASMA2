import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

ppo_params = {
    'learning_rate': 2.5e-4,
    'n_steps': 128,
    'batch_size': 256,
    'n_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.1,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'tensorboard_log': "./ppo_freeway_tensorboard/"
}

models_dir = "models/PPO_freeway"
logdir = "logs_PPO_freeway"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_vec_env("ALE/Freeway-v5", n_envs= 4)
env.reset()

model = PPO('CnnPolicy', env, **ppo_params, verbose=1)

TIMESTEPS = 10000
for i in range(1,51):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")