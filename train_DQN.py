import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import os

dqn_params = {
    'learning_rate': 0.0007,  # Learning rate
    'gamma': 0.99,  # Discount factor
    'max_grad_norm': 0.5,  # Maximum norm for the gradient clipping
    'tensorboard_log': "./dqn_Freeway_tensorboard/"
}

models_dir = "models/DQN_Freeway"
version = 10000

env = make_vec_env("ALE/Freeway-v5", n_envs= 4)
env.reset()

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model_path = f"{models_dir}/{version}"
if os.path.exists(model_path + ".zip"):
    model = DQN.load(model_path, env=env)
    print(f"Loaded model from {model_path}")
else:
    model = DQN('CnnPolicy', env, **dqn_params, verbose=1)
    print("Training new model from scratch")


TIMESTEPS = 10000
for i in range(int(version/10000), int(version/10000) + 200):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS*i}")