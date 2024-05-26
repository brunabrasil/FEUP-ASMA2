import gymnasium as gym
import os
from stable_baselines3 import PPO


class CustomFreeway(gym.Wrapper):
    def __init__(self, env):
        super(CustomFreeway, self).__init__(env)
        self.previous_y = None

    def reset(self, **kwargs):
        self.previous_state = self.env.reset(**kwargs)
        self.previous_y = None
        return self.previous_state

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        y_position = state[97:104, 54:59, 0].sum()
        #y_position = self.env.unwrapped.ale.getRAM()[83]

        print(y_position)
        print(self.previous_y)
        # Calculate the reward
        custom_reward = 0
        if self.previous_y is not None:
            if y_position < self.previous_y:
                custom_reward = 5  # Moving forward
            elif y_position > self.previous_y:
                custom_reward = -1  # Moving backward

        self.previous_y = y_position
        return state, custom_reward, done, truncated, info
    


ppo_params = {
    'learning_rate': 0.0007,
    'n_steps': 128,
    'batch_size': 128,
    'n_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.1,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'tensorboard_log': "./ppo_freeway_tensorboard_custom/"
}
  

models_dir = "models/PPO_custom"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# Create the environment
env = gym.make('ALE/Freeway-v5', render_mode='human')
env.reset()

# Wrap the environment with the custom wrapper
env = CustomFreeway(env)

# Initialize the model
model = PPO('CnnPolicy', env, **ppo_params, verbose=1)

TIMESTEPS = 10000
for i in range(1,201):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
