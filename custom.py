import gym
import os
from stable_baselines3 import PPO


class CustomAtlantisEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomAtlantisEnv, self).__init__(env)
        self.previous_state = None
        self.lanes_rewarded = []
        self.last_lane = None

    def reset(self, **kwargs):
        self.previous_state = self.env.reset(**kwargs)
        self.lanes_rewarded = []
        self.last_lane = None
        return self.previous_state

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        y_position = state[97:104, 54:59, 0].sum()

        # Calculate the lane index (assuming 10 lanes)
        lane_height = state.shape[0] / 10
        lane_index = int(y_position // lane_height)  # Calculate the current lane index

        # Calculate the reward
        custom_reward = 0
        if lane_index not in self.lanes_rewarded:
            custom_reward += 1  # Passed a lane for the first time
            self.lanes_rewarded.append(lane_index)

        self.last_lane = lane_index
        self.previous_state = state
        print(custom_reward)
        return state, custom_reward, done, truncated, info
    


ppo_params = {
    'learning_rate': 0.0007,
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
  

models_dir = "models/CUSTOM_RECURRENT"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the environment
env = gym.make('ALE/Atlantis-v5', render_mode="rgb_array", obs_type="grayscale")
env.reset()

# Wrap the environment with the custom wrapper
env = CustomAtlantisEnv(env)

# Initialize the model
model = PPO('CnnPolicy', env, **ppo_params, verbose=1)
