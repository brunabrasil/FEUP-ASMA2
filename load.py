import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

models_dir = "models/A2C_Params"

# Create the environment
env = gym.make('ALE/Frogger-v5', render_mode="human")
env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)

# Load the trained model
model_path = f"{models_dir}/300000"
model = A2C.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)
