import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.question_selection_env import QuestionSelectionEnv
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

df = pd.read_csv("data/cleaned_df.csv")
os.makedirs("../trainings", exist_ok=True)
os.makedirs("../models", exist_ok=True)

env = QuestionSelectionEnv()

vec_env = make_vec_env(lambda: env, n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="../trainings")
print("Starting training...")
model.learn(total_timesteps=100000)
model.save("../models/ppo_teacher_agent")
