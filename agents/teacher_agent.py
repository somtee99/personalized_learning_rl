import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.question_selection_env import QuestionSelectionEnv
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

questions_df = pd.read_csv("./data/questions.csv")
os.makedirs("./trainings", exist_ok=True)
os.makedirs("./models/teacher", exist_ok=True)

env = QuestionSelectionEnv(questions_df, max_steps=250)

vec_env = make_vec_env(lambda: env, n_envs=1)
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./trainings",
    learning_rate = 4e-4,      # Step size for policy updates; lower values = slower, more stable learning
    clip_range = 0.25,         # Limits how much the policy can change per update; prevents large, unstable jumps
    vf_coef = 0.4,             # Weight for value function loss; higher = more focus on value prediction
    gae_lambda = 0.95,         # Controls how much future rewards are considered in advantage estimation; higher = more smoothing
    gamma = 0.99,              # Discount factor for future rewards; higher = more focus on long-term rewards
    ent_coef = 0.05,           # Encourages exploration by adding entropy to the loss; higher = more random actions
    batch_size = 128,          # Number of samples per update; larger = more stable gradients, slower updates
    n_steps = 4096,            # Number of environment steps to collect before each update; larger = better estimates, more memory
    max_grad_norm = 0.5,       # Clips gradients to prevent exploding updates; improves training stability
    target_kl = 0.02,          # Stops training early if policy changes too much; helps prevent instability
    n_epochs = 10              # Number of passes over each batch per update; higher = more thorough learning per batch
)
print("Starting training...")
model.learn(total_timesteps=100000)
model.save("./models/teacher/ppo_teacher_agent")
