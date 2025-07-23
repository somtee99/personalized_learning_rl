import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment.question_selection_env import QuestionSelectionEnv

df = pd.read_csv("data/cleaned_df.csv")

def clean_skill_list(skill_str):
    if pd.isnull(skill_str):
        return []
    return [s.strip() for s in skill_str.split(',') if s.strip() != '']

df['skill_name_split'] = df['skill_name'].apply(clean_skill_list)
all_skills = sorted({skill for skills in df['skill_name_split'] for skill in skills})

print(all_skills)

env = QuestionSelectionEnv(all_skills=all_skills)

vec_env = make_vec_env(lambda: env, n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_teacher_agent")