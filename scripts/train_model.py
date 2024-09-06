import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

def train():
    # OpenAI Gym ortamını oluştur
    env = make_vec_env('CarRacing-v2', n_envs=1)  # v0'dan v2'ye geçtik

    # PPO modelini tanımla ve hiperparametreleri ayarla
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_car_racing_tensorboard/")

    # Modeli eğit
    model.learn(total_timesteps=100000, log_interval=10)

    # Eğitilmiş modeli kaydet
    model_path = "models/ppo_car_racing"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print("Model başarıyla eğitildi ve kaydedildi.")
