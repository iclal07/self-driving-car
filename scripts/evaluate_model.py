import gym
from stable_baselines3 import PPO
import numpy as np

def evaluate(num_episodes=10):
    # Ortamı yükle
    env = gym.make('CarRacing-v0')

    # Eğitilmiş modeli yükle
    model = PPO.load("models/ppo_car_racing")

    # Değerlendirme metrikleri
    rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Ortalama ve standart sapma hesapla
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Average Reward: {mean_reward}, Standard Deviation: {std_reward}")
    env.close()
