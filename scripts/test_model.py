import gym
from stable_baselines3 import PPO

def test():
    # Ortamı yükle
    env = gym.make('CarRacing-v2')  # v0'dan v2'ye geçtik

    # Eğitilmiş modeli yükle
    model = PPO.load("models/ppo_car_racing")

    # Modeli test et
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()
    print("Model başarıyla test edildi.")
