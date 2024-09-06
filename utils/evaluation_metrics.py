import numpy as np

def compute_mean_reward(rewards):
    """
    Verilen ödül listesinin ortalamasını hesaplar.
    """
    return np.mean(rewards)

def compute_standard_deviation(rewards):
    """
    Verilen ödül listesinin standart sapmasını hesaplar.
    """
    return np.std(rewards)
