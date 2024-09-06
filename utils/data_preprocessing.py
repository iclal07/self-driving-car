import cv2
import numpy as np

def preprocess_image(image):
    """
    Görüntüyü OpenAI Gym simülasyon ortamı için işler.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale'e çevir
    resized = cv2.resize(gray, (84, 84))  # Yeniden boyutlandır
    normalized = resized / 255.0  # Normalize et
    return np.expand_dims(normalized, axis=0)
