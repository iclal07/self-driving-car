import cv2
import numpy as np

def augment_image(image):
    """
    Görüntüyü rastgele döndür, yakınlaştır, veya parlaklık değiştir.
    """
    # Rastgele döndürme
    angle = np.random.uniform(-15, 15)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Rastgele parlaklık değişikliği
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    value = np.random.uniform(0.6, 1.4)
    hsv[..., 2] = np.clip(hsv[..., 2] * value, 0, 255)
    augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return augmented
