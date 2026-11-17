import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('messi.jpeg')
color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gammas = [0.2, 0.5, 1.5, 2, 3]

def gamma_correction(img, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(img, table)

for title, src, cmap in [("Grayscale", gray, 'gray'), ("Colour", color, None)]:
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 3, 1); plt.imshow(src, cmap=cmap); plt.title("Original"); plt.axis("off")
    for i, g in enumerate(gammas):
        plt.subplot(2, 3, i + 2)
        plt.imshow(gamma_correction(src, g), cmap=cmap)
        plt.title(f"Gamma = {g}")
        plt.axis("off")
    plt.suptitle(f"Gamma Correction - {title}")
    plt.tight_layout()
    plt.show()