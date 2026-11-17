import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('kid.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_float = gray.astype(np.float32)

c = 255 / np.log(1 + np.max(gray_float))
log_img = np.uint8(c * np.log(1 + gray_float))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(gray, cmap="gray"); plt.title("Grayscale"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(log_img, cmap="gray"); plt.title("Log Transformed"); plt.axis("off")
plt.tight_layout()
plt.show()