import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread('messi3.jpeg'), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1); plt.imshow(img); plt.title('Original'); plt.axis('off')

for i, size in enumerate([3, 5, 7]):
    kernel = np.ones((size, size), np.float32) / (size * size)
    plt.subplot(2, 2, i + 2)
    plt.imshow(cv2.filter2D(img, -1, kernel))
    plt.title(f'{size}x{size} Averaging Filter')
    plt.axis('off')
plt.tight_layout()
plt.show()