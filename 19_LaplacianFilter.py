import cv2
import matplotlib.pyplot as plt

img = cv2.imread('kid.jpeg', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')

for i, size in enumerate([3, 5, 7]):
    lap = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F, ksize=size))
    plt.subplot(2, 2, i + 2)
    plt.imshow(lap, cmap='gray')
    plt.title(f'Laplacian - {size}x{size}')
    plt.axis('off')
plt.tight_layout()
plt.show()