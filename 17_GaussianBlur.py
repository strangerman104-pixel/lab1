import cv2
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread('kid.jpeg'), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1); plt.imshow(img); plt.title('Original'); plt.axis('off')

for i, size in enumerate([3, 5, 7]):
    plt.subplot(2, 2, i + 2)
    plt.imshow(cv2.GaussianBlur(img, (size, size), 0))
    plt.title(f'Gaussian Blur - {size}x{size}')
    plt.axis('off')
plt.tight_layout()
plt.show()