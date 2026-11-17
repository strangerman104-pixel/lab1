import cv2
import matplotlib.pyplot as plt

gray = cv2.imread('messi1.jpeg', cv2.IMREAD_GRAYSCALE)
equalized = cv2.equalizeHist(gray)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1); plt.imshow(gray, cmap='gray'); plt.title("Original Grayscale"); plt.axis("off")
plt.subplot(2, 2, 2); plt.hist(gray.ravel(), 256, (0, 256), color='black'); plt.title("Histogram Before")
plt.subplot(2, 2, 3); plt.imshow(equalized, cmap='gray'); plt.title("Equalized Image"); plt.axis("off")
plt.subplot(2, 2, 4); plt.hist(equalized.ravel(), 256, (0, 256), color='black'); plt.title("Histogram After")
plt.tight_layout()
plt.show()