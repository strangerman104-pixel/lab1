import cv2
import matplotlib.pyplot as plt

rgb = cv2.cvtColor(cv2.imread('messi2.jpeg'), cv2.COLOR_BGR2RGB)
h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
v_eq = cv2.equalizeHist(v)
eq_rgb = cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2RGB)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1); plt.imshow(rgb); plt.title("Original Colour"); plt.axis("off")
plt.subplot(2, 2, 2); plt.hist(v.ravel(), 256, (0, 256), color='black'); plt.title("V-Channel Before")
plt.subplot(2, 2, 3); plt.imshow(eq_rgb); plt.title("Equalized Colour"); plt.axis("off")
plt.subplot(2, 2, 4); plt.hist(v_eq.ravel(), 256, (0, 256), color='black'); plt.title("V-Channel After")
plt.tight_layout()
plt.show()