import cv2
import matplotlib.pyplot as plt

_, img_bin = cv2.threshold(cv2.imread('stop.jpg', 0), 127, 255, cv2.THRESH_BINARY)
img_color = cv2.cvtColor(cv2.imread('kid.jpeg'), cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

pairs = [
    (img_bin, 255 - img_bin, "Binary", 'gray'),
    (img_gray, 255 - img_gray, "Grayscale", 'gray'),
    (img_color, 255 - img_color, "Color", None)
]

plt.figure(figsize=(10, 10))
for idx, (orig, neg, label, cmap) in enumerate(pairs):
    plt.subplot(3, 2, 2 * idx + 1)
    plt.imshow(orig, cmap=cmap)
    plt.title(f"Original {label}")
    plt.axis('off')

    plt.subplot(3, 2, 2 * idx + 2)
    plt.imshow(neg, cmap=cmap)
    plt.title(f"Negative {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()