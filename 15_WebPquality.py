import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

img = cv2.imread('messi2.jpeg')
qualities = [100, 70, 50, 30, 10]
data = []

plt.figure(figsize=(14, 7))
for i, q in enumerate(qualities):
    fname = f'webp_{q}.webp'
    cv2.imwrite(fname, img, [cv2.IMWRITE_WEBP_QUALITY, q])
    data.append([q, os.path.getsize(fname)])

    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB))
    plt.title(f'WebP Quality = {q}')
    plt.axis('off')

print(pd.DataFrame(data, columns=['Quality', 'File Size (bytes)']).to_string(index=False))
plt.tight_layout()
plt.show()