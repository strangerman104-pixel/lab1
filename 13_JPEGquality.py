import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

image = cv2.imread('messi.jpeg')
os.makedirs('jpeg_quality', exist_ok=True)
levels = [90, 70, 50, 30, 10]
results = []

plt.figure(figsize=(14, 7))
for i, q in enumerate(levels):
    path = os.path.join('jpeg_quality', f'image_quality_{q}.jpg')
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, q])
    results.append({"Quality": q, "File Size (KB)": round(os.path.getsize(path) / 1024, 2)})

    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    plt.title(f"JPEG Quality = {q}")
    plt.axis("off")

print(pd.DataFrame(results).to_string(index=False))
plt.tight_layout()
plt.show()