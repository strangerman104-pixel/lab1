import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

image = cv2.imread('kid.jpeg')
os.makedirs('png_compression', exist_ok=True)
levels = [0, 3, 5, 7, 9]
results = []

plt.figure(figsize=(14, 7))
for i, lvl in enumerate(levels):
    path = os.path.join('png_compression', f'image_compression_{lvl}.png')
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, lvl])
    results.append({"Compression": lvl, "File Size (KB)": round(os.path.getsize(path) / 1024, 2)})

    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    plt.title(f"PNG Compression = {lvl}")
    plt.axis("off")

print(pd.DataFrame(results).to_string(index=False))
plt.tight_layout()
plt.show()