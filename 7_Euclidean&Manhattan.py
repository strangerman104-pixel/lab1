import cv2
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread('kid.jpeg'), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
print(f"Dimensions: Width={w}, Height={h}")

x1 = int(input("Enter P1 x: "))
y1 = int(input("Enter P1 y: "))
x2 = int(input("Enter P2 x: "))
y2 = int(input("Enter P2 y: "))

if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
    eucl = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    manh = abs(x2 - x1) + abs(y2 - y1)
    print(f"Euclidean Distance: {eucl:.4f} | Manhattan Distance: {manh}")

    plt.imshow(img)
    plt.scatter([x1, x2], [y1, y2], color='red', s=50, zorder=5)
    plt.plot([x1, x2], [y1, y2], color='blue', linestyle='--', linewidth=2)
    plt.text(x1, y1 - 10, 'P1', color='red', fontweight='bold')
    plt.text(x2, y2 - 10, 'P2', color='red', fontweight='bold')
    plt.title(f"Euclidean={eucl:.1f}, Manhattan={manh}")
    plt.show()
else:
    print("Error: Coordinates outside image boundaries!")