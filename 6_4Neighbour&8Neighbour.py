import cv2
import numpy as np

img = cv2.imread('neighbour.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
rows, cols = binary_img.shape

c4_mat = np.zeros((rows, cols), dtype=np.uint8)
c8_mat = np.zeros((rows, cols), dtype=np.uint8)
offsets = [
    (-1, -1, False), (-1, 0, True), (-1, 1, False),
    (0, -1, True),                  (0, 1, True),
    (1, -1, False),  (1, 0, True),  (1, 1, False)
]

for i in range(rows):
    for j in range(cols):
        curr = binary_img[i, j]
        c4, c8 = 0, 0
        for r_off, c_off, is_4 in offsets:
            nr, nc = i + r_off, j + c_off
            if 0 <= nr < rows and 0 <= nc < cols and binary_img[nr, nc] == curr:
                c8 += 1
                if is_4:
                    c4 += 1
        c4_mat[i, j], c8_mat[i, j] = c4, c8

np.set_printoptions(linewidth=200, threshold=np.inf)
print("ORIGINAL BINARY IMAGE MATRIX:\n", binary_img)
print("\n4-NEIGHBOR COUNT MATRIX:\n", c4_mat)
print("\n8-NEIGHBOR COUNT MATRIX:\n", c8_mat)