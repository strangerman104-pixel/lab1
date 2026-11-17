import cv2
mport numpy as np

I = cv2.imread('messi.jpeg', 1)
rows, cols, _ = I.shape

I_gray = np.zeros((rows, cols), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        b, g, r = I[i, j]
        I_gray[i, j] = (int(r) + int(g) + int(b)) // 3

I_binary = np.zeros((rows, cols), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        I_binary[i, j] = 255 if I_gray[i, j] > 127 else 0

cv2.imshow('Original', I)
cv2.imshow('Grayscale', I_gray)
cv2.imshow('Binary (Threshold 127)', I_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()