import cv2
import numpy as np

img = cv2.imread('messi1.jpeg')
rows, cols, _ = img.shape

translated = cv2.warpAffine(img, np.float32([[1, 0, 50], [0, 1, 70]]), (cols, rows))
scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
rotated = cv2.warpAffine(img, cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.0), (cols, rows))

cv2.imshow('Original', img)
cv2.imshow('Translated', translated)
cv2.imshow('Scaled', scaled)
cv2.imshow('Rotated', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()