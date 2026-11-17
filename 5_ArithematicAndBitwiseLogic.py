import cv2
import numpy as np

img1 = cv2.imread('image1.jpg')
img2 = cv2.resize(cv2.imread('image2.jpg'), (img1.shape[1], img1.shape[0]))
rows, cols, _ = img1.shape

translated = cv2.warpAffine(img1, np.float32([[1, 0, 50], [0, 1, 70]]), (cols, rows))
scaled = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
rotated = cv2.warpAffine(img1, cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.0), (cols, rows))
blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
abs_diff = cv2.convertScaleAbs(cv2.subtract(img1.astype(np.float32), img2.astype(np.float32)))

_, bin1 = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
_, bin2 = cv2.threshold(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

cv2.imshow('Translated', translated)
cv2.imshow('Scaled', scaled)
cv2.imshow('Rotated', rotated)
cv2.imshow('Blended', blended)
cv2.imshow('Difference', abs_diff)
cv2.imshow('Bitwise AND', cv2.bitwise_and(bin1, bin2))
cv2.imshow('Bitwise OR', cv2.bitwise_or(bin1, bin2))
cv2.imshow('Bitwise XOR', cv2.bitwise_xor(bin1, bin2))
cv2.waitKey(0)
cv2.destroyAllWindows()