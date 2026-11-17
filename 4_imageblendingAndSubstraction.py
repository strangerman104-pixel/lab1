import cv2
import numpy as np

img1 = cv2.imread('messi2.jpeg')
img2 = cv2.resize(cv2.imread('messi3.jpeg'), (img1.shape[1], img1.shape[0]))

blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
abs_diff = cv2.convertScaleAbs(cv2.subtract(img1.astype(np.float32), img2.astype(np.float32)))

cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.imshow('Blended', blended)
cv2.imshow('Absolute Difference', abs_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()