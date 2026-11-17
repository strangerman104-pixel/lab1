import cv2

I = cv2.imread('messi.jpeg', -1)
I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
_, I_thresh = cv2.threshold(I_gray, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('original_image.jpeg', I)
cv2.imwrite('rgb_image.jpeg', I_rgb)
cv2.imwrite('grayscale_image.jpeg', I_gray)
cv2.imwrite('binary_image.jpeg', I_thresh)
print(f'Image shape: {I_rgb.shape}')

cv2.imshow('Original', I)
cv2.imshow('RGB', I_rgb)
cv2.imshow('Grayscale', I_gray)
cv2.imshow('Binary', I_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()