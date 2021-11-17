import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test.png', 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Blurring for removing the noise
img_blur = cv2.bilateralFilter(img, d = 7,
                               sigmaSpace = 75, sigmaColor =75)
# Convert to grayscale
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
# Apply the thresholding
a = img_gray.max()
_, thresh = cv2.threshold(img_gray, a/2+60, a,cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap = 'gray')

# Find the contour of the figure
image, contours, hierarchy = cv2.findContours(
                                   image = thresh,
                                   mode = cv2.RETR_TREE,
                                   method = cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours
contours = sorted(contours, key = cv2.contourArea, reverse = True)
# Draw the contour
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx = -1,
                         color = (255, 0, 0), thickness = 2)
plt.imshow(img_copy)

# The first order of the contours
c_0 = contours[0]
# image moment
M = cv2.moments(c_0)
print(M.keys())


# The area of contours
print("1st Contour Area : ", cv2.contourArea(contours[0])) # 37544.5
print("2nd Contour Area : ", cv2.contourArea(contours[1])) # 75.0
print("3rd Contour Area : ", cv2.contourArea(contours[2])) # 54.0

# The arc length of contours
print(cv2.arcLength(contours[0], closed = True))      # 2473.3190
print(cv2.arcLength(contours[0], closed = False))     # 2472.3190

plt.show(block=True)