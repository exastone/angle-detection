# Source: https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# loading image
# img0 = cv.imread('SanFrancisco.jpg',)
img0 = cv.imread("randomItems2.png", cv.IMREAD_COLOR)

# converting to gray scale
gray = cv.cvtColor(img0, cv.COLOR_RGB2GRAY)

# remove noise
img = cv.GaussianBlur(gray, (3, 3), 0)

# convolute with proper kernels
laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)  # x
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)  # y

# ----------------------
plt.figure("f1")

plt.subplot(2, 2, 1), plt.imshow(img, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap="gray")
plt.title("Sobel X"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap="gray")
plt.title("Sobel Y"), plt.xticks([]), plt.yticks([])


# plt.figure("histogrom")
# plt.hist(img.ravel(), 256, (0, 255))
# plt.title("histogram")

# gray_Thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
# plt.imshow(gray_Thresh, cmap="gray")


plt.show()
