# randomItems.py
import sys
import cv2 as cv
import numpy as numpy
from matplotlib import pyplot as plt


# def fetchBinaryImage():

srcimagefilename = "randomItems2.png"
# srcImg = cv.imread(srcimagefilename, cv.IMREAD_UNCHANGED)
srcImg = cv.imread(srcimagefilename, cv.IMREAD_COLOR)
if srcImg is None:
    sys.exit("Could not read" + srcimagefilename)
plt.figure("origional color image")
plt.imshow(srcImg)
(R, G, B) = cv.split(srcImg)
ret, thresh_R = cv.threshold(R, 0, 255, cv.THRESH_OTSU)
ret, thresh_G = cv.threshold(G, 0, 255, cv.THRESH_OTSU)
ret, thresh_B = cv.threshold(B, 5, 255, cv.THRESH_BINARY_INV)
plt.figure("R, G, B, channels")
plt.subplot(1, 3, 1), plt.imshow(thresh_R, cmap="gray")
plt.title("red channel")
plt.subplot(1, 3, 2), plt.imshow(thresh_G, cmap="gray")
plt.title("green channel")
plt.subplot(1, 3, 3), plt.imshow(thresh_B, cmap="gray")
plt.title("blue channel")
# plt.figure("R, G, B, channels")
# RGB = cv.merge([thresh_R, thresh_G, thresh_B])
# plt.imshow(RGB, cmap="gray")
# plt.figure("grayscale color image")
# # gray = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
# gray = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
# plt.imshow(gray, cmap="gray")
# Combine channels to create single binary image
# binary_combined = numpy.zeros(
#     [srcImg.shape[0], srcImg.shape[1]], dtype=numpy.float32
# )
# for channel in [thresh_R, thresh_G, thresh_B]:
#     for width in range(channel.shape[0]):
#         for height in range(channel.shape[1]):
#             # print(channel[width][height])
#             if channel[width][height] > 0.9 * channel.max():
#                 srcImg[width][height] = [255, 255, 255]
for width in range(thresh_R.shape[0]):
    for height in range(thresh_R.shape[1]):
        # print(thresh_R[width][height])
        if thresh_R[width][height] > 0.9 * thresh_R.max():
            srcImg[width][height][0] = 255
        else:
            srcImg[width][height][0] = 0
for width in range(thresh_G.shape[0]):
    for height in range(thresh_G.shape[1]):
        # print(thresh_G[width][height])
        if thresh_G[width][height] > 0.9 * thresh_G.max():
            srcImg[width][height][1] = 255
        else:
            srcImg[width][height][1] = 0
for width in range(thresh_B.shape[0]):
    for height in range(thresh_B.shape[1]):
        # print(thresh_B[width][height])
        if thresh_B[width][height] > 0.9 * thresh_B.max():
            srcImg[width][height][2] = 255
        else:
            srcImg[width][height][2] = 0

plt.figure("combined color channels after thresholding")
plt.imshow(srcImg)
plt.title("combined color channels after thresholding")

# plt.figure("combined")
# plt.imshow(binary_combined, cmap="gray")


plt.show()
# print("DONE")


# return srcImg
