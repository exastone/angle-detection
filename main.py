# main.py

import os
import sys
import cv2 as cv
import numpy
import json

# import imutils
from matplotlib import pyplot as plt

import corners as corners
import tests as tests
import fetchRandomItemsImage as fetchItemsImage


""" srcimagefilename = "triangle-filled.png"
# srcimagefilename = "shapeshollow.png"
# srcImg = cv.imread(srcimagefilename, cv.IMREAD_UNCHANGED)
srcImg = cv.imread(srcimagefilename, cv.IMREAD_COLOR)

if srcImg is None:
    sys.exit("Could not read" + srcimagefilename)

grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
grayImg = numpy.asarray(grayImg, dtype=numpy.float32)


# plt.imshow(srcImg, cmap="gray")
# plt.title("Original Image")

# detetedCorner = cv.cornerHarris(grayImg, 2, 3, 0.02)
# plt.imshow(detetedCorner, cmap="gray")

# f = corners.detectCorners(srcImg, 4, 3, 0.03)

mymatrix = numpy.zeros([200, 200])
test = tests.matrixToImage(mymatrix)


plt.show()
print("DONE") """


# Import image for shape identification, change srcImgFileName to match the desired image name
# in this folder, update srcImgInvert if image inversion is desired
# srcImgFileName = "shapesfilled.png"
srcImgFileName = "randomItems2.png"
srcImgInvert = True

srcImg = cv.imread(srcImgFileName, cv.IMREAD_COLOR)

# Exit program if file does not exist
if srcImg is None:
    sys.exit("Could not read image file.")

# Perform inversion of image; this step may not always be necessary but can help if correct
# edge detection fails initially
if srcImgInvert == True:
    srcImg = cv.bitwise_not(srcImg)

# grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)


# plt.figure("original color")
# plt.imshow(
#     srcImg,
# )
# cv.imshow("Image", srcImg)
# cv.waitKey(0)


srcFile2 = "randomItems3.png"
srcImg2 = cv.imread(srcFile2, cv.IMREAD_COLOR)
# plt.figure("original gray")
grayImg = cv.cvtColor(srcImg2, cv.COLOR_RGB2GRAY)
# plt.imshow(grayImg, cmap="gray")


""" plt.figure("Gaussian Blur and laplacian filtered")
img_blurred = cv.GaussianBlur(grayImg, (5, 5), cv.BORDER_DEFAULT)
img_laplacian = cv.Laplacian(img_blurred, -1, ksize=3)
plt.imshow(img_laplacian, cmap="gray")


plt.figure("Bilater Filter Applied")
img_bilat = cv.bilateralFilter(img_laplacian, 3, 3, cv.BORDER_DEFAULT)
plt.imshow(img_bilat) """


# plt.figure("histogrom")
# plt.hist(img_laplacian.ravel(), 256, (0, 255))
# plt.title("histogram")


# img_laplacian = cv.threshold(img_laplacian, 20, 255, cv.THRESH_BINARY)[1]
# plt.imshow(img_laplacian, cmap="gray")


# plt.figure("blurred and laplacian + threshold")
# grayImgThresh = cv.threshold(img_laplacian, 10, 255, cv.THRESH_BINARY)[1]
# plt.imshow(grayImgThresh, cmap="gray")


"""
Note: Not necessary for trivial images where shape are clearly defined
    Blurring helps to enhance any discreet edges in the image, and the
    Blur the image slightly, then perform thresholding
# grayImgBlurred = cv.GaussianBlur(grayImg, (5, 5), cv.BORDER_DEFAULT)
"""
grayImg = cv.GaussianBlur(grayImg, (5, 5), cv.BORDER_DEFAULT)


# thresholding aids in discerning between the shapes and the background
# plt.figure("threashold")
grayImgThresh = cv.threshold(grayImg, 200, 255, cv.THRESH_BINARY_INV)[1]
# plt.imshow(grayImgThresh, cmap="gray")

# plt.figure("Morph")
# kernel = numpy.ones((3, 3), numpy.uint8)
# closing = cv.morphologyEx(grayImgThresh, cv.MORPH_CLOSE, kernel)
# plt.imshow(closing, cmap="gray")

# Find the contours of the thresholded image; this is a crucial step in shape identification
grayImgContours = cv.findContours(grayImgThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[
    0
]

# Create a new shape detector class, which will determine what shapes are present based on the contours
shape = corners.ShapeDetector()


for index, con in enumerate(grayImgContours):

    # Calculate the moments of the contours in the image, allowing for shape identification
    moments = cv.moments(con)

    conX = int((moments["m10"] / moments["m00"]))
    conY = int((moments["m01"] / moments["m00"]))
    shape_detect = shape.detectShape(con)

    # with open(f"moments_{shape_detect}.json", "w") as json_file:
    #     json.dump(moments, json_file)

    # print(f"Centroid of {shape_detect} cx: {conX}, cy: {conY}")

    con = con.astype("float")
    con = con.astype("int")
    cv.drawContours(srcImg, [con], 0, (0, 255, 0), 2)

    if srcImgInvert == True:
        cv.putText(
            srcImg,
            shape_detect,
            (conX, conY),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    else:
        cv.putText(
            srcImg,
            shape_detect,
            (conX, conY),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    plt.figure()
    plt.imshow(srcImg)
    # cv.imshow("Image", srcImg)
    # cv.waitKey(0)
plt.show()
