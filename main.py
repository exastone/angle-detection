# main.py

import sys
import cv2 as cv
from matplotlib import pyplot as plt
import corner_detect as corner_detect
import tests as tests


# update srcImgInvert if image inversion is desired
img_filename = "images/Items2.png"
srcImgInvert = True

src_img = cv.imread(img_filename, cv.IMREAD_COLOR)

# Exit program if file does not exist
if src_img is None:
    sys.exit("Could not read image file.")

# Perform inversion of image; this step may not always be necessary but can help if correct
# edge detection fails initially
if srcImgInvert == True:
    src_img = cv.bitwise_not(src_img)

img2_filename = "images/Items3.png"
src_img2 = cv.imread(img2_filename, cv.IMREAD_COLOR)
gray_img = cv.cvtColor(src_img2, cv.COLOR_RGB2GRAY)


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
"""
gray_img = cv.GaussianBlur(gray_img, (5, 5), cv.BORDER_DEFAULT)


# thresholding aids in discerning between the shapes and the background
gray_img_thresh = cv.threshold(gray_img, 200, 255, cv.THRESH_BINARY_INV)[1]

# plt.figure("Morph")
# kernel = numpy.ones((3, 3), numpy.uint8)
# closing = cv.morphologyEx(grayImgThresh, cv.MORPH_CLOSE, kernel)
# plt.imshow(closing, cmap="gray")

# Find the contours of the thresholded image; this is a crucial step in shape identification
gray_img_contours = cv.findContours(
    gray_img_thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
)[0]

# Create a new shape detector class, which will determine what shapes are present based on the contours
shape = corner_detect.ShapeDetector()


for index, con in enumerate(gray_img_contours):
    # Calculate the moments of the contours in the image, allowing for shape identification
    moments = cv.moments(con)

    conX = int((moments["m10"] / moments["m00"]))
    conY = int((moments["m01"] / moments["m00"]))
    shape_detect = shape.detectShape(con)

    con = con.astype("float")
    con = con.astype("int")
    cv.drawContours(src_img, [con], 0, (0, 255, 0), 2)

    if srcImgInvert == True:
        cv.putText(
            src_img,
            shape_detect,
            (conX, conY),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    else:
        cv.putText(
            src_img,
            shape_detect,
            (conX, conY),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    plt.figure()
    plt.imshow(src_img)

plt.show()
