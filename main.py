# main.py
import sys
import cv2 as cv
import numpy
from matplotlib import pyplot as plt

from tests import test_cornerHarris

srcImg = cv.imread("shapeshollow.png", cv.IMREAD_UNCHANGED)

if srcImg is None:
    sys.exit("Could not read ShapesHallow.png")

# plt.imshow(img, cmap="gray")
# plt.title("Original Image")

grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
grayImg = numpy.asarray(grayImg, dtype=numpy.float32)


fig_testing_blocksize = test_cornerHarris(srcImg, grayImg, "blocksize", "source")
fig_testing_ksize = test_cornerHarris(srcImg, grayImg, "ksize", "source")
fig_testing_k = test_cornerHarris(srcImg, grayImg, "k", "detector")

plt.show()
