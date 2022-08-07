# tests.py
import sys
from typing import Generic
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as numpy
from PIL import Image
from corners import detectCorners


""" Test for determine optimal input parameters of cornerHarris Edge detection.
@test_cornerHarris(srcImg, grayImg, paramOption, outputType)
srcImg - RGB color image
grayImg - grayscale image
paramOption - which parameter to vary during test: "blocksize" || "ksize" || "k"
outputType - returned figure; src image with markings or detector output:  "source" || "detector"
"""


def cornerHarris_test(srcImg, grayImg, paramOption, outputType):
    blocksizeList = [2, 3, 4, 5, 6, 7, 8]
    ksizeList = [3, 5, 7, 9]
    kList = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

    if paramOption == "blocksize":
        resultFigure = plt.figure("testing blocksize parameter", tight_layout=True)

        for i in range(len(blocksizeList)):
            # plt.subplot(int(ceil(len(blocksizeList) / 2)), 2, (i + 1))
            plt.subplot(2, 4, (i + 1))

            """
            detectorResponse (often ref. as dst) is a copy of the input (gray) image that reinterprets
            grayscale intensities [0, 255], as float32s, assigning 0.0 to "background" pixels and
            assigning very large values to pixels that corrispond e.g. 4.6e+08.
            This results in a image that rather useless by itself so comparitive thresholding is
            done to the origional image. Since the detectorResponse image is an exact size copy, of the
            source image, the detectorResponse img iterated through, if a pixel is above some threshold
            the corrisponding pixel on the source image is set to red i.e. [255, 0 , 0]
            """
            detectorResponse = cv.cornerHarris(
                grayImg, blocksizeList[i], ksizeList[0], kList[0]
            )

            # dilate just enhances the markers
            detectorResponse = cv.dilate(detectorResponse, None)  # type: ignore

            # thresholding
            srcImg[detectorResponse > 0.01 * detectorResponse.max()] = [255, 0, 0]

            if outputType == "source":
                plt.imshow(srcImg, cmap="gray")
            elif outputType == "detector":
                plt.imshow(detectorResponse, cmap="gray")

            plt.title("blocksize = " + str(blocksizeList[i]))

    elif paramOption == "ksize":
        resultFigure = plt.figure("testing ksize parameter", tight_layout=True)

        for i in range(len(ksizeList)):
            plt.subplot(1, 4, (i + 1))

            detectorResponse = cv.cornerHarris(
                grayImg, blocksizeList[0], ksizeList[i], kList[0]
            )

            # dilate just enhances the markers
            detectorResponse = cv.dilate(detectorResponse, None)  # type: ignore

            # thresholding
            srcImg[detectorResponse > 0.01 * detectorResponse.max()] = [255, 0, 0]

            # plt.imshow(detectorResponse, cmap="gray")
            if outputType == "source":
                plt.imshow(srcImg, cmap="gray")
            elif outputType == "detector":
                plt.imshow(detectorResponse, cmap="gray")

            plt.title("ksize = " + str(ksizeList[i]))

    elif paramOption == "k":
        resultFigure = plt.figure("testing k parameter", tight_layout=True)

        for i in range(len(kList)):
            plt.subplot(2, 4, (i + 1))

            detectorResponse = cv.cornerHarris(
                grayImg, blocksizeList[0], ksizeList[0], kList[i]
            )

            # dilate just enhances the markers
            detectorResponse = cv.dilate(detectorResponse, None)  # type: ignore

            # thresholding
            srcImg[detectorResponse > 0.01 * detectorResponse.max()] = [255, 0, 0]

            # plt.imshow(detectorResponse, cmap="gray")
            if outputType == "source":
                plt.imshow(srcImg, cmap="gray")
            elif outputType == "detector":
                plt.imshow(detectorResponse, cmap="gray")

            plt.title("k = " + str(kList[i]))
    else:
        raise NameError("invalid input for Corner Harris Test")

    return resultFigure


"""
Example usage:

...

srcImg = cv.imread("shapeshollow.png", cv.IMREAD_UNCHANGED)

if srcImg is None:
    sys.exit("Could not read shapeshallow.png")

grayImg = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)
grayImg = numpy.asarray(grayImg, dtype=numpy.float32)

fig_testing_blocksize = tests.cornerHarris_test(srcImg, grayImg, "blocksize", "source")
fig_testing_ksize = tests.cornerHarris_test(srcImg, grayImg, "ksize", "source")
fig_testing_k = tests.cornerHarris_test(srcImg, grayImg, "k", "detector")

plt.show()

"""


def matrixToImage(inputMatrix):
    """Takes matrix and returns a figure"""
    fig = plt.figure("Testing Matrix to Figure")
    width, height = inputMatrix.shape
    data = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    data[0 : height // 2, 0 : width // 4] = [255, 0, 0]  # red patch in upper left
    plt.imshow(data)
    return fig
