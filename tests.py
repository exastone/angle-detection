# tests.py
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as numpy

"""
Test for determining the optimal input parameters of cornerHarris edge detection.
    
Args:
    srcImg: RGB color image.
    grayImg: Grayscale image.
    paramOption: Parameter to vary during the test: "blocksize" || "ksize" || "k".
    outputType: Returned figure; source image with markings or detector output: "source" || 
"detector".
    
Returns:
    Resulting figure.
"""


def cornerHarris_test(srcImg, grayImg, paramOption, outputType):
    list_blocksize = [2, 3, 4, 5, 6, 7, 8]
    list_ksize = [3, 5, 7, 9]
    list_kvalue = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

    if paramOption == "blocksize":
        resultFigure = plt.figure("testing blocksize parameter", tight_layout=True)

        for i in range(len(list_blocksize)):
            # plt.subplot(int(ceil(len(blocksizeList) / 2)), 2, (i + 1))
            plt.subplot(2, 4, (i + 1))

            """
            The `detectorResponse` (often referred to as `dst`) is a floating-point copy of the input grayscale image. It assigns a value of 0.0 to the "background" pixels and very large values to pixels that correspond to certain features. This results in an image that is not very informative by itself. To extract meaningful information, comparative thresholding is performed on the original image. Since the `detectorResponse` image has the same size as the source image, it is iterated through, and if a pixel exceeds a certain threshold, the corresponding pixel in the source image is set to the color red, represented as [255, 0, 0].
            """

            detectorResponse = cv.cornerHarris(
                grayImg, list_blocksize[i], list_ksize[0], list_kvalue[0]
            )

            # dilate just enhances the markers
            detectorResponse = cv.dilate(detectorResponse, None)  # type: ignore

            # thresholding
            srcImg[detectorResponse > 0.01 * detectorResponse.max()] = [255, 0, 0]

            if outputType == "source":
                plt.imshow(srcImg, cmap="gray")
            elif outputType == "detector":
                plt.imshow(detectorResponse, cmap="gray")

            plt.title("blocksize = " + str(list_blocksize[i]))

    elif paramOption == "ksize":
        resultFigure = plt.figure("testing ksize parameter", tight_layout=True)

        for i in range(len(list_ksize)):
            plt.subplot(1, 4, (i + 1))

            detectorResponse = cv.cornerHarris(
                grayImg, list_blocksize[0], list_ksize[i], list_kvalue[0]
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

            plt.title("ksize = " + str(list_ksize[i]))

    elif paramOption == "k":
        resultFigure = plt.figure("testing k parameter", tight_layout=True)

        for i in range(len(list_kvalue)):
            plt.subplot(2, 4, (i + 1))

            detectorResponse = cv.cornerHarris(
                grayImg, list_blocksize[0], list_ksize[0], list_kvalue[i]
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

            plt.title("k = " + str(list_kvalue[i]))
    else:
        raise NameError("invalid input for Corner Harris Test")

    return resultFigure


"""
Example usage:

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
