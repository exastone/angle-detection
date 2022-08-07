# corners.py
import cv2 as cv
import matplotlib.pylab as plt
import numpy as numpy
from PIL import Image


def detectCorners(srcImg, blockSize, kSize, k):
    # return cv.cornerHarris(img, blockSize, kSize, k)
    resultFigure = plt.figure("Harris Corner Detection", tight_layout=True)

    gray = cv.cvtColor(srcImg, cv.COLOR_RGB2GRAY)

    detectorResponce = cv.cornerHarris(gray, blockSize, kSize, k)

    for w in range(detectorResponce.shape[0]):
        for h in range(detectorResponce.shape[1]):
            if detectorResponce[w][h] > 0.04 * detectorResponce.max():
                # print(w, h, detectorResponce[w, h])
                # print(w, h, srcImg[w][h])
                srcImg[w][h] = [255, 0, 0]
                # srcImg[w][h] = [255, 0, 0, 255]

    # for w in range(detectorResponce.shape[0]):
    #     for h in range(detectorResponce.shape[1]):
    #         if detectorResponce[w][h] > 0.04 * detectorResponce.max():
    #             # print(w, h, detectorResponce[w, h])
    #             for subw in range(w - 2, w + 2):
    #                 for subh in range(h - 2, h + 2):
    #                     srcImg[subw][subh] = [255, 0, 0]

    plt.imshow(srcImg, cmap="gray")
    return resultFigure


class ShapeDetector:
    def __init__(self):
        pass

    def detectShape(self, c):
        shape = "none"
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.03 * peri, True)

        if len(approx) == 3:
            shape = "triangle"

        elif len(approx) == 4:

            (x, y, w, h) = cv.boundingRect(approx)
            ar = w / float(h)

            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        elif len(approx) == 5:
            shape = "pentagon"

        elif len(approx) == 6:
            shape = "hexagon"

        elif len(approx) == 7:
            shape = "heptagon"

        elif len(approx) == 8:
            shape = "octatagon"

        else:
            shape = "Circle"

        return shape
