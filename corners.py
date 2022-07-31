# corners.py
import cv2 as cv
import numpy

""" 
    img - Input image. It should be grayscale and float32 type.
    blockSize - It is the size of neighbourhood considered for corner detection
    ksize - Aperture parameter of the Sobel derivative used.
    k - Harris detector free parameter in the equation.
"""


def detectCorners(img, blockSize, kSize, k):
    return cv.cornerHarris(img, blockSize, kSize, k)
