import cv2 as cv
from matplotlib import pyplot as plt


def create_figure(subplots):
    """
    Creates a matplotlib figure with the given subplots and names.

    Args:
        subplots: List of subplots.
        names: List of names for the subplots.

    Returns:
        Matplotlib figure containing the subplots.
    """
    figure = plt.figure()

    for i, subplot in enumerate(subplots):
        ax = figure.add_subplot(*subplot[1])
        ax.imshow(subplot[0], cmap="gray")
        ax.set_title(subplot[2])
        ax.set_xticks([])
        ax.set_yticks([])

    return figure


src_img = cv.imread("images/Items2.png", cv.IMREAD_COLOR)
gray = cv.cvtColor(src_img, cv.COLOR_RGB2GRAY)
img = cv.GaussianBlur(gray, (3, 3), 0)
laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

subplots = [
    (img, (2, 2, 1), "Original"),
    (laplacian, (2, 2, 2), "Laplacian"),
    (sobelx, (2, 2, 3), "Sobel X"),
    (sobely, (2, 2, 4), "Sobel Y"),
]

figure = create_figure(subplots)
plt.show()
