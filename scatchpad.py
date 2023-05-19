# scatchpad.py
from fileinput import filename

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import corner_detect as corner_detect


src_img = "images/Items2.png"
# img_org = cv.imread(origfile, cv.IMREAD_COLOR)
img = mpimg.imread(src_img)
# imgplot = plt.imshow(img)


filename = "images/Items3r.png"
img_color = cv.imread(filename, cv.IMREAD_COLOR)


plt.figure("original image")
plt.imshow(img_color, cmap="gray")


img_gray = cv.cvtColor(img_color, cv.COLOR_RGB2GRAY)
plt.figure("gray image")
plt.imshow(img_gray, cmap="gray")


# [!] Smoothing
img_gray = cv.medianBlur(img_gray, 5)

# [!] Closing -> Dilation followed by Erosion
plt.figure("Morph")
kernel = np.ones((5, 5), dtype=img_gray.dtype)
img_gray = cv.morphologyEx(
    img_gray, cv.MORPH_CLOSE, kernel, borderType=cv.BORDER_DEFAULT
)
plt.imshow(img_gray, cmap="gray")

# [!] Threshold
img_gray = cv.threshold(img_gray, 60, 255, cv.THRESH_BINARY)[1]
plt.imshow(img_gray, cmap="gray")

# [!] Drawn contours over original image
img_copy = img.copy()
contours, hierarchy = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img_copy, contours, -1, (255, 0, 0, 255), 2)

plt.figure("Contours")
plt.imshow(img_copy)


# [!] Moments
# cnt = contours[0]
# M = cv.moments(cnt)
# cx = int(M["m10"] / M["m00"])
# cy = int(M["m01"] / M["m00"])
# area = cv.contourArea(cnt)
# perimeter = cv.arcLength(cnt, True)
# print(f"cx: {cx} \t cy: {cy}")

shape = corner_detect.ShapeDetector()

for idx, contour in enumerate(contours):
    M = cv.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    shape_detect = shape.detectShape(contour)

    cv.putText(
        img_copy,
        shape_detect,
        (cx, cy),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0, 255),
        2,
    )

plt.figure("shape detection")
plt.imshow(img_copy)


dst = cv.cornerHarris(img_gray, 3, 3, 0.04)
dst = cv.dilate(dst, None)
img[dst > 0.03 * dst.max()] = [255, 0, 0, 255]
plt.figure("Harris Edge Detection")
plt.imshow(img)

plt.show()
