import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread(r"E:\OneDrive\Code\Python\2\iris.jpg", 0)
fil1 = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
fil2 = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
fil3 = 1 / 10 * np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
fil4 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
ImgSmoothed1 = cv.filter2D(img, -1, fil1, borderType=cv.BORDER_DEFAULT)
ImgSmoothed2 = cv.filter2D(img, -1, fil2, borderType=cv.BORDER_DEFAULT)
ImgSmoothed3 = cv.filter2D(img, -1, fil3, borderType=cv.BORDER_DEFAULT)
ImgSharp = cv.filter2D(img, -1, fil4, borderType=cv.BORDER_DEFAULT)

plt.figure()
plt.subplot(221)
plt.imshow(ImgSmoothed1, cmap="gray")
plt.title("smoothed1")
plt.subplot(222)
plt.imshow(ImgSmoothed2, cmap="gray")
plt.title("smoothed2")
plt.subplot(223)
plt.imshow(ImgSmoothed3, cmap="gray")
plt.title("smoothed3")
plt.subplot(224)
plt.imshow(ImgSharp, cmap="gray")
plt.title("sharp")
plt.show()
