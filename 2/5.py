import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    img = cv.imread(r"src.jpg", 0)
    lplc = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    lplcEnhance = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    ImgLplc = cv.filter2D(img, -1, lplc, borderType=cv.BORDER_DEFAULT)
    ImgLplcEnhance = cv.filter2D(img, -1, lplcEnhance, borderType=cv.BORDER_DEFAULT)

    plt.figure()
    plt.subplot(131)
    plt.imshow(img, cmap="gray")
    plt.title("original")
    plt.subplot(132)
    plt.imshow(ImgLplc, cmap="gray")
    plt.title("laplacian")
    plt.subplot(133)
    plt.imshow(ImgLplcEnhance, cmap="gray")
    plt.title("laplacian enhanced")
    plt.show()
