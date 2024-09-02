import random as rd

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def addSaltAndPepper(src, percentage):
    NoiseImg = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = rd.randint(0, src.shape[0] - 1)
        randY = rd.randint(0, src.shape[1] - 1)
        if rd.randint(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def addGaussianNoise(src, means, sigma):
    NoiseImg = src / src.max()
    rows = NoiseImg.shape[0]
    cols = NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            NoiseImg[i, j] = NoiseImg[i, j] + rd.gauss(means, sigma)
            if NoiseImg[i, j] < 0:
                NoiseImg[i, j] = 0
            if NoiseImg[i, j] > 1:
                NoiseImg[i, j] = 1
    NoiseImg = np.uint8(NoiseImg * 255)
    return NoiseImg


if __name__ == "__main__":
    im = cv.imread(r"img\iris.jpg", cv.IMREAD_GRAYSCALE)
    im1 = addSaltAndPepper(im, 0.1)
    im11 = cv.blur(im1, (3, 3))
    im12 = cv.medianBlur(im1, 3)
    im13 = cv.GaussianBlur(im1, (3, 3), 1)
    im2 = addGaussianNoise(im, 0, 0.1)
    im21 = cv.blur(im2, (3, 3))  # type: ignore
    im22 = cv.medianBlur(im2, 3)  # type: ignore
    im23 = cv.GaussianBlur(im2, (3, 3), 1)  # type: ignore
    plt.figure()
    plt.subplot(241)
    plt.imshow(im1, cmap="gray")
    plt.title("salt and pepper")
    plt.axis("off")
    plt.subplot(242)
    plt.imshow(im11, cmap="gray")
    plt.title("blur")
    plt.axis("off")
    plt.subplot(243)
    plt.imshow(im12, cmap="gray")
    plt.title("median")
    plt.axis("off")
    plt.subplot(244)
    plt.imshow(im13, cmap="gray")
    plt.title("gaussian")
    plt.axis("off")
    plt.subplot(245)
    plt.imshow(im2, cmap="gray")
    plt.title("gaussian noise")
    plt.axis("off")
    plt.subplot(246)
    plt.imshow(im21, cmap="gray")
    plt.title("blur")
    plt.axis("off")
    plt.subplot(247)
    plt.imshow(im22, cmap="gray")
    plt.title("median")
    plt.axis("off")
    plt.subplot(248)
    plt.imshow(im23, cmap="gray")
    plt.title("gaussian")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
