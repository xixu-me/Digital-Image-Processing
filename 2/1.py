import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def global_linear_transmation(im, c=0, d=255):
    img = im.copy()
    maxV = img.max()
    minV = img.min()
    if maxV == minV:
        return np.uint8(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = ((d - c) / (maxV - minV)) * (img[i, j] - minV) + c
    return np.uint8(img)


def histogram_equalization(im):
    return np.uint8(cv.equalizeHist(im))


if __name__ == "__main__":
    im = cv.imread(r"img\iris.jpg", cv.IMREAD_GRAYSCALE)
    im1 = global_linear_transmation(im, 0, 150)
    im2 = global_linear_transmation(im, 100)
    im3 = global_linear_transmation(im, 50, 150)
    im4 = histogram_equalization(im)
    plt.figure()
    plt.subplot(241)
    plt.imshow(im1, cmap="gray")
    plt.title("darker")
    plt.axis("off")
    plt.subplot(242)
    plt.imshow(im2, cmap="gray")
    plt.title("brighter")
    plt.axis("off")
    plt.subplot(243)
    plt.imshow(im3, cmap="gray")
    plt.title("lower contrast")
    plt.axis("off")
    plt.subplot(244)
    plt.imshow(im4, cmap="gray")
    plt.title("equalized")
    plt.axis("off")
    plt.subplot(245)
    plt.hist(im1.flatten(), 256, [0, 256])  # type: ignore
    plt.subplot(246)
    plt.hist(im2.flatten(), 256, [0, 256])  # type: ignore
    plt.subplot(247)
    plt.hist(im3.flatten(), 256, [0, 256])  # type: ignore
    plt.subplot(248)
    plt.hist(im4.flatten(), 256, [0, 256])  # type: ignore
    plt.show()
