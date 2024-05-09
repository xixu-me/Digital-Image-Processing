import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def gamma_trans(img, gamma=1.0):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv.LUT(img, gamma_table)


if __name__ == "__main__":
    im = cv.imread(r"img\iris.jpg", cv.IMREAD_GRAYSCALE)
    im1 = gamma_trans(im, 0.5)
    im2 = gamma_trans(im, 1.5)
    plt.figure()
    plt.subplot(131)
    plt.imshow(im, cmap="gray")
    plt.title("original")
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(im1, cmap="gray")
    plt.title("gamma = 0.5")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(im2, cmap="gray")
    plt.title("gamma = 1.5")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
