import cv2 as cv
import numpy as np
from math import *  # type: ignore
import random
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def addGaussianNoise(src, means, sigma):
    NoiseImg = src / src.max()
    rows = NoiseImg.shape[0]
    cols = NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            NoiseImg[i, j] = NoiseImg[i, j] + random.gauss(means, sigma)
            if NoiseImg[i, j] < 0:
                NoiseImg[i, j] = 0
            elif NoiseImg[i, j] > 1:
                NoiseImg[i, j] = 1
    return NoiseImg


if __name__ == "__main__":
    img0 = cv.imread(r"img\peppers.bmp", cv.IMREAD_GRAYSCALE)
    img = addGaussianNoise(img0, 0, 0.1)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum0 = 20 * np.log(1 + np.abs(fshift))
    plt.figure(figsize=(10, 5))
    plt.subplot(141)
    plt.imshow(img, cmap="gray")
    plt.title("噪声图像")
    plt.axis("off")
    plt.subplot(142)
    plt.imshow(magnitude_spectrum0, cmap="gray")
    plt.title("噪声图像幅值谱")
    plt.axis("off")
    r = 50
    m, n = fshift.shape
    H = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = sqrt((i - m / 2) ** 2 + (j - n / 2) ** 2)
            if d < r:
                H[i, j] = 1
    G = H * fshift
    magnitude_spectrum1 = 20 * np.log(1 + np.abs(G))
    f1 = np.fft.ifftshift(G)
    img1 = np.abs(np.fft.ifft2(f1))
    plt.subplot(143)
    plt.imshow(magnitude_spectrum1, cmap="gray")
    plt.title("ILPF滤波后幅值谱")
    plt.axis("off")
    plt.subplot(144)
    plt.imshow(img1, cmap="gray")
    plt.title("ILPF滤波后重构图像")
    plt.axis("off")
    plt.show()
