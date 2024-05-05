import cv2 as cv
import numpy as np
from math import *  # type: ignore
import random
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False


if __name__ == "__main__":
    img = cv.imread(r"img\alphabet.jpg", cv.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum0 = 20 * np.log(1 + np.abs(fshift))
    plt.figure(figsize=(10, 5))
    plt.subplot(151)
    plt.imshow(img, cmap="gray")
    plt.title("原图像")
    plt.axis("off")
    plt.subplot(152)
    plt.imshow(magnitude_spectrum0, cmap="gray")
    plt.title("原幅值谱")
    plt.axis("off")
    D0 = 20
    n = 4
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    H = np.zeros((rows, cols))
    for u in range(rows):
        for v in range(cols):
            D = sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            H[u, v] = 1 / (1 + (D / D0) ** (2 * n))
    G = H * fshift
    magnitude_spectrum1 = 20 * np.log(1 + np.abs(G))
    f1 = np.fft.ifftshift(G)
    img1 = np.abs(np.fft.ifft2(f1))
    plt.subplot(153)
    plt.imshow(H, cmap="gray")
    plt.title("巴特沃斯传递函数")
    plt.axis("off")
    plt.subplot(154)
    plt.imshow(magnitude_spectrum1, cmap="gray")
    plt.title("巴特沃斯滤波后的幅值谱")
    plt.axis("off")
    plt.subplot(155)
    plt.imshow(img1, cmap="gray")
    plt.title("重构图像")
    plt.axis("off")
    plt.show()
