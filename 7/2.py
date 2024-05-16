import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]


def histogram(img):
    row, col = img.shape
    hist = [0] * 256
    for i in range(row):
        for j in range(col):
            hist[img[i, j]] += 1
    return hist


if __name__ == "__main__":
    img = cv.imread(r"img\polygon_draw.jpg", cv.IMREAD_GRAYSCALE)
    img_noisy = np.uint8(img + 0.8 * img.std() * np.random.standard_normal(img.shape))
    img_noisy_blur = cv.GaussianBlur(img_noisy, (9, 9), 0)  # type: ignore
    _, img_result = cv.threshold(img_noisy, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # type: ignore
    _, img_result_blur = cv.threshold(
        img_noisy_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    plt.figure()
    plt.subplot(231)
    plt.axis("off")
    plt.imshow(img_noisy, cmap="gray")
    plt.title("带噪声的图像")
    plt.subplot(232)
    plt.xlabel("灰度值")
    plt.ylabel("像素个数")
    plt.bar(range(256), histogram(img_noisy))
    plt.title("噪声图像直方图")
    plt.subplot(233)
    plt.axis("off")
    plt.imshow(img_result, cmap="gray")
    plt.title("带噪声图像的 OTSU 分割")
    plt.subplot(234)
    plt.axis("off")
    plt.imshow(img_noisy_blur, cmap="gray")
    plt.title("高斯平滑的图像")
    plt.subplot(235)
    plt.xlabel("灰度值")
    plt.ylabel("像素个数")
    plt.bar(range(256), histogram(img_noisy_blur))
    plt.title("平滑图像直方图")
    plt.subplot(236)
    plt.axis("off")
    plt.imshow(img_result_blur, cmap="gray")
    plt.title("平滑图像的 OTSU 分割")
    plt.tight_layout()
    plt.show()
