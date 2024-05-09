import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]

img = cv.imread(r"img\alphabet.jpg", cv.IMREAD_GRAYSCALE)

dx = dy = 50
rows, cols = img.shape[:2]
M1 = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore
dst1 = cv.warpAffine(img, M1, (cols, rows))  # type: ignore

M2 = cv.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
dst2 = cv.warpAffine(img, M2, (cols, rows))

M3 = np.float32([[1, 0, 0], [0, -1, rows]])  # type: ignore
dst3 = cv.warpAffine(img, M3, (cols, rows))  # type: ignore

plt.figure()
plt.subplot(141)
plt.imshow(img, cmap="gray")
plt.title("待处理灰度图像")
plt.axis("off")
plt.subplot(142)
plt.imshow(dst1, cmap="gray")
plt.title("平移变换")
plt.axis("off")
plt.subplot(143)
plt.imshow(dst2, cmap="gray")
plt.title("旋转 30 度")
plt.axis("off")
plt.subplot(144)
plt.imshow(dst3, cmap="gray")
plt.title("垂直镜像")
plt.axis("off")
plt.tight_layout()
plt.show()
