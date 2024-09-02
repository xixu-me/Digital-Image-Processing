import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimSun"]

img = cv.imread(r"img\paopao.jpg", cv.IMREAD_GRAYSCALE)
G1 = np.zeros(img.shape, np.uint8)
G2 = np.zeros(img.shape, np.uint8)
T1 = np.mean(img)  # type: ignore
diff = 255
T0 = 0.01
while diff > T0:
    _, G1 = cv.threshold(img, T1, 255, cv.THRESH_TOZERO_INV)
    _, G2 = cv.threshold(img, T1, 255, cv.THRESH_TOZERO)
    loc1 = np.where(G1 > 0.001)  # type: ignore
    loc2 = np.where(G2 > 0.001)  # type: ignore
    ave1 = np.mean(G1[loc1])  # type: ignore
    ave2 = np.mean(G2[loc2])  # type: ignore
    T2 = (ave1 + ave2) / 2
    diff = np.abs(T1 - T2)
    T1 = T2
_, img_result = cv.threshold(img, T1, 255, cv.THRESH_BINARY)
plt.figure()
plt.subplot(121)
plt.axis("off")
plt.imshow(img, cmap="gray")
plt.title("原灰度图像")
plt.subplot(122)
plt.axis("off")
plt.imshow(img_result, cmap="gray")
plt.title("迭代全阈值分割二值图像")
plt.tight_layout()
plt.show()
