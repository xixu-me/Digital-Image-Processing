import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]

img0 = cv.imread(r"img\alphabet.jpg", cv.IMREAD_GRAYSCALE)
img1 = cv.copyMakeBorder(img0, 50, 50, 50, 50, cv.BORDER_CONSTANT, value=0)  # type: ignore
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # type: ignore
pts2 = np.float32([[10, 70], [160, 50], [40, 220]])  # type: ignore
rows, cols = img1.shape[:2]
M1 = cv.getAffineTransform(pts1, pts2)  # type: ignore
dst1 = cv.warpAffine(img1, M1, (cols, rows))
M2 = cv.getAffineTransform(pts2, pts1)  # type: ignore
dst2 = cv.warpAffine(dst1, M2, (cols, rows))

plt.figure()
plt.subplot(131)
plt.imshow(img1, cmap="gray")
plt.title("扩展后的图像")
plt.axis("off")
plt.subplot(132)
plt.imshow(dst1, cmap="gray")
plt.title("仿射变换图像")
plt.axis("off")
plt.subplot(133)
plt.imshow(dst2, cmap="gray")
plt.title("放射校正图像")
plt.axis("off")
plt.tight_layout()
plt.show()
