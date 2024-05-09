import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimSun"]
src = cv.imread(r"img\peppers.bmp", cv.IMREAD_GRAYSCALE)

dft0 = cv.dft(np.float32(src), flags=cv.DFT_COMPLEX_OUTPUT)  # type: ignore
row, col = src.shape[:2]
src1 = np.zeros((row, col), dtype=np.float32)
for i in range(row):
    for j in range(col):
        src1[i, j] = src[i, j] * ((-1) ** (i + j))
dft1 = cv.dft(src1, flags=cv.DFT_COMPLEX_OUTPUT)
magnitude0 = cv.magnitude(dft1[:, :, 0], dft1[:, :, 1])
magnitude1 = 20 * np.log(1 + magnitude0)  # type: ignore
phase = cv.phase(dft1[:, :, 0], dft1[:, :, 1])
idft0 = cv.idft(dft0)
idft1 = cv.magnitude(idft0[:, :, 0], idft0[:, :, 1])

fft2 = np.fft.fft2(src)
fftshift = np.fft.fftshift(fft2)
abs0 = np.abs(fftshift)
abs1 = 20 * np.log(1 + abs0)
angle = np.angle(fftshift)
ifft20 = np.fft.ifft2(fftshift)
ifft21 = np.abs(ifft20)

plt.figure()
plt.subplot(2, 4, (1, 5))
plt.title("输入图像")
plt.imshow(src, cmap="gray")
plt.axis("off")
plt.subplot(2, 4, 2)
plt.title("幅值谱（OpenCV）")
plt.imshow(magnitude1, cmap="gray")
plt.axis("off")
plt.subplot(2, 4, 3)
plt.title("相位谱（OpenCV）")
plt.imshow(phase, cmap="gray")
plt.axis("off")
plt.subplot(2, 4, 4)
plt.title("重构图像（OpenCV）")
plt.imshow(idft1, cmap="gray")
plt.axis("off")
plt.subplot(2, 4, 6)
plt.title("幅值谱（NumPy）")
plt.imshow(abs1, cmap="gray")
plt.axis("off")
plt.subplot(2, 4, 7)
plt.title("相位谱（NumPy）")
plt.imshow(angle, cmap="gray")
plt.axis("off")
plt.subplot(2, 4, 8)
plt.title("重构图像（NumPy）")
plt.imshow(ifft21, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
