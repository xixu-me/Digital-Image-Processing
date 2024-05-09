import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r"img\peppers.bmp", 0)
m, n = img.shape
if m % 2 == 0:
    m = m + 1
if n % 2 == 0:
    n = n + 1
img = cv.resize(img, (m, n))
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fmax = np.max(np.abs(fshift))
u0, v0 = (m - 1) // 2, (n - 1) // 2
u1 = u0 - 40
v1 = v0 + 40
fshift[v1, u1] = fmax / 5
u2 = m - 1 - u1
v2 = n - 1 - v1
fshift[v2, u2] = fmax / 5
f1 = np.fft.ifftshift(fshift)
img1 = abs(np.fft.ifft2(f1))
plt.imshow(img1, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
