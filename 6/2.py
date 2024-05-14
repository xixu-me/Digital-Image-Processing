import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]

img = cv.imread(r"6\2.png", cv.IMREAD_GRAYSCALE)
_, img_binary = cv.threshold(img, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
kernelSize = 40
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernelSize, kernelSize))
img_open = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel)
img_close = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)
plt.figure()
plt.subplot(131)
plt.axis("off")
plt.imshow(img_binary, cmap="gray")
plt.title("待处理的原图像")
plt.subplot(132)
plt.axis("off")
plt.imshow(img_open, cmap="gray")
plt.title("开运算的结果")
plt.subplot(133)
plt.axis("off")
plt.imshow(img_close, cmap="gray")
plt.title("闭运算的结果")
plt.tight_layout()
plt.show()
