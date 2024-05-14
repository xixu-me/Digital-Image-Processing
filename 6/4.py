import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]

img = cv.imread(r"img\mapleleaf.tif", cv.IMREAD_GRAYSCALE)
_, img_binary = cv.threshold(img, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
plt.figure()
plt.subplot(141)
plt.axis("off")
plt.imshow(img_binary, cmap="gray")
plt.title("原图像")
i = 1
for kernelSize in [3, 9, 15]:
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelSize, kernelSize))
    img_result = cv.morphologyEx(img_binary, cv.MORPH_GRADIENT, kernel)
    i += 1
    plt.subplot(140 + i)
    plt.axis("off")
    plt.imshow(img_result, cmap="gray")
    plt.title(f"{kernelSize}×{kernelSize}")
plt.tight_layout()
plt.show()
