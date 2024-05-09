import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]

img0 = cv.imread(r"img\alphabet.jpg")
imgT = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
cv.rectangle(img0, (160, 140), (190, 170), (255, 0, 0), 3)
img = imgT[140:170, 160:190]
img1 = cv.resize(img, None, fx=10, fy=10, interpolation=cv.INTER_NEAREST)
img2 = cv.resize(img, None, fx=10, fy=10, interpolation=cv.INTER_LINEAR)
img3 = cv.resize(img, None, fx=10, fy=10, interpolation=cv.INTER_CUBIC)

plt.figure()
plt.subplot(141)
plt.imshow(img0, cmap="gray")
plt.title("原图像")
plt.axis("off")
plt.subplot(142)
plt.imshow(img1, cmap="gray")
plt.title("最近邻插值")
plt.axis("off")
plt.subplot(143)
plt.imshow(img2, cmap="gray")
plt.title("双线性插值")
plt.axis("off")
plt.subplot(144)
plt.imshow(img3, cmap="gray")
plt.title("双三次插值")
plt.axis("off")
plt.tight_layout()
plt.show()
