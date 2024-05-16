import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun"]

img = cv.imread(r"img\light_circle.jpg", cv.IMREAD_GRAYSCALE)
kernalSize = 19
img_adapt = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, kernalSize, 6
)
img_tophat = cv.morphologyEx(
    img, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (45, 45))
)
plt.figure()
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("待处理灰度图像")
plt.axis("off")
plt.subplot(122)
plt.imshow(img_adapt, cmap="gray")
plt.title("自适应阈值分割结果")
plt.axis("off")
plt.tight_layout()
plt.show()

# # 交互式调整参数
# kernalSize = 7
# plt.ion()
# for i in range(10):
#     c = 4
#     for j in range(10):
#         plt.cla()
#         img_seg_adapt = cv.adaptiveThreshold(
#             img0, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, kernalSize, c
#         )
#         plt.imshow(img_seg_adapt, cmap="gray")
#         print(kernalSize, c)
#         plt.pause(0.01)
#         c += 2
#     kernalSize += 2
# plt.show()
