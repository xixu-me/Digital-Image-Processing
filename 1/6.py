import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
img_BGR = cv.imread(r"img\iris.jpg")
img_RGB = cv.cvtColor(img_BGR, cv.COLOR_BGR2RGB)
plt.imshow(img_RGB)
plt.show()
