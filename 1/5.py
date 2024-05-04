import cv2 as cv

lenna = cv.imread(r"img\iris.jpg")
print(type(lenna))
cv.namedWindow("Lena", cv.WINDOW_AUTOSIZE)
cv.imshow("Lena", lenna)
cv.waitKey(0)
cv.destroyWindow("Lena")
cv.imwrite(r"1\imwrite.png", lenna, (cv.IMWRITE_PNG_COMPRESSION, 5))
