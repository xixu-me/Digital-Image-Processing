import cv2

lenna = cv2.imread(r"img\Lenna.png")
print(type(lenna))
cv2.namedWindow("Lena", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Lena", lenna)
cv2.waitKey(0)
cv2.destroyWindow("Lena")
cv2.imwrite(r"img\test_imwrite.png", lenna, (cv2.IMWRITE_PNG_COMPRESSION, 5))
