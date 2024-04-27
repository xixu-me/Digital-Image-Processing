import cv2

if __name__ == "__main__":
    lenna = cv2.imread(r"src.jpg")
    print(type(lenna))
    cv2.namedWindow("Lena", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Lena", lenna)
    cv2.waitKey(0)
    cv2.destroyWindow("Lena")
    cv2.imwrite(r"1\imwrite.png", lenna, (cv2.IMWRITE_PNG_COMPRESSION, 5))
