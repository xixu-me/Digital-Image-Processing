import cv2
from PIL import Image
import numpy as np
import math
#平移
def move(img,dx=50,dy=50):
    rows,cols= img.shape[:2]
    M0 = np.float32([[1,0,dx],[0,1,dy]])
    dst0 = cv2.warpAffine(img,M0,(cols,rows))
    return dst0
# def move(img,dx=50,dy=50):
#     rows,cols= img.shape[:2]
#     M0 = np.float32([[1,0,dx],[0,1,dy],[0,0,1]])
#     dst0 = cv2.warpPerspective(img,M0,(cols,rows))
#     return dst0

#旋转
def rotate(img,angle=45):
    rows,cols=img.shape[:2]
    M1=cv2.getRotationMatrix2D((cols/2,rows/2),angle ,1)
    # 第三个参数是输出图像的尺寸
    dst1=cv2.warpAffine(img,M1,(cols,rows))
    return dst1
#镜像
#垂直镜像
def reflect_x(img):
    rows,cols=img.shape[:2]
    M2 = np.float32([[1,0,0],[0,-1,rows]])
    dst2 = cv2.warpAffine(img,M2,(cols,rows))
    return dst2
#水平镜像
def reflect_y(img):
    rows,cols=img.shape[:2]
    M3 = np.float32([[-1,0,cols],[0,1,0]])
    dst3 = cv2.warpAffine(img,M3,(cols,rows))
    return dst3
#缩放
def zoom(img,a,b):
    # 下面的 None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子
    # 因此这里为 None
    res=cv2.resize(img,None,fx=a,fy=b,interpolation=cv2.INTER_CUBIC)
    return res

#仿射变换
def affine(img,pts1,pts2):
    rows, cols = img.shape[:2]
    # pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    # pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst
#透视变换
def perspective(img,pts1,pts2):
    # pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    # pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300, 300))
    return dst
#最近邻插值
def NN_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=round((i+1)*(scrH/dstH))
            scry=round((j+1)*(scrW/dstW))
            retimg[i,j]=img[scrx-1,scry-1]
    return retimg
#双线性插值
def BiLinear_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH/dstH)-1
            scry=(j+1)*(scrW/dstW)-1
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    return retimg
#最佳插法函数
def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0
#三次内插法
def BiCubic_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=i*(scrH/dstH)
            scry=j*(scrW/dstW)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    return retimg


if __name__=="__main__":
    img = cv2.imread('Lenna.png')
    rows, cols, ch = img.shape
    cv2.imshow('img', img)
    img0 = move(img, 100, 50)
    img1 = rotate(img,45)
    img2 = reflect_x()
    img3 = reflect_y()
    img4 = zoom(img,2,2)
    img5 = affine(img)
    img6 = toushi()
    cv2.imshow('img0', img0)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.imshow('img4', img4)
    cv2.imshow('img5', img5)
    cv2.imshow('img6', img6)
    im_path = 'dog.png'
    image = np.array(Image.open(im_path))
    image1 = NN_interpolation(image, image.shape[0] * 2, image.shape[1] * 2)
    image2 = BiLinear_interpolation(image, image.shape[0] * 2, image.shape[1] * 2)
    image3 = BiCubic_interpolation(image, image.shape[0] * 2, image.shape[1] * 2)
    cv2.imshow('BiCubic_interpolation', image3)
    cv2.imshow('BiLinear_interpolation', image2)
    cv2.imshow('NN_interpolation', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()