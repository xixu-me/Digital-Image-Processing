# 阈值邻域平滑滤波和均值滤波
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
# from ImageAddNoise import *
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

# 添加高斯噪声
def addGaussianNoise(src,mu,sigma):
    NoiseImg=src.copy()
    NoiseImg=NoiseImg/NoiseImg.max()
    rows,cols=NoiseImg.shape[:2]
    for i in range(rows):
        for j in range(cols):
            #python里使用random.gauss函数加高斯噪声
            NoiseImg[i,j]=NoiseImg[i,j]+random.gauss(mu,sigma)
#             NoiseImg[i,j]=NoiseImg[i,j]+np.random.normal(mu,sigma)
            if  NoiseImg[i,j]< 0:
                 NoiseImg[i,j]=0
            elif  NoiseImg[i,j]>1:
                 NoiseImg[i,j]=1
    NoiseImg=np.uint8(NoiseImg*255)
    return NoiseImg

img = cv2.imread(r"..\img\train1.jpg",0)
row,col=img.shape
ImgGuassNoise = addGaussianNoise(img,0,0.1)    #添加0均值，0.2方差的高斯分布噪声
imgAver=cv2.blur(ImgGuassNoise,(5,5))
imgThresh=np.zeros((row,col))
T=20
for i in range(row):
    for j in range(col):
        if np.abs(ImgGuassNoise[i,j]-imgAver[i,j])>T:
            imgThresh[i,j]=imgAver[i,j]
        else:
            imgThresh[i,j]=ImgGuassNoise[i,j]

plt.figure(figsize=(10,6))
plt.subplot(221)
plt.imshow(img,cmap='gray')
plt.title("原图")
plt.axis('off')  #不显示坐标轴
plt.subplot(222)
plt.imshow(ImgGuassNoise,cmap='gray')
plt.title("加高斯噪声图像")
plt.axis('off')  #不显示坐标轴
plt.subplot(223)
plt.imshow(imgAver,cmap='gray')
plt.title("7x7均值滤波")
plt.axis('off')  #不显示坐标轴
plt.subplot(224)
plt.imshow(imgThresh,cmap='gray')
plt.title("阈值邻域平滑滤波")
plt.axis('off')  #不显示坐标轴
plt.show()
plt.savefig("ch03-29-thresh.jpg")