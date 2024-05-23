import cv2
import numpy as np
from math import *
import random
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

img0 = cv2.imread(r".\img\iris.jpg",0)
f = np.fft.fft2(img0)
fshift = np.fft.fftshift(f)
magnitude_spectrum0 = 20*np.log(1+np.abs(fshift))
plt.figure(figsize=(10,5))
plt.subplot(151),plt.imshow(img0, cmap = 'gray')  #显示加噪图像
plt.title('原图像'), plt.axis("off")
plt.subplot(152),plt.imshow(magnitude_spectrum0, cmap = 'gray')  #显示加噪图像
plt.title('原图像幅值谱')
plt.axis("off")
#进行理想低通滤波
r=50                 #截止频率的设置
[m,n]=fshift.shape
H=np.zeros((m,n),dtype=complex)
for i in range(m):
    for j in range(n):
        d=sqrt((i-m/2)**2+(j-n/2)**2)
        if d<r:
            # a = 2
            H[i,j]=1/(1+(d/50)**2)
G=H*fshift#理想低通滤波
BT=np.abs(H)
magnitude_spectrum1 =20*np.log(1+np.abs(G))  #理想低通滤波后的幅值谱
f1 = np.fft.ifftshift(G)
img1 =dst=np.fft.ifft2(f1 )  #重构图像
img1 =np.abs(img1)
plt.subplot(153),plt.imshow(BT, cmap = 'gray')  #显示重构图像
plt.title('巴特斯沃传递函数'), plt.axis("off")
plt.subplot(154),plt.imshow(magnitude_spectrum1, cmap = 'gray')  #显示滤波后幅值谱
plt.title('巴特斯沃滤波后幅值谱'), plt.axis("off")
plt.subplot(155),plt.imshow(img1, cmap = 'gray')  #显示重构图像
plt.title('巴特斯沃滤波后重构图像'), plt.axis("off")
plt.show()