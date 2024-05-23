import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import sys
import cv2
import random
import Fourier

#此函数生成的运动模糊核是由旋转来控制运动方向
def motion_PSF(kernel_size=15, angle=60):
    PSF = np.diag(np.ones(kernel_size))# 初始模糊核的方向是-45度
    angle = angle + 45  # 抵消-45度的影响
    M =cv2.getRotationMatrix2D((kernel_size/2,kernel_size/2), angle, 1) # 生成旋转矩阵
    PSF = cv2.warpAffine(PSF, M, (kernel_size, kernel_size), flags=cv2.INTER_NEAREST)
    PSF = PSF / PSF.sum()    #模糊核的权重和为1
    return PSF

#生成高斯模糊核
def Gaussian_PSF(kernel_size=15,sigma=0.1):  #生成高斯模糊核
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))

#生成大气湍流模糊核
def turbulence_PSF(input,k):
    [m, n] = input.shape
    PSF = np.zeros((m, n))
    p = m / 2
    q = n / 2
    for u in range(m):
        for v in range(n):
            PSF[u, v]=math.exp(-k*((u-p)*(u-p)+(v-q)*(v-q))**(5/6))
    PSF1= fft.ifft2(PSF)
    PSF1 = np.abs(PSF1)
    PSF1 = PSF1 / PSF1.sum()  # 模糊核的权重和为1
    return PSF1

#此函数扩展PSF0，使之与image0一样大小
def extension_PSF(image0,PSF0):
    [img_h,img_w] = image0.shape
    [h,w] = PSF0.shape
    PSF=np.zeros((img_h,img_w))
    PSF[0:h, 0:w] =PSF0[0:h, 0:w]
    return PSF

# 在频域对图片进行运动模糊
def make_blurred(input, PSF, eps=0.01):
    input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred=np.abs(blurred)
    return blurred

def inverse(input, PSF, eps=0.01):  # 逆滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) +eps
    Output_fft =input_fft/ PSF_fft  #在频域进行逆滤波
    result =fft.ifft2(Output_fft) # 计算F(u,v)的傅里叶反变换
    result = np.abs(result)
    return result
# def improved_inverse(input, PSF,w=70, k=0.1, eps=0.01):  # 逆滤波
#     input_fft = fft.fft2(input)
#     input_fftshift=fft.fftshift(input_fft)
#     PSF_fft = fft.fft2(PSF)
#     PSF_fftshift=fft.fftshift(PSF_fft)+eps
#     rows,cols = input_fftshift.shape[:2]
#     duv = Fourier.fft_distances(rows,cols)
#     for u in range(rows):
#         for v in range(cols):
#             if duv[u,v]<w:
#                 PSF_fftshift[u,v]=1/PSF_fftshift[u,v]
#             else:
#                 PSF_fftshift[u,v]=k
#     # PSF_fftshift=1/PSF_fftshift
#     output_fftshift =input_fftshift* PSF_fftshift  #在频域进行逆滤波
#     output_fft=fft.ifftshift(output_fftshift)
#     result =fft.ifft2(output_fft) # 计算F(u,v)的傅里叶反变换
#     result = np.abs(result)
#     return result

def improved_inverse(input, PSF, w=70, k=0.1, eps=0.01):
    input_fft = fft.fft2(input)
    input_fftshift=fft.fftshift(input_fft)
    PSF_fft = fft.fft2(PSF)
    PSF_fftshift=fft.fftshift(PSF_fft)+eps
    rows,cols = input_fftshift.shape[:2]
    duv = Fourier.fft_distances(rows,cols)
    for u in range(rows):
        for v in range(cols):
            if duv[u,v]<w:
                PSF_fftshift[u,v]=1/PSF_fftshift[u,v]
            else:
                PSF_fftshift[u,v]=k
    # PSF_fftshift=1/PSF_fftshift
    output_fftshift = input_fftshift*PSF_fftshift
    output_fft=fft.ifftshift(output_fftshift)
    result = fft.ifft2(output_fft)
    result = np.abs(result)
    return result

def wiener(input, PSF, K=0.1, eps=0.01):  #维纳滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft)
    result = np.abs(result)
    return result

def constrained_least_squares(input, PSF, r, eps=0.01):  #最小二乘滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    Q=np.array([[ 0,-1,0],                        #这个是设置的滤波，也就是卷积核
                [ -1,4,-1],
                [ 0,-1,0]])
    Q= extension_PSF(input, Q)
    Q_fft = fft.fft2(Q)
    PSF_fft = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + r * np.abs(Q_fft) ** 2)
    result = fft.ifft2(input_fft * PSF_fft)
    result=np.abs(result)
    return result

def standardization(img,L=255): #将0-255范围的图像转化为0-1，将彩色图像转换为灰度图像
    if len(img.shape)>2 :#判断img.shape元组的长度
        if img.shape[2]!=1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / L
    return img

def meanSquare(image0,image1,L=255):#求均方误差
    image0 = standardization(image0)
    image1 = standardization(image1)
    [m, n] = image0.shape
    MSE=0
    for i in range(m):
        for j in range(n):
            MSE=MSE+(image0[i,j]-image1[i,j])**2
    MSE=MSE/(m*n)*(L**2)
    return MSE

def SignalNoiseRatio(image0,image1):#求信噪比，输出单位：dB
    image0 = standardization(image0)
    image1 = standardization(image1)
    [m, n] = image0.shape
    Np=0
    Sp=0
    for i in range(m):
        for j in range(n):
            Np=Np+(image0[i,j]-image1[i,j])**2
            Sp=Sp+image1[i,j]**2
    SNR=10*math.log(Sp/(Np+1e-14),10)   #防止除数为0
    return SNR

def PeakSignalNoiseRatio(image0,image1,L=255):#求峰值信噪比，输出单位：dB
    MSE=meanSquare(image0,image1)
    PSNR=10*math.log(L**2/(MSE+1e-14),10)  #防止除数为0
    return PSNR


if __name__=="__main__":   #主程序判断语句
    image = cv2.imread('2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure('运动模糊')
    plt.subplot(231)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.gray()
    plt.imshow(image)  # 显示原图像
    PSF = motion_PSF()   # 生成运动模糊核
    PSF = extension_PSF(image, PSF)   # 在频域进行运动模糊，需要扩展PSF，使其与图像一样大小
    blurred = make_blurred(image, PSF)
    blurred = blurred + 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    plt.subplot(232)
    plt.imshow(blurred)
    plt.title('Motion blurred'), plt.xticks([]), plt.yticks([])
    result1 = inverse(blurred, PSF) # 逆滤波
    plt.subplot(233)
    plt.imshow(result1)
    plt.title('inverse '), plt.xticks([]), plt.yticks([])
    result2 = improving_inverse(blurred, PSF, 30, 1)  # 改进的逆滤波
    plt.subplot(234)
    plt.title('improving_inverse '), plt.xticks([]), plt.yticks([])
    plt.imshow(result2)
    result3 = wiener(blurred, PSF, 0.01)  # 对添加噪声的图像进行维纳滤波
    plt.subplot(235)
    plt.title('wiener '), plt.xticks([]), plt.yticks([])
    plt.imshow(result3)
    result4 = constrained_least_squares(blurred, PSF, 0.002)
    plt.subplot(236)
    plt.title('constrained_least_squares'), plt.xticks([]), plt.yticks([])
    plt.imshow(result4)

    plt.figure('大气湍流模糊')
    plt.subplot(231)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.gray()
    plt.imshow(image)  # 显示原图像
    PSF1 = daqi_PSF(image, 0.0025)
    PSF1 = extension_PSF(image,PSF1)
    da_blurred = make_blurred(image, PSF1)
    da_blurred = da_blurred + 0.1 * da_blurred.std() * np.random.standard_normal(da_blurred.shape)
    plt.subplot(232)
    plt.title('daqi_blurred'), plt.xticks([]), plt.yticks([])
    plt.imshow(da_blurred)
    result5=inverse(da_blurred, PSF1)
    plt.subplot(233)
    plt.title('inverse'), plt.xticks([]), plt.yticks([])
    plt.imshow(result5)
    result6 = improving_inverse(da_blurred, PSF1, 30, 1)  # 改进的逆滤波
    plt.subplot(234)
    plt.title('improving_inverse'), plt.xticks([]), plt.yticks([])
    plt.imshow(result6)
    result7 = wiener(da_blurred, PSF1, 0.01)  # 对添加噪声的图像进行维纳滤波
    plt.subplot(235)
    plt.title('wiener'), plt.xticks([]), plt.yticks([])
    plt.imshow(result7)
    result8 = constrained_least_squares(da_blurred, PSF1, 0.002)
    plt.subplot(236)
    plt.title('constrained_least_squares'), plt.xticks([]), plt.yticks([])
    plt.imshow(result8)

    plt.figure('高斯模糊')
    plt.subplot(231)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.gray()
    plt.imshow(image)  # 显示原图像
    G_PSF = G_PSF(9,10)
    G_PSF = extension_PSF(image, G_PSF)
    G_blurred = make_blurred(image, G_PSF)
    G_blurred = G_blurred + 0.1 * G_blurred.std() * np.random.standard_normal(G_blurred.shape)
    plt.subplot(232)
    plt.imshow(G_blurred)
    plt.title('Gaussian blurred'), plt.xticks([]), plt.yticks([])
    result9 = inverse(G_blurred, G_PSF)
    plt.subplot(233)
    plt.title('inverse'), plt.xticks([]), plt.yticks([])
    plt.imshow(result9)
    result10 = improving_inverse(G_blurred, G_PSF, 30, 1)  # 改进的逆滤波
    plt.subplot(234)
    plt.title('improving_inverse'), plt.xticks([]), plt.yticks([])
    plt.imshow(result10)
    result11 = wiener(G_blurred, G_PSF, 0.01)  # 对添加噪声的图像进行维纳滤波
    plt.subplot(235)
    plt.title('wiener'), plt.xticks([]), plt.yticks([])
    plt.imshow(result11)
    result12 = constrained_least_squares(G_blurred, G_PSF, 0.0025)
    plt.subplot(236)
    plt.title('constrained_least_squares'), plt.xticks([]), plt.yticks([])
    plt.imshow(result12)
    plt.show()
'''
    x = np.arange(0, 0.0015, 0.00001)
    length = len(x)
    y = np.zeros(length)
    for i in range(length):
        result = wiener(blurred, PSF, x[i])
        y[i] = meanSquare(image, result, L=255)
    plt.figure('运动模糊维纳滤波均方误差')
    plt.plot(x, y, color='green', label='meanSquare')
    plt.legend()

    x = np.arange(0, 0.0015, 0.00001)
    length = len(x)
    y = np.zeros(length)
    for i in range(length):
        result = wiener(blurred, PSF, x[i])
        y[i] = SignalNoiseRatio(image, result)
    plt.figure('运动模糊维纳滤波信噪比')
    plt.plot(x, y, color='red', label='SignalNoiseRatio')
    plt.legend()

    x = np.arange(0, 0.0015, 0.00001)
    length = len(x)
    y = np.zeros(length)
    for i in range(length):
        result = wiener(blurred, PSF, x[i])
        y[i] = PeakSignalNoiseRatio(image, result, L=255)
    plt.figure('运动模糊维纳滤波峰值信噪比')
    plt.plot(x, y, color='blue', label='PeakSignalNoiseRatio')
    plt.legend()

    x = np.arange(0.001, 0.02, 0.0001)
    length = len(x)
    y = np.zeros(length)
    for i in range(length):
        result = constrained_least_squares(image, PSF, x[i])
        y[i] = meanSquare(image, result, L=255)
    plt.figure('运动模糊约束最小二乘滤波均方误差')
    plt.plot(x, y, color='green', label='constrained_least_squares')
    plt.legend()

    x = np.arange(0, 0.02, 0.0001)
    length = len(x)
    y = np.zeros(length)
    for i in range(length):
        result = constrained_least_squares(image, PSF, x[i])
        y[i] = SignalNoiseRatio(image, result)
    plt.figure('运动模糊约束最小二乘滤波信噪比')
    plt.plot(x, y, color='red', label='SignalNoiseRatio')
    plt.legend()

    x = np.arange(0.001, 0.02, 0.0001)
    length = len(x)
    y = np.zeros(length)
    for i in range(length):
        result = constrained_least_squares(image, PSF, x[i])
        y[i] = PeakSignalNoiseRatio(image, result, L=255)
    plt.figure('运动模糊约束最小二乘滤波峰值信噪比')
    plt.plot(x, y, color='blue', label='PeakSignalNoiseRatio')
    plt.legend()

    plt.show()
'''




