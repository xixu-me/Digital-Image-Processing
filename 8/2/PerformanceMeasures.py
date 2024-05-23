import math
import cv2
import numpy
import matplotlib.pyplot as plt

#自动将图像像素值转化成0-1的灰度图像并求简单的图像复原性能指标

#image0:滤波前图像 image1:滤波后图像 图像的长或宽不能为3个像素

def standardization(img): #将0-255范围的图像转化为0-1，将彩色图像转换为灰度图像
    if len(img.shape)>2 :#判断img.shape元组的长度
        [m, n, k] = img.shape
        if k!=1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else :              #img.shape长度为2，则是灰度图像
        [m ,n] = img.shape
    for i in range(m):
        for j in range(n):
            if(img[i,j]!=0):
                if(img[i,j]>1):
                    img = img / 255
                    break
    return img

def meanSquare(image0,image1):#求均方误差
    image0 = standardization(image0)
    image1 = standardization(image1)
    [m, n] = image0.shape
    MSE=0
    for i in range(m):
        for j in range(n):
            MSE=MSE+(image0[i,j]-image1[i,j])**2
    MSE=MSE/(m*n)
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

def PeakSignalNoiseRatio(image0,image1):#求峰值信噪比，输出单位：dB
    image0 = standardization(image0)
    image1 = standardization(image1)
    [m, n] = image0.shape
    MSE=0
    for i in range(m):
        for j in range(n):
            MSE = MSE + (image0[i, j] - image1[i, j]) ** 2
    MSE = MSE / (m * n)
    PSNR=10*math.log(1/(MSE+1e-14),10)  #防止除数为0
    return PSNR

def SimpleRPMPrint(image0,image1):#输出均方误差，信噪比，峰值信噪比
    MSE=meanSquare(image0,image1)
    SNR=SignalNoiseRatio(image0,image1)
    PSNR=PeakSignalNoiseRatio(image0,image1)
    print("均方误差为：",MSE)
    print("信噪比为：", SNR)
    print("峰值信噪比为：", PSNR)
    return

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
def PeakSignalNoiseRatio(image0,image1):#求峰值信噪比，输出单位：dB
    image0 = standardization(image0)
    image1 = standardization(image1)
    [m, n] = image0.shape
    MSE=0
    for i in range(m):
        for j in range(n):
            MSE = MSE + (image0[i, j] - image1[i, j]) ** 2
    MSE = MSE / (m * n)
    PSNR=10*math.log(1/(MSE+1e-14),10)  #防止除数为0
    return PSNR
# #以下是信噪比曲线，lp0理想低通，lp1巴特沃斯低通，lp2高斯低通，hp0理想高通，hp1巴特沃斯高通，hp2高斯高通
#     a1 = [0] * 80
#     a2 = [0] * 80
#     a3 = [0] * 80
#     a4 = [0] * 80
#     a5 = [0] * 80
#     a6 = [0] * 80
#     img1 = cv2.imread('c:/9.jpg', 0)
#     for i in range(20, 100):
#         a1[i - 20] = SignalNoiseRatio(img1, ifft(lpfilter(0, 375, 375, i, 1) * fft_mat))
#         a2[i - 20] = SignalNoiseRatio(img1, ifft(lpfilter(1, 375, 375, i, 1) * fft_mat))
#         a3[i - 20] = SignalNoiseRatio(img1, ifft(lpfilter(2, 375, 375, i, 1) * fft_mat))
#         a4[i - 20] = SignalNoiseRatio(img1, ifft(hpfilter(0, 375, 375, i, 1) * fft_mat))
#         a5[i - 20] = SignalNoiseRatio(img1, ifft(hpfilter(1, 375, 375, i, 1) * fft_mat))
#         a6[i - 20] = SignalNoiseRatio(img1, ifft(hpfilter(2, 375, 375, i, 1) * fft_mat))
#     x = np.linspace(20, 80, 80)
#     plt.figure(figsize=(10, 7))
#     plt.plot(x, a1, lw=2.0, ls='-', color='r', label='lp_0')
#     plt.plot(x, a2, lw=2.0, ls='-', color='b', label='lp_1')
#     plt.plot(x, a3, lw=2.0, ls='-', color='y', label='lp_2')
#     plt.plot(x, a4, lw=2.0, ls='-', color='k', label='hp_0')
#     plt.plot(x, a5, lw=2.0, ls='-', color='c', label='hp_1')
#     plt.plot(x, a6, lw=2.0, ls='-', color='g', label='hp_2')
#     plt.xlabel("d0", fontsize=12)
#     plt.ylabel("SNR", fontsize=12)
#     plt.legend(loc='upper right', fontsize=12)
#     plt.show()

if __name__=='__main__':
    img = cv2.imread(r'.\img\lena.bmp',0)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 对图像进行傅里叶变换
    # dft[:,:,0]为傅里叶变换的实部,dft[:,:,1]为傅里叶变换的虚部
    magnitude0 = 20 * np.log(1 + cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))  # 幅值谱
    phase0 = cv2.phase(dft[:, :, 0], dft[:, :, 1])  # 相位谱
    dft_shift = np.fft.fftshift(dft)
    magnitude1 = 20 * np.log(1 + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))  # 幅值谱
    phase1 = cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1])  # 相位谱
    plt.subplot(221),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(phase1, cmap = 'gray')
    plt.title('phase angle'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(magnitude0, cmap = 'gray')
    plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(magnitude1, cmap = 'gray')    #显示移中后的幅值谱
    plt.title('magnitude_spectrum after shift'), plt.xticks([]), plt.yticks([])
    plt.show()
