import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
import skimage.filters.rank as sfr
from scipy.signal import convolve2d
import gc


# 创建图像
def create_img():
    box = np.zeros((300, 300), np.uint8)+10
    box[30:270,30:270]=128
    shape = box.shape
    center=(150,150)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if De(i,j,center)<70:
                box[i,j]=200
    return box

# 计算欧式距离
def De(x,y,center):
    return math.sqrt((x-center[0])**2+(y-center[1])**2)

#----添加噪声--------------------------------------------------
'''def add_gaussian_noise(image_in, u=0,noise_sigma=0.01):
    """
    给图片添加高斯噪声
    image_in:输入图片
    noise_sigma：
    """
    src = np.float64(np.copy(image_in))/255
    if len(src.shape) == 2:
        h, w = src.shape
    else:
        h, w, _ = src.shape
    # 标准正态分布*noise_sigma
    noise = u + np.random.randn(h, w) * noise_sigma  #产生高斯分布的噪声
    NoiseImg = np.zeros(src.shape, np.float64)
    if len(NoiseImg.shape) == 2:
        NoiseImg = src+noise
    else:
        NoiseImg[:,:, 0] = src[:,:, 0]+noise
        NoiseImg[:,:, 1] = src[:,:, 1]+noise
        NoiseImg[:,:, 2] = src[:,:, 2]+noise
    return np.uint8(NoiseImg*255)

def add_salt_and_pepper_noise(src, percentage=0.1):
    NoiseImg = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if len(NoiseImg.shape) == 2:
            if random.randint(0, 1) == 0:
                NoiseImg[randX, randY] = 0
            else:
                NoiseImg[randX, randY] = 255
        else:
            if random.randint(0, 1) == 0:
                NoiseImg[randX, randY, 0] = 0
                NoiseImg[randX, randY, 1] = 0
                NoiseImg[randX, randY, 2] = 0
            else:
                NoiseImg[randX, randY, 0] = 255
                NoiseImg[randX, randY, 1] = 255
                NoiseImg[randX, randY, 2] = 255
    # plt.imshow(NoiseImg, cmap='gray')
    # plt.title(title)
    # plt.show()
    return NoiseImg

def add_mean_noise(img, a=100, b=150, percentage=1):
    NoiseImg = img.copy()
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if len(NoiseImg.shape) == 2:
            NoiseImg[randX, randY] = np.random.randint(a, b)
        else:
            NoiseImg[randX, randY, 0] = np.random.randint(a, b)
            NoiseImg[randX, randY, 1] = np.random.randint(a, b)
            NoiseImg[randX, randY, 2] = np.random.randint(a, b)
    # plt.imshow(NoiseImg,cmap='gray')
    # plt.title('add_mean_noise')
    # plt.show()
    return NoiseImg

def add_rayleigh_noise(img, a=100, b=150, percentage=1):
    NoiseImg = img.copy()
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if len(NoiseImg.shape) == 2:
            NoiseImg[randX, randY] = a + (-b * np.log(1 - np.random.rand())) ** 0.5 #由均值分布产生瑞利分布
        else:
            NoiseImg[randX, randY, 0] = a + (-b * np.log(1 - np.random.rand())) ** 0.5
            NoiseImg[randX, randY, 1] = a + (-b * np.log(1 - np.random.rand())) ** 0.5
            NoiseImg[randX, randY, 2] = a + (-b * np.log(1 - np.random.rand())) ** 0.5
            # plt.imshow(NoiseImg, cmap='gray')
    # plt.title('add_rayleigh_noise')
    # plt.show()
    return NoiseImg

def add_erlang_noise(img, a=100, b=0.1,percentage=1):
    NoiseImg = img.copy()
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if len(NoiseImg.shape) == 2:
            NoiseImg[randX, randY] = (- np.log(1 - np.random.rand()) / b) +a
        else:
            NoiseImg[randX, randY, 0] = - np.log(1 - np.random.rand()) / b +a
            NoiseImg[randX, randY, 1] = - np.log(1 - np.random.rand()) / b +a
            NoiseImg[randX, randY, 2] = - np.log(1 - np.random.rand()) / b +a

    # plt.imshow(NoiseImg, cmap='gray')
    # plt.title('add_erlang_noise')
    # plt.show()
    return NoiseImg
'''
#####################################
###############滤波模块##############
#####################################

#高斯滤波函数
def gaussian_filter(src,kernalSize=5):
    img_Guassian = cv2.GaussianBlur(src, (kernalSize, kernalSize), 0)
    return img_Guassian
#中值滤波函数
def median_filter(src,kernalSize=5):
    img_median = cv2.medianBlur(src, kernalSize)
    return img_median

#最大滤波函数
def max_filter(src,kernalSize=5):
    img_Max = sfr.maximum(src, disk(kernalSize))
    return img_Max
#最小滤波函数
def min_filter(src,kernalSize=5):
    img_Min = sfr.minimum(src, disk(kernalSize))
    return img_Min
#使用opencv的卷积函数
def smooth_filter(src,fil,type=cv2.BORDER_DEFAULT):
    Smoothed = cv2.filter2D(src, -1, fil, borderType=type)
    return Smoothed
#####################################
#############边缘填充模块############
#####################################

#比较四种不同的边缘填充方式，并输出其高斯滤波后的性能指标和图像
def border_fill(src,type=cv2.BORDER_DEFAULT,size=10):
    borderImg = cv2.copyMakeBorder(src, size, size, size, size, borderType=type)
    return borderImg
'''def border_fill(src):
    # 使用opencv的卷积函数
    CONSTANT= cv2.copyMakeBorder(src,10,10,10,10,borderType = cv2.BORDER_CONSTANT)#补零0cv2.BORDER_REFLECT2
    REPLICATE= cv2.copyMakeBorder(src,10,10,10,10,borderType = cv2.BORDER_REPLICATE)#复制1
    DEFAULT= cv2.copyMakeBorder(src,10,10,10,10,borderType = cv2.BORDER_DEFAULT)#对称4
    WRAP = cv2.copyMakeBorder(src,10,10,10,10,borderType =cv2.BORDER_WRAP)#翻转3

    constant = gaussian_filter(CONSTANT)
    replicate= gaussian_filter(REPLICATE)
    default= gaussian_filter(DEFAULT)
    wrap= gaussian_filter(WRAP)

    # 打印输出重构后图像的均方误差
    # print("CONSTANT：")
    # RPMPrint(src, constant)
    # print("REPLICATE：")
    # RPMPrint(src, replicate)
    # print("DEFAULT：")
    # RPMPrint(src, default)
    # print("WRAP：")
    # RPMPrint(src, wrap)

    plt.figure(0)
    plt.subplot(221),plt.imshow(constant,'gray'),plt.title('constant image')
    plt.subplot(222),plt.imshow(replicate,'gray'),plt.title('replicate image ')
    plt.subplot(223),plt.imshow(default,'gray'),plt.title('default image ')
    plt.subplot(224),plt.imshow(wrap,'gray'),plt.title('wrap image ')
    plt.show()'''

#####################################
#############锐化算子模块############
#####################################

#Roberts锐化
'''def RobertsOperator(roi):#锐化算子
    operator_first = np.array([[-1, 0], [0, 1]])
    operator_second = np.array([[0, 1], [1, 0]])
    return np.abs(np.sum(roi[1:,1:]*operator_first))+np.abs(np.sum(roi[1:,1:]*operator_second))

def roberts_filter(image):#Roberts锐化
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]):
        for j in range(1,image.shape[1]):
            image[i,j] = RobertsOperator(image[i-1:i+2,j-1:j+2])
    return image[1:image.shape[0],1:image.shape[1]]'''
def roberts_filter(grayImage):#Roberts锐化
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts

# Sobel operator
def sobel_filter(src):
    x = cv2.Sobel(src, cv2.CV_16S, 1, 0)  # 对x求导
    y = cv2.Sobel(src, cv2.CV_16S, 0, 1)  # 对y求导
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst
def scharr_filter(src):
    x = cv2.Scharr(src, cv2.CV_32F, 1, 0) #X方向
    y = cv2.Scharr(src, cv2.CV_32F, 0, 1) #Y方向
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst
# Prewitt算子
def prewitt_filter(src):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(src, cv2.CV_16S, kernelx)
    y = cv2.filter2D(src, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


# Laplacian 锐化
def laplacian_filter(src,kernalType=1):
    # 为了让结果更清晰，这里的ksize设为3，
    if kernalType==1:#四邻域模板
        gray_lap = cv2.Laplacian(src, cv2.CV_16S, ksize=1,borderType = cv2.BORDER_CONSTANT)  # 拉式算子
    elif kernalType==2:  #八邻域模板
        gray_lap = cv2.Laplacian(src, cv2.CV_16S, ksize=3,borderType = cv2.BORDER_CONSTANT)  # 拉式算子
    dst = cv2.convertScaleAbs(gray_lap)
    return dst

# LOG 锐化
def log_filter(src):
    src0=cv2.GaussianBlur(src, (3,3), 0)
    gray_lap = cv2.Laplacian(src0, cv2.CV_16S, ksize=3,borderType = cv2.BORDER_DEFAULT)  # 拉式算子
    dst = cv2.convertScaleAbs(gray_lap)
    return dst

# canny operator
def canny_filter(src):
    # 在进行抠取轮廓，其中apertureSize默认为3。
    src0=cv2.GaussianBlur(src, (3, 3), 0)
    canny = cv2.Canny(src0, 50, 150)#threshold1表示第一个滞后性阈值,threshold2表示第二个滞后性阈值
    return canny


# Laplacian operator
# def Laplacian(src):
#     # 为了让结果更清晰，这里的ksize设为3，
#     gray_lap = cv2.Laplacian(src, cv2.CV_16S, ksize=3)  # 拉式算子
#     dst = cv2.convertScaleAbs(gray_lap)
#     cv2.imshow('laplacian', dst)






#####################################
#############性能指标模块############
#####################################

#求简单的图像复原性能指标,默认像素的范围[0,L]=[0.255]

#image0:原图像（Original image） image1:复原的图像（restored image）

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
    SNR=10*math.log10(Sp/(Np+1e-9))   #防止除数为0
    return SNR

def PeakSignalNoiseRatio(image0,image1,L=255):#求峰值信噪比，输出单位：dB
    MSE=meanSquare(image0,image1)
    PSNR=10*math.log(L**2/(MSE+1e-14),10)  #防止除数为0
    return PSNR

def fspecialgauss2D(shape=(3, 3), sigma=0.5):  # 生成高斯滤波和
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    ## The statement below determines the machine
    ## epsilon - if gaussian is smaller than that
    ## set to 0.
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    ## This notation states that it takes the input
    ## h and then it divides by the sum and returns h
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def StructuralSimilarityIndex(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):  # 求结构相似度
    #彩色图像转换成灰度图像
    if len(im1.shape)>2 :#判断img.shape元组的长度
        if im1.shape[2]!=1:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = fspecialgauss2D(shape=(win_size, win_size), sigma=1.5)#高斯滤波和
    window = window / np.sum(np.sum(window))#归一化
    if im1.dtype == np.uint8:#整型转换成浮点型
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
    mu1 = filter2(im1, window, 'valid')     #图像1的均值矩阵
    mu2 = filter2(im2, window, 'valid')     #图像2的均值矩阵
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq   #图像1的方差矩阵
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq   #图像2的方差矩阵
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2    #图像1和2的协方差矩阵
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)) #结构相似度矩阵
    return np.mean(np.mean(ssim_map)) #返回结构相似度矩阵的均值

# LOWPASSFILTER  - 构造一个低通的butterworth过滤器。
# 用法：f = lowpassfilter（sze，cutoff，n）
# 其中：sze是一个双元素向量，指定构造[rows cols]的过滤器的大小。
# 截止频率cutoff是滤波器的截止频率0  -  0.5
# n是滤波器的阶数，n越高，转换越清晰。 （n必须是> = 1的整数）。
# 请注意，n加倍，确保它始终是偶数。
#               1
#        f = --------------------
#                       2N
#          1.0 +（w/cutoff）
# 返回的滤波器的频率原点位于拐角处。
def lowpassfilter(sze, cutoff, n):
    if cutoff < 0 or cutoff > 0.5:
       print('cutoff frequency must be between 0 and 0.5')
    if n%1 != 0 or n < 1:
        print('n must be an integer >= 1')
    if len(sze) == 1:
        rows = sze
        cols = sze
    else:
        rows = sze[0]
        cols = sze[1]
    # 设置X和Y矩阵，范围归一化为 （- 0.5,0.5)
    # 以下代码针对行和列的奇数和偶数值适当调整事物。
    if cols % 2:
        xrange = []
        for i in range(cols):
            xrange.append(i)
        for i in range(cols):
            xrange[i]=(i-(cols-1)/2)/(cols-1)
    else:
        xrange = []
        for i in range(cols):
            xrange.append(i)
        for i in range(cols):
            xrange[i]=(i-cols/2)/cols
    if rows % 2:
        yrange = []
        for i in range(rows):
            yrange.append(i)
        for i in range(rows):
            yrange[i] = (i - (rows - 1) / 2) / (rows - 1)
    else:
        yrange = []
        for i in range(rows):
            yrange.append(i)
        for i in range(rows):
            yrange[i] = (i - rows / 2) / rows
    x, y = numpy.meshgrid(xrange, yrange)
    radius = (x**2 + y**2)**0.5 #每个像素=相对于中心的半径的矩阵。
    f = numpy.fft.ifftshift(1. / (1.0 + (radius / cutoff)** (2 * n)))#所求滤波器
    return f

def phasecong2(im):#求相位一致性
    nscale = 4 #小波滤波尺度
    norient = 4  #滤波器方向
    minWaveLength = 6 #最小规模滤波器的波长
    mult = 2 #连续滤波器之间的缩放因子
    sigmaOnf = 0.55 #描述log Gabor滤波器在频域中的传递函数的
                    # 高斯的标准偏差与滤波器中心频率的比率
    dThetaOnSigma = 1.2 #滤波器方向之间的角度间隔的比率
                        # 和用于在频率平面中构造滤波器的
                        # 角度高斯函数的标准偏差
    k = 2.0  #噪声能量的标准偏差超过我们设定噪声阈值点的平均值
            # 低于哪个阶段一致性值受到惩罚
    epsilon = .0001 #防止除0
    thetaSigma = math.pi / norient / dThetaOnSigma#计算用于在频率平面中构造滤波器的角度高斯函数的标准偏差

    [rows, cols] = im.shape
    imagefft = numpy.fft.fft2(im)#图像的傅里叶变换

    zero = numpy.zeros([rows, cols])
    EO = numpy.empty([nscale, norient])#卷积结果数组

    estMeanE2n = []
    ifftFilterArray = numpy.zeros([nscale,rows,cols])#滤波器的逆傅里叶变换阵列

    # 设置X和Y矩阵，范围归一化为 （- 0.5,0.5)
    # 以下代码针对行和列的奇数和偶数值适当调整事物。
    if cols % 2:
        xrange = []
        for i in range(cols):
            xrange.append(i)
        for i in range(cols):
            xrange[i]=(i-(cols-1)/2)/(cols-1)
    else:
        xrange = []
        for i in range(cols):
            xrange.append(i)
        for i in range(cols):
            xrange[i]=(i-cols/2)/cols
    if rows % 2:
        yrange = []
        for i in range(rows):
            yrange.append(i)
        for i in range(rows):
            yrange[i] = (i - (rows - 1) / 2) / (rows - 1)
    else:
        yrange = []
        for i in range(rows):
            yrange.append(i)
        for i in range(rows):
            yrange[i] = (i - rows / 2) / rows

    x, y= numpy.meshgrid(xrange, yrange)

    radius = (x**2 + y**2)**0.5 #矩阵值包含从中心开始的归一化半径
    theta = numpy.zeros([rows,cols])
    for i in range(rows):
        for j in range(cols):
            theta[i,j] = math.atan2(-y[i,j], x[i,j])  #矩阵值包含极角
                                # （注意-ve y用于给出+ ve逆时针角度）
    radius = numpy.fft.ifftshift(radius)        #象限移位半径和theta
    theta = numpy.fft.ifftshift(theta)          #使得滤波器在拐角处构造为0频率

    radius[0,0] = 1    #摆脱0频率点（现在位于左上角）的0半径值
                        # 以便获取半径的对数不会造成麻烦

    sintheta = numpy.zeros([rows,cols])
    costheta = numpy.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            sintheta[i,j] = math.sin(theta[i,j])
            costheta[i,j] = math.cos(theta[i,j])

    del x
    del y
    del theta
    gc.collect()    #节省内存

    # 过滤器根据两个组件构建。
    # 1）径向分量，控制滤波器响应的频带
    # 2）角度分量，控制滤波器对应的方向。
    # 将这两个组件相乘以构建整个过滤器。
    #
    # 构建径向滤波器组件......
    #
    # 首先构造一个尽可能大的低通滤波器，但在边界处会下降到零。 所有log
    # Gabor滤波器都乘以此值，以确保不包含傅里叶变换“拐角”处的额外频率，因为这在计算相位一致性时似乎扰乱了归一化过程。
    lp = lowpassfilter([rows,cols], .45, 15)

    logGabor = numpy.zeros([norient,rows,cols])

    for s in range(nscale):
        wavelength = minWaveLength * mult**s
        fo = 1.0 / wavelength           #过滤器的中心频率
        Gabor=numpy.zeros([rows,cols])
        for i in range(rows):
            for j in range(cols):
                Gabor[i,j]= math.exp((-(math.log(radius[i,j] / fo))** 2) / (2 * math.log(sigmaOnf)**2))
        logGabor[s]= Gabor
        logGabor[s] = logGabor[s] * lp
        logGabor[s,0,0] = 0       #将滤波器的0频率点处的值设置回零

    #然后构造角度滤波器组件......
    spread = numpy.zeros([norient,rows,cols])

    for o in range(norient):
        angl = o * math.pi / norient      #过滤角度

        # 对于滤镜矩阵中的每个点，计算距指定滤镜方向的角距离。
        # 为了克服角度环绕问题，首先计算正弦差和余弦差值，然后使用atan2函数确定角距离。
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)#正弦差异
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)#余弦差异
        dtheta=np.zeros([rows,cols])
        for i in range(rows):
            for j in range(cols):
                dtheta[i,j] = abs(math.atan2(ds[i,j], dc[i,j]))                          #绝对角距离
                spread [o,i,j]= math.exp((-dtheta[i,j]**2) / (2 * thetaSigma **2))#计算角度滤波器组件
    #主循环
    EnergyAll = np.zeros([rows,cols])
    AnAll = np.zeros([rows,cols])
    EO = np.zeros([nscale,norient,rows,cols],dtype=complex)
    for o in range(norient):    #对于每个方向
        sumE_ThisOrient = zero      #初始化累加器矩阵
        sumO_ThisOrient = zero
        sumAn_ThisOrient = zero
        Energy = zero
        for s in range(nscale):
            filter = logGabor[s]* spread[o]     #将径向和角度分量相乘以得到滤波器
            ifftFilt = numpy.fft.ifft2(filter).real * (rows * cols)**0.5    #注意重新缩放以匹配功率
            ifftFilterArray[s] = ifftFilt       #记录滤波器的ifft2
            EO[s, o] = numpy.fft.ifft2(imagefft * filter)#使用偶数和奇数滤波器卷积图像，将结果返回到EO

            An = abs(EO[s, o])              #偶数和奇数滤波器响应的幅度
            sumAn_ThisOrient = sumAn_ThisOrient + An    #幅度响应之和
            sumE_ThisOrient = sumE_ThisOrient + EO[s, o].real   #偶数滤波器卷积结果的总和
            sumO_ThisOrient = sumO_ThisOrient + EO[s, o].imag   #奇数滤波器卷积结果的总和
            if s == 0:
                EM_n = sum(sum(filter**2))  #用于噪声估计
                maxAn = An                  #在所有比例上记录最大值An
            else:
                for i in range(rows):
                    for j in range(cols):
                        maxAn[i,j] = max(maxAn[i,j], An[i,j])
                #并处理下一个比例

        #获得加权平均滤波器响应矢量，给出加权平均相位角
        XEnergy = (sumE_ThisOrient**2 + sumO_ThisOrient**2)**0.5 + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        #现在计算An（cos（phase_deviation） -  | sin（phase_deviation））|
        # 通过在每个尺度上使用加权平均滤波器响应矢量和各个滤波器响应矢量之间的点和叉积来实现
        # 这个数量是相位一致性乘以An，我们称之为energy
        for s in range(nscale):
            E = EO[s, o].real
            O = EO[s, o].imag
            Energy = Energy + E * MeanE + O * MeanO - abs(E * MeanO - O * MeanE)
        # 补偿噪音
        #    我们从最小尺度的能量平方响应估计噪声功率
        #      如果噪声是高斯噪声，则能量平方将服从卡方分布
        #      我们计算中值能量平方响应，因为这是一个稳健的统计数据，由此我们估计平均值
        #      通过将均方能量值除以均方滤波器值来获得噪声功率的估计
        EO2=abs(EO[0, o])** 2
        medianE2n = numpy.median(EO2.reshape(1, rows * cols))
        meanE2n = -medianE2n / math.log(0.5)
        estMeanE2n.append(o)
        estMeanE2n[o] = meanE2n

        noisePower = meanE2n / EM_n     #估计噪声功率

        #现在估计由噪声引起的总能量的平方
        #通过sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))估计

        EstSumAn2 = zero
        for s in range(nscale):
            EstSumAn2 = EstSumAn2 + ifftFilterArray[s]**2

        EstSumAiAj = zero
        for si in range(nscale - 1):
            for sj in range(si+1,nscale-1-si):
                EstSumAiAj = EstSumAiAj + ifftFilterArray[si]* ifftFilterArray[sj]

        sumEstSumAn2 = sum(sum(EstSumAn2))
        sumEstSumAiAj = sum(sum(EstSumAiAj))

        EstNoiseEnergy2 = 2 * noisePower * sumEstSumAn2 + 4 * noisePower * sumEstSumAiAj

        tau = (EstNoiseEnergy2 / 2) **0.5   #瑞利参数
        EstNoiseEnergy = tau * (math.pi / 2)**0.5   #噪声能量的预期值
        EstNoiseEnergySigma = ((2 - math.pi / 2) * tau**2)**0.5

        T = EstNoiseEnergy + k * EstNoiseEnergySigma    #噪音阈值
        # 上面计算的估计噪声影响仅对PC_1度量有效
        # PC_2措施不适合同样的分析
        # 然而，根据经验，这里使用的滤波器参数的噪声效应似乎被高估了1.7倍
        T = T / 1.7     # 估计噪声影响的经验，重新调整以适应PC_2相位一致性度量
        for i in range(rows):
            for j in range(cols):
                Energy[i,j] = max(Energy[i,j] - T, 0)

        EnergyAll = EnergyAll + Energy
        AnAll = AnAll + sumAn_ThisOrient
        #对于每个方向
    ResultPC = EnergyAll /AnAll
    return ResultPC

def getGM(img):#求图像的梯度
    Scharr_x = 1 / 16 * np.array([[3, 0, -3],  # x方向的梯度算子
                                  [10, 0, -10],
                                  [3, 0, -3]])
    Scharr_y = 1 / 16 * np.array([[3, 10, 3],  # y方向的梯度算子
                                  [0, 0, 0],
                                  [-3, -10, -3]])
    GM_x = cv2.filter2D(img, -1, Scharr_x, borderType=cv2.BORDER_DEFAULT)
    GM_y = cv2.filter2D(img, -1, Scharr_y, borderType=cv2.BORDER_DEFAULT)
    GM=(GM_x**2+GM_y**2)**0.5
    return GM

def FeatureSimilarityIndex(img1,img2,L=255):#求特征相似度
    if len(img1.shape)>2 :#判断img.shape元组的长度
        if img1.shape[2]!=1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    m,n=img1.shape
    minDimension = min(m, n)
    F = max(1, round(minDimension / 256))#采样步长
    aver1 = cv2.blur(img1, (3, 3))  # 均值滤波
    aver2 = cv2.blur(img2, (3, 3))
    im1=np.zeros([math.ceil(m/F),math.ceil(n/F)])#定义采样后矩阵
    im2=np.zeros([math.ceil(m/F),math.ceil(n/F)])
    for i in range(0,m,F):
        for j in range(0,n,F):
            im1[int(i/F),int(j/F)]=aver1[i,j]#采样赋值
            im2[int(i/F),int(j/F)]=aver2[i,j]
    [rows,cols]=im1.shape
    GM1=getGM(im1)#求梯度
    GM2=getGM(im2)
    PC1=phasecong2(im1)#求相位一致性
    PC2=phasecong2(im2)
    T1 = 0.85
    T2 = 160
    S_PC = (2 * PC1* PC2 + T1)/ (PC1**2 + PC2**2 + T1)
    S_GM= (2 * GM1 * GM2 + T2) / (GM1**2 + GM2**2 + T2)
    PCm=np.zeros([rows,cols])
    for i in range(rows):
        for j in range(cols):
            PCm[i,j] = max(PC1[i,j], PC2[i,j])
    SimMatrix =S_GM* S_PC* PCm
    FSIM = sum(sum(SimMatrix)) / sum(sum(PCm))#分别求和后求比值
    return FSIM

#输出均方误差，信噪比，峰值信噪比
def SimpleRPMPrint(image0,image1):
    MSE=meanSquare(image0,image1)
    SNR=SignalNoiseRatio(image0,image1)
    PSNR=PeakSignalNoiseRatio(image0,image1)
    print("均方误差为：",MSE)
    print("信噪比为：", SNR)
    print("峰值信噪比为：", PSNR)

#输出均方误差，信噪比，峰值信噪比，结构相似度，特征相似度
def RPMPrint(image0,image1):
    MSE=meanSquare(image0,image1)
    SNR=SignalNoiseRatio(image0,image1)
    PSNR=PeakSignalNoiseRatio(image0,image1)
    SSIM=StructuralSimilarityIndex(image0,image1)
    FSIM=FeatureSimilarityIndex(image0,image1)
    print("均方误差为：",MSE)
    print("信噪比为：", SNR)
    print("峰值信噪比为：", PSNR)
    print("结构相似度为：", SSIM)
    print("特征相似度为：",FSIM)





if __name__=="__main__":
    img = cv2.imread(r'.\img\6.png',0)
    border_fill(img)
    # img=create_img()
    # cv2.imwrite(r".\img\idealImg.jpg",img)

    # noise_sigma = 0.1
    # # NoiseImg = add_mean_noise(img,100,150,0.5)
    # NoiseImg = add_mean_noise(img)
    # # NoiseImg = add_mean_noise(NoiseImg, 100, 150, 0.3)
    # # NoiseImg = add_mean_noise(NoiseImg, 200, 250, 0.3)
    # histr=Histogram.gray_histogram(NoiseImg)
    # plt.subplot(121)
    # plt.bar(np.arange(256),histr.flatten())
    # plt.subplot(122)
    # plt.imshow(NoiseImg, cmap='gray')
    plt.show()


   #  cv2.imshow('original image',img)
   #  ImgNoise = addGaussianNoise(img,0,0.1)    #添加0均值，0.1方差的高斯分布噪声
   #  print(ImgNoise)
   #  cv2.imshow('Image added gaussian noise',ImgNoise)
   #  border_fill(ImgNoise)
   #  # 显示去噪后的图片
   #  cv2.imshow('constant image ', constant)
   #  cv2.imshow('replicate image ', replicate)
   #  cv2.imshow('default image ', default)
   #  cv2.imshow('wrap image ', wrap)
   #
   #  # 打印输出重构后图像的均方误差
   #  print("CONSTANT：")
   #  SimpleRPMPrint(ImgNoise, constant)
   #  print("REPLICATE：")
   #  SimpleRPMPrint(ImgNoise, replicate)
   #  print("DEFAULT：")
   #  SimpleRPMPrint(ImgNoise, default)
   #  print("WRAP：")
   #  SimpleRPMPrint(ImgNoise, wrap)
   #  img_L = cv2.GaussianBlur(img, (3, 3), 0)  # 先进行高斯滤波降噪。
   #  Laplacian(img_L)
   #  Sobel(img_L)
   #  canny(img_L)
   #  '''
   #  fil = 1/16*np.array([[ 1,2, 1],                        #这个是设置的滤波，也就是卷积核
   #                      [ 2, 4, 2],
   #                      [ 1, 2, 1]])
   #  '''
   #
   #
   #  '''
   # constant= cv2.filter2D(ImgNoise,-1,fil,borderType = cv2.BORDER_CONSTANT)
   # replicate= cv2.filter2D(ImgNoise,-1,fil,borderType = cv2.BORDER_REPLICATE)
   # default= cv2.filter2D(ImgNoise,-1,fil,borderType = cv2.BORDER_DEFAULT)
   # wrap= cv2.filter2D(ImgNoise,-1,fil,borderType = cv2.BORDER_WRAP)
   #  '''
   #
   #  '''
   # plt.figure(1)
   # plt.subplot(231),plt.imshow(img,'gray'),plt.title('original image')
   # plt.subplot(232),plt.imshow(ImgNoise,'gray'),plt.title('Image added gaussian noise')
   # plt.subplot(233),plt.imshow(constant,'gray'),plt.title('constant image')
   # plt.subplot(234),plt.imshow(replicate,'gray'),plt.title('replicate image ')
   # plt.subplot(235),plt.imshow(default,'gray'),plt.title('default image ')
   # plt.subplot(236),plt.imshow(wrap,'gray'),plt.title('wrap image ')
   # plt.show()
   #  '''
   #  cv2.waitKey(0)