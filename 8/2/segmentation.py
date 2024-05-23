import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

def basic_global_thresholding(Img,T0=0.1): #输入的图像要求为灰度图像
    '''
    :param Img: 需进行全阈值分割的图像
    :param T0: 迭代终止容差，当相临迭代得到的阈值差小于此值，则终止迭代
    :return: 全阈值
    '''
    G1 = np.zeros(Img.shape, np.uint8)  # 定义矩阵分别用来装被阈值T1分开的两部分
    G2 = np.zeros(Img.shape, np.uint8)
    T1 = np.mean(Img)
    diff=255
    while(diff>T0):
        _,G1=cv2.threshold(Img,T1,255,cv2.THRESH_TOZERO_INV) #THRESH_TOZERO	超过thresh的像素不变, 其他设为0
        _,G2=cv2.threshold(Img,T1,255,cv2.THRESH_TOZERO)
        garray1 = np.array(G1)
        garray2 = np.array(G2)
        loc1 = np.where(garray1>0.001)  #可以对二维数组操作
        loc2 = np.where(garray2 > 0.001)
        # g1 = list(filter(lambda a: a > 0, G1.flatten()))#只能对一维列表筛选，得到的是一个筛选对象
        # g2 = list(filter(lambda a: a > 0, G2.flatten()))
        ave1=np.mean(garray1[loc1])
        ave2=np.mean(garray2[loc2])
        T2=(ave1+ave2)/2.0
        diff=abs(T2 - T1)
        T1=T2
    return T2



def moving_threshold(image, num,b=0.5):
    '''
    :param image: 将进行阈值分割的图像，为单通道灰度图像
    :param num: 滑动窗口大小
    :param b:  分割权重比例，灰度值大于b*平均值的像素点将设置为白色
    :return: 滑动平均阈值分割图像
    '''
    width = image.shape[0]
    height = image.shape[1]
    widthStep = width
    data = image.flatten()  # 转换成一维向量
    dstdata = data.copy()
    n = float(num)
    m_pre = data[0]/n

    for i in range(0,height-1):
        for j in range(0,width-1):
            index = i * width + j
            if index < num + 1:
                dif = data[index]
            else:
                dif = int(data[index]) - int(data[index-num-1])
            dif *= 1/n
            m_now = m_pre + dif #m_now存放着当前像素点的滑动平均
            m_pre = m_now
            if data[index] > round(b * m_now):  #b是一个阈值权重
                dstdata[index] = 255;
            else:
                dstdata[index] = 0;
    return np.array(dstdata).reshape(width, height)

# 自适应中值滤波
def get_window(res_img, noise_mask, sc, i, j, k):
    listx = []

    if i - sc >= 0:
        starti = i - sc
    else:
        starti = 0
    if j + 1 <= res_img.shape[1] - 1 and noise_mask[0, j + 1, k] != 0:
        listx.append(res_img[0, j + 1, k])
    if j - 1 >= 0 and noise_mask[0, j - 1, k] != 0:
        listx.append(res_img[0, j - 1, k])

    if i + sc <= res_img.shape[0] - 1:
        endi = i + sc
    else:
        endi = res_img.shape[0] - 1
    if j + 1 <= res_img.shape[1] - 1 and noise_mask[endi, j + 1, k] != 0:
        listx.append(res_img[endi, j + 1, k])
    if j - 1 >= 0 and noise_mask[endi, j - 1, k] != 0:
        listx.append(res_img[endi, j - 1, k])

    if j + sc <= res_img.shape[1] - 1:
        endj = j + sc
    else:
        endj = res_img.shape[1] - 1
    if i + 1 <= res_img.shape[0] - 1 and noise_mask[i + 1, endj, k] != 0:
        listx.append(res_img[i + 1, endj, k])
    if i - 1 >= 0 and noise_mask[i - 1, endj, k] != 0:
        listx.append(res_img[i - 1, endj, k])

    if j - sc >= 0:
        startj = j - sc
    else:
        startj = 0
    if i + 1 <= res_img.shape[0] - 1 and noise_mask[i + 1, 0, k] != 0:
        listx.append(res_img[i + 1, 0, k])
    if i - 1 >= 0 and noise_mask[i - 1, 0, k] != 0:
        listx.append(res_img[i - 1, 0, k])

    for m in range(starti, endi + 1):
        for n in range(startj, endj + 1):
            if noise_mask[m, n, k] != 0:
                listx.append(res_img[m, n, k])
    listx.sort()
    return listx


def get_window_small(res_img, noise_mask, i, j, k):
    listx = []
    sc = 1
    if i - sc >= 0 and noise_mask[i - 1, j, k] != 0:
        listx.append(res_img[i - 1, j, k])

    if i + sc <= res_img.shape[0] - 1 and noise_mask[i + 1, j, k] != 0:
        listx.append(res_img[i + 1, j, k])

    if j + sc <= res_img.shape[1] - 1 and noise_mask[i, j + 1, k] != 0:
        listx.append(res_img[i, j + 1, k])

    if j - sc >= 0 and noise_mask[i, j - 1, k] != 0:
        listx.append(res_img[i, j - 1, k])
    listx.sort()
    return listx


def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    for i in range(noise_mask.shape[0]):
        for j in range(noise_mask.shape[1]):
            for k in range(noise_mask.shape[2]):
                if noise_mask[i, j, k] == 0:
                    sc = 1
                    listx = get_window_small(res_img, noise_mask, i, j, k)
                    if len(listx) != 0:
                        res_img[i, j, k] = listx[len(listx) // 2]
                    else:
                        while (len(listx) == 0):
                            listx = get_window(res_img, noise_mask, sc, i, j, k)
                            sc = sc + 1
                        if sc > 4:
                            res_img[i, j, k] = np.mean(listx)
                        else:
                            res_img[i, j, k] = listx[len(listx) // 2]

    return res_img

if __name__ == '__main__':
    # 读入图像
    srcImage = cv2.imread(r".\img\kennysmall.jpg", 0)
    b=0.65
    dstImage = moving_threshold(srcImage, 11,b)
    plt.subplot(121), plt.imshow(srcImage, "gray")
    plt.title("source image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dstImage, "gray")
    plt.title("processed image"), plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.waitKey(0)
