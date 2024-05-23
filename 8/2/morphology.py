import cv2
from skimage import morphology,data,color,io,measure,filters,feature

import numpy as np

def binary_img(img):
    '''
    图像二值化
    :param img:需要处理的图片
    :return:返回二值化后的图片
    '''
    if len(img.shape)>2:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,dst=cv2.threshold(img,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return dst

def liantongquyu(img):
    '''
    提取连通区域
    :param img: 待处理图像
    :return: 连通区域
    '''
    dst=binary_img(img)
    labels=measure.label(dst,connectivity=2)  #8连通区域标记
    dst0=color.label2rgb(labels)  #根据不同的标记显示不同的颜色
    print('regions number:',labels.max()+1)  #显示连通区域块数(从0开始标记)
    return dst0


def fill_color_demo(img):
    '''
    孔洞填充
    :param img: 待处理图像
    :return: 填充孔洞后的图像
    '''
    copy_image = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    cv2.floodFill(copy_image, mask, (30,30), (0, 255, 255),(100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)#区域填充参数设置#
    return copy_image

def get_connective_region(img,minArea=100):
    img0=img.copy()
    img1=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    img1 = cv2.Laplacian(img1,cv2.CV_64F)
    img1 = cv2.convertScaleAbs(img1)
    row,col=img1.shape
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img1)
    for istat in stats:
        if istat[4] >minArea and istat[2]<col and istat[3]<row:
            # print(istat[0:2])
            # if istat[3] > istat[4]:
            #     r = istat[3]
            # else:
            #     r = istat[4]
            cv2.rectangle(img0, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), (0, 0, 255), thickness=-1)
            # plt.imshow(img,cmap=)
            # plt.show()
    return img0


def fill_holes(imgBinary,kernel):
    '''
    孔洞填充
    :param imgBinary: 待处理二值图像
    :param kernel: 结构算子
    :return: 填充孔洞后的图像
    '''
    # 原图取补得到MASK图像
    mask = 255 - imgBinary
    # 构造Marker图像
    marker = np.zeros_like(imgBinary)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255
    marker_0 = marker.copy()
    while True:
        marker_pre = marker
        dilation = cv2.dilate(marker, kernel)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    dst = 255 - marker
    filling = dst - imgBinary
    return dst



