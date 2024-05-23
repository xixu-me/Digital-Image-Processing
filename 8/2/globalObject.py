# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import os
import sys
# 全局变量
picSize=400    #显示图像大小
img_open = None  #存放读取的图片
img_gray = None
img_result = None  #存放处理结果图片
img_binary = None  #二值图像
# stepFlag = False  #操作对象标志
#一些模块是否运行标志
PSF=None #退化模糊核
contoursMinArea=None  #最小区域面积，如果运行了目标区域轮廓模块，则此值非空
path_ = r".\img"      #存放图片路径
img_empty = cv2.imread('.\img\empty.png',0)#重载的空图像
current=os.getcwd()
sys.path.append(current)