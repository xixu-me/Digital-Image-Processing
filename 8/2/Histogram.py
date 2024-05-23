#coding:utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

#绘制灰度直方图
def gray_histogram(img,bins=256):
    img=np.uint8(img)
    # img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#opencv默认的imread是以BGR的方式进行存储的
    histr = cv2.calcHist([img],[0],None,[bins],[0,255])#绘制直方图
    # plt.bar(np.arange(bins),histr.flatten())
    return histr


#绘制彩色直方图
def colorHistogram(imgName):
    color = ('b', 'g', 'r')
    #绘制灰度图像的直方图
    img1=cv2.imread(imgName)
    img2=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)#opencv默认的imread是以BGR的方式进行存储的
    # 而matplotlib的imshow默认则是以RGB格式展示所以此处我们必须对图片的通道进行转换
    plt.subplot(221)
    plt.title('color image'), plt.xticks([]), plt.yticks([])
    plt.imshow(img2)
    plt.subplot(222)
    plt.title('colorHistogram image')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img2], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
    plt.savefig("pic_change.jpg")

#掩膜直方图
def mask_histogram(img,imgMask,binSize):
    hight,weight = img.shape
    mask0 = cv2.resize(imgMask,(weight,hight))
    _ , mask= cv2.threshold(mask0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked_img = cv2.bitwise_and(img, img, mask=mask)  # 掩模的黑色区域(像素值为0)用来遮盖原图img,cv2.bitwise_and()函数的功能是位与
    cv2.imwrite('mask.jpg',masked_img)
    histr = cv2.calcHist([img], [0], mask, [256], [0, 256])
    plt.plot(histr)
    return histr, masked_img
#
# #局部图像直方图
# def locality_histogram (imgName,x=0,y=0,w=0.5,h=0.6):
#     imgNewName = 'huan.png'
#     img = Image.open(imgName)
#     img_size = img.size
#     weight,hight = img_size
#     x = x * weight
#     y = y * hight
#     w = w * weight
#     h = h * hight
#     region = img.crop((x, y, x + w, y + h))
#     region.save(imgNewName)
#     img1=cv2.imread(imgNewName)
#     histr = cv2.calcHist([img1], [0], None, [256], [0, 256])#绘制直方图
#     return  histr, region

#直方图均衡化
def equalization_histogram(img):
    equ = cv2.equalizeHist(img)   #对图像进行直方图均衡化处理
    histogram = cv2.calcHist([equ], [0], None, [256], [0, 256])#分组越多得到的均衡化直方图效果越明显
    return histogram, equ

#规定化直方图
# type=1目标直方图是正三角形， type=2倒三角形， type=3平形，
def regulation_histogram(img,type=1,mapSML=True):
    '''
    :param img:
    :param type: type=1目标直方图是正三角形， type=2倒三角形， type=3平形，
    :param mapSML: True是SML False是GML
    :return: 直方图和图像
    '''
    rows,cols = img.shape
    gray_flat = img.reshape((rows*cols,))

    dif = np.zeros((256,256),np.float) #用于存放原直方图与目标直方图的差
    S = np.zeros((rows,cols),np.uint8) #单映射规定化后图像
    G = np.zeros((rows,cols),np.uint8) #组映射规定化后图像
    src = np.zeros((256,),np.int32)    #原直方图
    dst = np.zeros((256,),np.int32)    #规定直方图
    H_SML = np.zeros((256,),np.int32)   #单映射的映射关系
    H_GML = np.zeros((256,), np.int32)  #组映射的映射关系
    SH = np.zeros((256,),np.float)     #单映射规定化后直方图
    GH = np.zeros((256,),np.float)    #组映射规定化后直方图

    # 计算原图像各个灰度级数量
    for index,value in enumerate(gray_flat):
        src[value] += 1

    #归一化处理
    src_pro = src/sum(src)

    # 计算灰度级的累计分布
    for i in range(1,256):
        src_pro[i] = src_pro[i - 1] + src_pro[i]

    #目标直方图为正直角三角形
    if type==1:
        for i in range(256):
            dst[i] = i
    #目标直方图为倒三角形
    if type==2:
        for i in range(256):
            dst[i] = 256-i
    #目标直方图是平行的
    if type==3:
        for i in range(256):
            dst[i] = 128

    # plt.plot(dst)
    # plt.show()
    #归一化处理
    dst_pro=dst/sum(dst)

    # 计算规定化灰度级的累计分布
    for i in range(1,256):
        dst_pro[i] = dst_pro[i - 1] + dst_pro[i]

    #|V2-V1|计算目标直方图与原直方图的差
    for i in range(256):
        for j in range(256):
            dif[i,j] = abs(src_pro[i]-dst_pro[j])

    #SML单映射
    if mapSML==True:
        for i in range(256):
            minx = 0
            minvalue = dif[i,0]
            for j in range(1,256):
                if(dif[i,j]<minvalue):
                    minvalue=dif[i,j]
                    minx=j
            H_SML[i]=minx   #将灰度i映射为灰度minx

        for i in range(256):
            SH[H_SML[i]]+= src[i]  #src[i]是灰度为i的像素个数，SH是规定化后的直方图

        SHpro = SH/sum(SH)

        for i in range(rows):
            for j in range(cols):
                S[i,j]=H_SML[img[i,j]]   #S是单映射得到的图像，将灰度值img[i,j]映射为H_SML[img[i,j]]
        return SHpro, S
    else:
        #GML群映射
        lastStartY = 0
        lastEndY = 0
        startY = 0
        endY = 0
        for i in range(256):
            minvalue = dif[0,i]
            for j in range(1,256):
                if(minvalue>dif[j,i]):
                    minvalue=dif[j,i]
                    endY=j
            if(startY != lastStartY ) or (endY != lastEndY):
                for k in range(startY,endY+1):
                    H_GML[k]=i

                lastStartY=startY
                lastEndY=endY
                startY=lastEndY+1

        for i in range(256):
            GH[H_GML[i]]+= src[i]   #组映射直方图

        for i in range(rows):
            for j in range(cols):
                G[i,j]=H_GML[img[i,j]]   #G是组映射得到的图像

        GHpro = GH/sum(GH)
        return GHpro, G

    # plt.subplot(231)
    # # plt.hist(img.ravel(), 256, [0, 256])  # 绘制直方图，img.ravel()将图像转为一维数组
    # plt.plot(src)
    # plt.title('original image')
    # plt.subplot(232)
    # plt.title('SML单映射直方图')
    # plt.plot(SH)
    # plt.subplot(233)
    # plt.title('GML组映射直方图')
    # plt.plot(GH)
    # plt.subplot(234)
    # plt.title('original image'),plt.xticks([]), plt.yticks([])
    # plt.imshow(img)
    # plt.subplot(235)
    # plt.title('SML image'), plt.xticks([]), plt.yticks([])
    # plt.imshow(S)
    # plt.subplot(236)
    # plt.title('GML image'), plt.xticks([]), plt.yticks([])
    # plt.imshow(G)



if __name__=="__main__":
    img=cv2.imread(r'.\img\6.png',0)
    # imgMask=cv2.imread(r'.\img\mask.jpg',0)
    # mask_histogram(img,imgMask,64)
    #colorHistogram('timg.jpg')
    #maskHistogram('timg.jpg')
    #equalizationHistogram('timg.jpg')
    #localityHistogram('timg.jpg')
    regulation_histogram(img)
    plt.show()