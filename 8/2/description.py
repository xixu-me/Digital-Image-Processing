#coding='utf-8'
import cv2
import numpy as np
import random
import math

def draw_contours(imgColor,minArea=50):
    if len(imgColor.shape)>2:
        imgResult=cv2.cvtColor(imgColor,cv2.COLOR_BGR2GRAY)
    else:
        imgResult = imgColor.copy()
    ret, img_binary = cv2.threshold(imgResult, 125, 255, 0)
    contours0, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=list([])
    for cnt in contours0:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] * rect[1][1] < minArea:  # 此矩形区域的面积小于10，则忽略
            continue
        contours.append(cnt)#contours=contours.append(cnt)会返回NoneType错误
    cv2.drawContours(imgResult, contours, -1, (0, 0, 255), 3)
    return imgResult
def getDis(pointX, pointY, lineX1, lineY1, lineX2, lineY2):
    '''返回点线之间的距离
    '''
    a = lineY2 - lineY1
    b = lineX1 - lineX2
    c = lineX2 * lineY1 - lineX1 * lineY2
    dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))
    return dis
def get_region_description(imgColor,minArea=50):
    if len(imgColor.shape)>2:
        imgResult=cv2.cvtColor(imgColor,cv2.COLOR_BGR2GRAY)
    else:
        imgResult = imgColor.copy()
    _, imgBinary = cv2.threshold(imgResult, 125, 255, 0)
    row,col=imgBinary.shape[:2]
    contours0, hierarchy = cv2.findContours(imgBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 去除小面积区域
    contours_=[]
    for cnt in contours0:
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] * rect[1][1] < minArea:  # 此矩形区域的面积小于10，则忽略
            continue
        contours_.append(cnt)#contours=contours.append(cnt)会返回NoneType错误
    cv2.drawContours(imgResult, contours_, -1, (0, 0, 255), 3)

    contours=[]  #存放区域描述子，包括重心、周长、面积、圆形度、矩形度
    # 进行区域描述
    for cnt in contours_:
        oneContour = []  # 存放一个区域的描述子
        # minAreaRect返回值：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
        #角度是【-90，0】，0度会写成-90度，方向不准确
        rect = cv2.minAreaRect(cnt)
        # maxRegion=min(rect[1][0],rect[1][1])/2
        # maxV=maxRegion*maxRegion #区域最大半径平方值
        points=cv2.boxPoints(rect)#可以返回四个点的值

        #获得重心
        cx,cy=int(rect[0][0]),int(rect[0][1])
        oneContour.append((cx,cy))   #添加重心
        # #获得区域方向，并绘制方向
        '''minV = math.inf  #区域上的点到直线距离和的最小值
        min_i = 0
        min_j = 0
        for i in range(row):
            for j in range(col):
                sum = 0
                # 获得一个区域最大半径圆上的点，做点到质心的线
                if (i - cx) * (i - cx) + (j - cy) * (j - cy) == maxV:  # 一个圆上的点
                    for c0 in cnt[:][:]:  # cnt中存放中区域轮廓上的点坐标
                        sum += getDis(c0[0][0], c0[0][1], cx, cy, i, j)
                    if sum < minV:
                        minV = sum
                        min_i = i
                        min_j = j
        cv2.line(imgResult, (cx, cy), (min_i, min_j), (0, 0, 0), 3, 8)
        regionDirect = math.atan2(min_j-cy,min_i-cx)
        # regionDirect = rect[2]
        oneContour.append(regionDirect)'''
        minV = math.inf
        findPoint=(0,0)
        rectPoints=[]#获得区域最上边、下边、左边、右边线上的点,做点到中心的线
        minRow=int(min(points[i][0] for i in range(4))) #最小行坐标
        maxRow=int(max(points[i][0] for i in range(4)))
        minCol=int(min(points[i][1] for i in range(4)))
        maxCol=int(max(points[i][1] for i in range(4)))
        highPointNum=int(maxRow-minRow+1)
        widthPointNum=int(maxCol-minCol+1)
        fixI1=zip([minRow]*widthPointNum,range(minCol,maxCol+1,1))
        rectPoints.extend(fixI1)
        fixI2=zip([maxRow]*widthPointNum,range(minCol,maxCol+1,1))
        rectPoints.extend(fixI2)
        fixJ1=zip(range(minRow,maxRow+1,1),[minCol]*highPointNum)
        rectPoints.extend(fixJ1)
        fixJ2=zip(range(minRow,maxRow+1,1),[maxCol]*highPointNum)
        rectPoints.extend(fixJ2)
        for point in rectPoints:
            sum = 0
            for c0 in cnt[:][:]:  # cnt中存放中区域轮廓上的点坐标
                sum += getDis(c0[0][0], c0[0][1], cx, cy, point[0],point[1])
            if sum < minV:
                minV = sum
                findPoint=point
        cv2.line(imgResult, (cx, cy), findPoint, (0, 0, 0), 3, 8)
        regionDirect =math.atan2(cy-findPoint[1],cx-findPoint[0])*180/3.14
        oneContour.append(regionDirect)  # 添加重心
        # 计算轮廓面积和周长
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,False)
        circle = 4.0*np.pi*area/(perimeter*perimeter)  # 计算圆形度
        rectangle = area/(rect[1][0]*rect[1][1])     # 计算矩形度
        oneContour.append(area)
        oneContour.append(perimeter)
        oneContour.append(circle)
        oneContour.append(rectangle)
        contours.append(oneContour)
    return imgResult,contours

# Hu Moments 胡不变矩 wiki： https://en.wikipedia.org/wiki/Image_moment
# 公式 https://docs.opencv.org/4.1.2/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944
# 代码 https://github.com/opencv/opencv/blob/b6a58818bb6b30a1f9d982b3f3f53228ea5a13c1/modules/imgproc/src/moments.cpp
# void cv::HuMoments( const Moments& m, double hu[7] )
# {
#     CV_INSTRUMENT_REGION();
#
#     double t0 = m.nu30 + m.nu12;
#     double t1 = m.nu21 + m.nu03;
#
#     double q0 = t0 * t0, q1 = t1 * t1;
#
#     double n4 = 4 * m.nu11;
#     double s = m.nu20 + m.nu02;
#     double d = m.nu20 - m.nu02;
#
#     hu[0] = s;
#     hu[1] = d * d + n4 * m.nu11;
#     hu[3] = q0 + q1;
#     hu[5] = d * (q0 - q1) + n4 * t0 * t1;
#
#     t0 *= q0 - 3 * q1;
#     t1 *= 3 * q0 - q1;
#
#     q0 = m.nu30 - 3 * m.nu12;
#     q1 = 3 * m.nu21 - m.nu03;
#
#     hu[2] = q0 * q0 + q1 * q1;
#     hu[4] = q0 * t0 + q1 * t1;
#     hu[6] = q1 * t0 - q0 * t1;
# }
#求图像的24个矩
def get_moments(img,moments=None):
    # row == heigh == Point.y
    # col == width == Point.x
    # Mat::at(Point(x, y)) == Mat::at(y,x)
    # https://blog.csdn.net/puqian13/article/details/87937483
    mom = [0] * 10  #10个原点矩
    umom = [0] * 7   #7个中心矩
    numom = [0] * 7  #7个归一化中心矩
    rows, cols = img.shape
    for y in range(rows):
        # temp var
        x0 = 0
        x1 = 0
        x2 = 0
        x3 = 0
        for x in range(cols):
            p = img[y,x]
            xp = x * p
            xxp = x * xp
            x0 = x0 + p
            x1 = x1 + xp
            x2 = x2 + xxp
            x3 = x3 + xxp * x
        py = y * x0
        sy = y*y
        mom[9] += (py) * sy # m03
        mom[8] += (x1) * sy # m12
        mom[7] += (x2) * y # m21
        mom[6] += x3 # m30
        mom[5] += x0 * sy # m02
        mom[4] += x1 * y # m11
        mom[3] += x2 # m20
        mom[2] += py # m01
        mom[1] += x1 # m10
        mom[0] += x0 # m00
    x_a = mom[1] / mom[0]
    y_a = mom[2] / mom[0]
    for y in range(0,rows):
        x_a_0 = 0
        x_a_1 = 0
        x_a_2 = 0
        x_a_3 = 0
        for x in range(0,cols):
            p = img[y,x]
            x_a_0 = x_a_0 +  p
            x_a_1 += p * (x - x_a )
            x_a_2 += x_a_1 * (x - x_a )
            x_a_3 += x_a_2 *  (x - x_a )
        y_a_1 =  (y - y_a)
        y_a_2  = y_a_1 * (y - y_a)
        y_a_3  = y_a_2 * (y - y_a)
        umom[0] += x_a_2
        umom[1] += x_a_1 * y_a_1
        umom[2] += x_a_0 * y_a_2
        umom[3] += x_a_3
        umom[4] += x_a_2 * y_a_1
        umom[5] += x_a_1 * y_a_2
        umom[6] += x_a_0 * y_a_3

    cx = mom[1] * 1.0 / mom[0]
    cy = mom[2] * 1.0 / mom[0]

    umom[0] = mom[3] - mom[1] * cx
    umom[1] = mom[4] - mom[1] * cy
    umom[2] = mom[5] - mom[2] * cy

    umom[3] = mom[6] - cx * (3 * umom[0] + cx * mom[1])
    umom[4] = mom[7] - cx * (2 * umom[1] + cx * mom[2]) - cy * umom[0]
    umom[5] = mom[8] - cy * (2 * umom[1] + cy * mom[1]) - cx * umom[2]
    umom[6] = mom[9] - cy * (3 * umom[2] + cy * mom[2])
    #  nu
    inv_sqrt_m00 = np.sqrt(abs(1.0 / mom[0]))
    s2 = (1.0 / mom[0]) * (1.0 / mom[0])
    s3 = s2 * inv_sqrt_m00
    numom[0] = umom[0] * s2
    numom[1] = umom[1] * s2
    numom[2] = umom[2] * s2
    numom[3] = umom[3] * s3
    numom[4] = umom[4] * s3
    numom[5] = umom[5] * s3
    numom[6] = umom[6] * s3
    moments = mom + umom + numom
    return moments

def get_hu_moments(numom):
    '''
    :param numom: 图像的原点矩
    :return: 返回7个不变矩
    '''
    hu = [0]*7
    t0 = numom[3] + numom[5]
    t1 = numom[4] + numom[6]

    q0 = t0 * t0
    q1 = t1 * t1

    n4 = 4 * numom[1];
    s = numom[0] + numom[2];
    d = numom[0] - numom[2];

    hu[0] = s
    hu[1] = d * d + n4 * numom[1]
    hu[3] = q0 + q1
    hu[5] = d * (q0 - q1) + n4 * t0 * t1
    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1

    q0 = numom[3] - 3 * numom[5]
    q1 = 3 * numom[4]  - numom[6] ;

    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1
    return hu

def moment_invariants(img):
    textLists=[]  #用来存放几何变换名称
    imgLists=[]   #用来存放几何变换图像
    # 旋转45度
    rotate_45_matrix = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), -45, 1)
    rotate_45_image = cv2.warpAffine(img, rotate_45_matrix, dsize=(img.shape[1], img.shape[0]))
    textLists.append("rotate-45")
    imgLists.append(rotate_45_image)
    # 平移
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    translation_img = cv2.warpAffine(img, M, dsize=(img.shape[1], img.shape[0]))
    textLists.append("translation")
    imgLists.append(translation_img)
    # 顺时针旋转90
    rotate_90_iamge = cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    textLists.append("rotate-90")
    imgLists.append(rotate_90_iamge)
    # 顺时针旋转180
    rotate_180_iamge = cv2.rotate(img, cv2.ROTATE_180)
    textLists.append("rotate-180")
    imgLists.append(rotate_180_iamge)
    # 缩小一半
    resize_0_5_img=np.zeros(img.shape)
    halfImg = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    textLists.append("shrink  ")
    imgLists.append(halfImg)
    # 逆时针旋转90
    rotate_270_image = cv2.rotate(img, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    textLists.append("rotate-270")
    imgLists.append(rotate_270_image)

    # calc the moments of the image
    '''get 24 moments
    10 spatial moments 空间矩、原点矩
    m00  m10  m01 m20 m11 m02 m30 m21 m12 m03
     7 central moments 七个中心矩
    mu20 mu11 mu02 mu30 mu21 mu12 mu03
     7 central normalized moments个归一化中心矩
    nu20 nu11 nu02 nu30 nu21 nu12 nu03
    本来应该都是10个的 but
    mu00 = m00, nu00 = 1  nu10 = mu10 = mu01 = mu10 = 0, hence the values are not stored.
     原理 见论文
    论文 https://pdfs.semanticscholar.org/afc2/e9d5dfbd666bf4dd34adeb78a17393c8ee64.pdf?_ga=2.259665167.462545856.1577780532-1866022657.1577780532
    refrence https://docs.opencv.org/4.1.2/d8/d23/classcv_1_1Moments.html#a8b1b4917d1123abc3a3c16b007a7319b
    https://github.com/opencv/opencv/blob/b6a58818bb6b30a1f9d982b3f3f53228ea5a13c1/modules/imgproc/src/moments.cpp'''
    huMomentLists=[]
    for img in imgLists:
        m_=cv2.moments(img)
        hu = cv2.HuMoments(m_)
        huMomentLists.append(np.array(hu))
    # normal hu matrix
    huMomentLists = np.abs(huMomentLists)
    huMomentLists = np.log(huMomentLists)
    huMomentLists = np.abs(huMomentLists)
    return imgLists,textLists,huMomentLists

if __name__=='__main__':
    img=cv2.imread(r'.\img\describe.png')
    # img=draw_contours(img)
    # cv2.imshow("contours",img)

    imgReturn,contours=get_region_description(img)
    i=0
    for cnt in contours:
        i+=1
        print("第%d个区域信息："%i)
        print("质心：",cnt[0])
        print("方向：%f度"%cnt[1])
        print("面积：", cnt[2])
        print("周长：", cnt[3])
        print("圆形度：", cnt[4])
        print("矩形度：", cnt[5])
    cv2.imshow("result",imgReturn)
    cv2.waitKey(0)