# -*- coding: UTF-8 -*-
from globalObject import *
import tkinter
from tkinter.simpledialog import askinteger, askfloat, askstring
from tkinter.filedialog import askopenfilename, askopenfilenames, asksaveasfilename, askdirectory
from tkinter.messagebox import showinfo, showwarning, showerror,askyesno
from PIL import Image, ImageTk
import os
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PointProcessing
import Histogram
import SpatialFilter
import Fourier
import geometric
import spatialRestore
import Restore
import morphology
import segmentation
import description
import carLicense.predict as predict
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负


def cv_imread(filename,colorMode=1):
    '''
    读取文件路径中带汉字的图片,替代cv2.imread(filename)
    :param filename: 需要读取的文件
    :param colorMode: 彩色模式
    :return: 读取的图片
    '''
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), colorMode)

def cv_imwrite(path, img):
    '''
    解决路径中有汉字的图片保存问题
    :param path: 保存的路径
    :param img: 保存的图片
    '''
    cv2.imencode('.jpg', img)[1].tofile(path)

def placePic1(img_show,text1=u"原始图像"):
    '''
    将打开的初始图像放置在左边窗格
    :param img_show: 需要显示的图片
    :param text1: 图片说明
    '''
    global picSize
    if img_show is None:
        showwarning(title='警告', message='未打开图片！')
        return
    img=img_show.copy()
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height,width=img.shape[:2]
    scaling=max(width,height)/picSize
    newH = int(height / scaling)
    newW = int(width /scaling)
    img0 = Image.fromarray(img)  # 由OpenCV图片转换为PIL图片格式
    img0 = img0.resize((newW, newH))
    img0 = ImageTk.PhotoImage(img0)
    myWindow.originalText.set(text1)
    myWindow.label3.config(image=img0)
    myWindow.label3.image = img0
    myWindow.label3.place(relx=0.25, rely=0.40, width=picSize, height=picSize,anchor=tkinter.CENTER)  # 设置绝对座标

def placePic2(img_show, text2="处理结果图",RGB=True):
    '''
    将处理结果图像放置在右边窗格
    :param img_show: 需要显示的图像
    :param text2: 图像文字说明
    :param RGB: 是否转换成RGB,默认转换
    '''
    global picSize, img_result
    if img_show is None:
        showwarning(title='警告', message='没有结果图片！')
        return
    img=img_show.copy()
    if len(img.shape) > 2 and RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    img = PointProcessing.global_linear_transmation(img)
    scaling = max(width, height) / picSize
    newH = int(height / scaling)
    newW = int(width / scaling)
    img0 = Image.fromarray(img)  # 由OpenCV图片转换为PIL图片格式
    img0 = img0.resize((newW, newH))
    img0 = ImageTk.PhotoImage(img0)
    myWindow.resultText.set(text2)
    myWindow.label4.config(image=img0)
    myWindow.label4.image = img0
    myWindow.label4.place(relx=0.75, rely=0.40, width=picSize, height=picSize,anchor=tkinter.CENTER)

def Choosepic():
    '''打开文件选择对话框，并读取图片文件，存放在img_open全局变量中'''
    global picSize, path_, img_open, img_gray, img_empty
    path_ = askopenfilename(title="打开需要处理的图片")
    # path.set(path_)
    if path_ != 0 and path_ != '':
        img_open = cv_imread(path_)  # 读取文件路径中有汉字的图片文件
        img_gray = cv2.cvtColor(img_open,cv2.COLOR_BGR2GRAY)
        if img_open is not None:
            placePic1(img_open)
        else:
            placePic1(img_empty)
            showwarning(title='警告', message='无图片！')
    else:
        img_open = None
        img_gray = None
        placePic1(img_empty)
        showwarning(title='警告', message='未选择图片！')

def initWindows():
    '''将系统界面恢复到初始界面,并不清除结果图片'''
    myWindow.setVisibleLeft()
    myWindow.setVisibleRight0()
    myWindow.hideRight1()
    myWindow.setVisibleBottom()
    myWindow.hideFig0()
    myWindow.hideFig1()
    myWindow.hideExplain()
    myWindow.explainText.set("图像处理教学演示软件实现了图像处理中的一些经典算法。\n使用此软件需要先点文件-打开命令，打开一个图像，"
                             "原始图片将显示在左边窗格内;\n然后从菜单中调用某算法处理图片，处理的结果将显示在右边窗格内。")
    if img_result is not None:
        placePic2(img_result)

def Reload():
    '''将系统界面恢复到初始界面，并且清除结果图片'''
    global img_open, img_gray, img_result,img_binary ,PSF,contoursMinArea
    img_result = None  # 存放处理结果图片
    img_binary = None  # 二值图像
    PSF = None  # 退化模糊核
    contoursMinArea = None  # 最小区域面积，如果运行了目标区域轮廓模块，则此值非空
    if img_open is not None:
        img_gray = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
        placePic1(img_open)
    else:
        placePic1(img_empty)
        showwarning(title='警告', message='未打开图片！')
    initWindows()
    placePic2(img_empty, "无处理结果")

def Convert_gray():
    '''将读入的图片转换为灰度图像，并存储到img_gray'''
    global picSize, img_open, img_result, img_gray
    initWindows()
    if img_open is not None:
        # if len(img_open.shape)<3:
        #     showwarning(title='警告', message='只能对彩色图像进行色彩转换！')
        img_result = cv2.cvtColor(img_open, cv2.COLOR_BGR2GRAY)
        img_gray=img_result.copy()
        placePic2(img_result, "灰度图像")
        myWindow.explainText.set("OpenCV默认的彩色图像的颜色空间是BGR\n"
                                 "cv::cvtColor()支持多种颜色空间之间的转换\n"
                                 "例如：cv::COLOR_BGR2GRAY \n此功能是将三通道的彩色图片转换成单通道的灰度图片")
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Convert_HSV():
    '''将色彩模式转换为HSV模式'''
    global picSize, img_open, img_result
    initWindows()
    if img_open is not None:
        img_result = cv2.cvtColor(img_open, cv2.COLOR_BGR2HSV)
        placePic2(img_result, "HSV彩色图",False)
        myWindow.explainText.set("HSV彩色模型：H(Hue)色度 、S(Saturation)饱合度、V(Value)亮度\n" \
               "RGB彩色模式转HSV彩色模式方法：V=max(R,G,B)\nS=(V-min(R,G,B))*255/V   if V!=0, \nS=0                     otherwise\nH=(G - B)*60/S,        if V=R\nH= 180+(B - R)*60/S, if V=G   \nH=240+(R - G)*60/S,    if V=B\n若 H<0，则 H=H+360")
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Convert_BGR():
    '''将色彩模式转换为BGR模式'''
    global picSize, img_open, img_result
    initWindows()
    if img_open is not None:
        img_result=img_open
        placePic2(img_result,"BGR彩色图像",False)
        myWindow.explainText.set("OpenCV默认的彩色图像的颜色空间是BGR\n"
                                 "cv::cvtColor()支持多种颜色空间之间的转换\n"
                                 "例如：cv::COLOR_BGR2RGB \n cv::COLOR_RGB2BGR \n cv::COLOR_RGBA2BGRA \n cv::COLOR_BGRA2RGBA")
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Savepic():
    '''打开文件对话框，将img_result图片保存到指定文件'''
    global img_result
    if img_result is not None:
        fname = asksaveasfilename(title=u'保存文件', filetypes=[("JPG", ".jpg")])
        if fname:
            cv_imwrite(str(fname), img_result)
            showinfo(title='提示', message='图片已保存！')
        else:
            showwarning(title='警告', message='请先输入文件名！')
    else:
        showwarning(title='警告', message='请先处理图片！')
#---------------------------------------------------------
def Negative():
    '''    对灰度图像反转    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        placePic1(img_gray, "原灰度图像")
        img_result = PointProcessing.negative(img_gray.copy())
        placePic2(img_result, "反转图像")
        myWindow.explainText.set(" 图像反转用于增强图像暗色区域的白色或灰色细节，适用于黑色较多的图像增强。\n "
                                 "图像反转变换的表达式为： s=(L-1)-r \n 其中，L为图像的灰度级,r为输入灰度值，s为输出灰度值。 ")
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Global_linear_transmation():
    '''    灰度图像全局线性变换    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        placePic1(img_gray, "原灰度图像")
        img_result = PointProcessing.global_linear_transmation(img_gray.copy())
        placePic2(img_result, "全局灰度线性变换图像")
        imgExplain = cv2.imread(r'.\explain\global_linear.jpg', 0)
        text="全局线性变换中，输入图像与输出图像灰度值之间的关系式：\n s = (r-a)*(d-c)/(b-a)+c \n " \
             "\n 本算法是将图像的灰度区间变换为[0,255]"
        picLists = [["输入灰度值与输出灰度值之间的关系", imgExplain]]
        myWindow.showExplain(text,picLists)
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Piecewise_linear_transformation():
    '''灰度图像分段线性变换，可由用户设置分段数以及灰度值范围'''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        imgExplain = cv2.imread(r'.\explain\piece_trans.jpg', 0)
        text="分段灰度线性变换可以将图像的灰度值分成若干段，如图所示。\n各段采用不同的线性变换，分成三段的数学表达式如下：\n" \
             "s=(c/a)*r                if 0<=r<a\n" \
             "s=(d-c)/(b-a)*(r-a)+c    if a<=r<b\n" \
             "s=(mg-d)/(mf-b)*(r-b)+d  if b<=r<mf\n"
        picLists = [["分段灰度线性变换图示如下：", imgExplain]]
        myWindow.showExplain(text,picLists)
        placePic1(img_gray, "原灰度图像")
        n_int = askinteger(title="请输入分段数",prompt = "分段数：",initialvalue=3)
        if n_int:
            grayLists0 = [[0, 80, 0, 50], [80, 150, 50, 200], [150, 255, 200, 255]]  # 初始默认为3段
            grayLists = n_int*[[0,255,0,255]]    #给分段线性变换的初始参数赋值
            if len(grayLists0)>=n_int:
                grayLists = grayLists0[0:n_int]
            else:
                grayLists[0:3] = grayLists0
            grayLists[0][0]=grayLists[0][2]=0          #限定灰度值在0-255
            grayLists[n_int-1][1] = grayLists[n_int-1][3] = 255
            paraW=paraWindow2(root,grayLists,"分段线性变换参数设置","变换前灰度区间     变换后灰度区间")
            img_result = PointProcessing.piecewise_linear_transformation(img_gray.copy(),paraW.paraLists)
            placePic2(img_result, "分段灰度线性变换图像")
        else:
            showwarning(title='警告', message='请先输入分段数！')
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Logarithmic_transformations():
    '''    灰度图像对数变换    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        imgExplain = cv2.imread(r'.\explain\log_trans.jpg', 0)
        text="对数变换是一种非线性的灰度变换。\n对数变换的表达式如下：\ns=a+c×log（r+1）\n此对数是以自然数e为底"
        picLists = [["对数变换的输入图像与输出图像之间关系如图所示：", imgExplain]]
        myWindow.showExplain(text,picLists)
        placePic1(img_gray, "原灰度图像")
        paraC=askfloat("参数设置","s=C*log(1+r),请输入对数变换常数C",initialvalue=20.0)
        img_result = PointProcessing.logarithmic_transformations(img_gray.copy(),paraC)
        placePic2(img_result, "灰度对数变换图像")

    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Power_law_transformations():
    '''    灰度图像幂次变换    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        myWindow.explainText.set("幂次变换的表达式如下：S=c×(r^γ)；\n"
                                 "其中，c取默认值1，幂次γ是可调的参数。\n"
                                 "幂次变换随着幂次γ的不同，可以得到不同的增强效果。\n"
                                 "当γ<1时，幂次变换相当于对数变换，此非线性变换可以扩展图像中的暗区，而压缩亮区；\n"
                                 "当γ>1时，幂次变换的效果与对数变换的效果相反，此非线性变换可以扩展图像中的亮区，而压缩暗区。")
        placePic1(img_gray, "原灰度图像")
        paraGamma = askfloat("参数设置", "s=r^γ"+",请输入幂次γ", initialvalue=1.0)
        img_result = PointProcessing.power_law_transformations(img_gray.copy(), paraGamma)
        placePic2(img_result, "灰度幂次变换图像")
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Gray_histogram():
    '''    灰度图像直方图    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        myWindow.explainText.set("直方图是图像的灰度值统计图，\n"
                                 "即对于每个灰度值，统计在图像中具有该灰度值的像素个数或灰度值出现的概率，并绘制而成的图形称为灰度直方图（简称直方图）。\n直方图反映了图像的灰度分布范围等特征，在很多场合下，往往包含了图像的重要信息。 ")
        binSize = askinteger(title="直方图参数设置", prompt="请输入直方图的灰度级数", initialvalue=256)
        placePic1(img_gray, "原灰度图像")
        histr = Histogram.gray_histogram(img_gray.copy(),binSize)
        myWindow.myFig0.clear()
        f_plot = myWindow.myFig0.add_subplot(111)
        f_plot.bar(np.arange(binSize),histr.flatten(),color='blue')
        myWindow.setVisibleFig0()
        myWindow.resultText.set("灰度直方图")
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Mask_histogram():
    '''灰度图像掩膜直方图，可以由用户选择掩膜图像'''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        placePic1(img_gray, "原灰度图像")
        myWindow.explainText.set(
            "掩膜直方图可以统计图像某些局部区域的直方图。"
            "\n要生成掩膜直方图需要提供一副掩膜图像,将要统计的部分设置为白色，其余部分为黑色。"
            "\n在生成直方图时，把这个掩膜图像与原图像做位与运算。 ")
        binSize = askinteger(title="直方图参数设置", prompt="请输入直方图的灰度级数", initialvalue=256)
        pathMask = askopenfilename(title="打开掩膜二值图像")
        if pathMask != 0 and pathMask != '':
            imgMask = cv_imread(pathMask,0)
            histr,masked_img = Histogram.mask_histogram(img_gray.copy(), imgMask, binSize)
            myWindow.showAndBarFig0(masked_img,histr)
            myWindow.resultText.set("掩膜直方图")
        else:
            showwarning(title='警告', message='请先打开掩膜二值图片！')
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def Equalization_histogram():
    '''    直方图均衡化    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        placePic1(img_gray, "原灰度图像")
        histr,equ = Histogram.equalization_histogram(img_gray.copy())
        myWindow.showAndBarFig0(equ,histr)
        myWindow.resultText.set("直方图均衡化")
        myWindow.explainText.set('''直方图均衡化是一种常用的灰度增强算法，也称为直方图均匀化。它是将原图像的直方图经过变换函数修整为均匀直方图，使得图像的灰度分布趋向均匀。 ''')
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')

def SML():
    '''    直方图规定化——单映射    '''
    global picSize, img_gray, img_result
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    initWindows()
    placePic1(img_gray, "原灰度图像")
    #显示目标直方图
    dst = [[], [], []]
    dst[0]=np.int32([a for a in range(256)])
    dst[1]= np.int32([256 - j for j in range(256)])
    dst[2] = np.zeros((256,), np.int32)
    dst[2][:] = 128
    picLists=[["1-目标直方图是正三角形",dst[0]],["2-目标直方图为倒三角形",dst[1]],["3-目标直方图为水平直线",dst[2]]]
    myWindow.showFig1(picLists,type='bar')
    dest=["正三角形","倒三角形","均匀分布"]
    typeHistgram=askinteger("选择目标直方图","请选择目标直方图，1=正三角形，2-倒三角形，3=均匀分布", initialvalue=1,minvalue=1,maxvalue=3)
    if typeHistgram==1 or typeHistgram==2 or typeHistgram==3:
        histr, equ = Histogram.regulation_histogram(img_gray.copy(),typeHistgram,True)
        myWindow.showAndBarFig0(equ,histr)
        myWindow.resultText.set("单映射"+dest[typeHistgram-1]+"规定化")
    else:
        showwarning(title='警告', message='请正确选择目标直方图！')

def GML():
    '''    直方图规定化——组映射    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is not None:
        placePic1(img_gray, "原灰度图像")
        # 显示目标直方图
        dst = [[], [], []]
        dst[0] = np.zeros((256,), np.int32)  # 规定直方图
        dst[1] = np.zeros((256,), np.int32)
        dst[2] = np.zeros((256,), np.int32)
        for j in range(256):
            dst[0][j] = j
            dst[1][j] = 256 - j
            dst[2][j] = 128
        picLists = [["1-目标直方图是正三角形", dst[0]], ["2-目标直方图为倒三角形", dst[1]], ["3-目标直方图为水平直线", dst[2]]]
        myWindow.showFig1(picLists, type='bar')
        dest = ["正三角形", "倒三角形", "均匀分布"]
        typeHistgram = askinteger("选择目标直方图", "请选择目标直方图，1=正三角形，2=倒三角形，3=均匀分布", initialvalue=1)
        if typeHistgram == 1 or typeHistgram == 2 or typeHistgram == 3:
            histr, equ = Histogram.regulation_histogram(img_gray.copy(), typeHistgram, False)
            myWindow.showAndBarFig0(equ,histr)
            myWindow.resultText.set("组映射"+dest[typeHistgram-1]+"规定化")
        else:
            showwarning(title='警告', message='请正确选择目标直方图！')
    else:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
#---------------------------------------------------------
def Gaussian_filter():
    '''    高斯平滑滤波    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img = cv2.imread(r'.\explain\Gaussian_weight.jpg')
    imgExplain= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = "高斯滤波是比较经典的加权的平滑滤波,适用于消除高斯噪声。\n"\
        "高斯滤波运用了高斯的正态分布的密度函数来计算模板上的权重。" \
           "正态分布是一种钟形曲线，越接近中心，取值越大，越远离中心，取值越小，如图所示"
    picLists = [["滤波核权重按正态分布", imgExplain]]
    myWindow.showExplain(text, picLists)
    placePic1(img_gray, "待处理灰度图像")
    kernalSize = askinteger("设置滤波核大小", "请输入高斯滤波核的大小（通常为奇数）",initialvalue=3)
    if kernalSize and kernalSize%2==1:
        img_result = SpatialFilter.gaussian_filter(img_gray.copy(), kernalSize)
        placePic2(img_result,"核大小为"+str(kernalSize)+"的高斯滤波")
    else:
        showwarning(title='警告', message='请先设置滤波核大小')

def Median_filter():
    '''    中值滤波    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    if img_result is not None:
        answer = askyesno("选择待滤波图像", "Y-对打开图像进行滤波，N-对处理后图像进行滤波")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    myWindow.explainText.set("中值滤波的方法是：对邻域内的像素灰度值按大小排序，然后取中间位置的灰度值取代当前像素的灰度值。"\
                             "\n中值滤波对滤除脉冲干扰及图像扫描噪声最为有效，它可以滤除小于1/2窗口的脉冲信号。")
    placePic1(img_, "原灰度图像")
    kernalSize = askinteger("设置滤波核", "请输入中值滤波核的大小（通常为奇数）",initialvalue=3)
    if kernalSize and kernalSize%2==1:
        img_result=SpatialFilter.median_filter(img_.copy(),kernalSize)
        placePic2(img_result)
        myWindow.resultText.set("中值滤波")
    else:
        showwarning(title='警告', message='请先设置滤波核大小,且滤波核大小应为奇数')

def Max_filter():
    '''    最大值滤波    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    if img_result is not None:
        answer = askyesno("选择待滤波图像", "Y-对打开图像进行滤波，N-对处理后图像进行滤波")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    myWindow.explainText.set("最大值滤波的方法是：对邻域内的像素灰度值按大小排序，然后取最大的灰度值取代当前像素的灰度值。"\
                             "\n最大值滤波适合去除胡椒噪声。")
    placePic1(img_, "原灰度图像")
    kernalSize = askinteger("设置滤波核", "请输入最大值滤波核的大小",initialvalue=3)
    if kernalSize:
        img_result=SpatialFilter.max_filter(img_.copy(),kernalSize)
        placePic2(img_result)
        myWindow.resultText.set("最大值滤波")
    else:
        showwarning(title='警告', message='请先设置滤波核大小！')

def Min_filter():
    '''    最小值滤波    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    if img_result is not None:
        answer = askyesno("选择待滤波图像", "Y-对打开图像进行滤波，N-对处理后图像进行滤波")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    myWindow.explainText.set("最小值滤波的方法是：对邻域内的像素灰度值按大小排序，然后取最小的灰度值取代当前像素的灰度值。"\
                             "\n最小值滤波适合去除盐噪声。")
    placePic1(img_, "原灰度图像")
    kernalSize = askinteger("设置滤波核", "请输入最小值滤波核的大小",initialvalue=3)
    if kernalSize:
        img_result=SpatialFilter.min_filter(img_.copy(),kernalSize)
        placePic2(img_result)
        myWindow.resultText.set("最小值滤波")
    else:
        showwarning(title='警告', message='请先设置滤波核大小！')

def Smooth_filter():
    '''自定义平滑滤波，由用户定义滤波核权重'''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    if img_result is not None:
        answer = askyesno("选择待滤波图像", "Y-对打开图像进行滤波，N-对处理后图像进行滤波")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    myWindow.explainText.set("自定义滤波可以由用户定义滤波核的大小和权重，\n并根据定义的滤波器进行图像的滤波处理。")
    placePic1(img_, "原灰度图像")
    kernalSize = askinteger("设置滤波核", "请输入滤波核的大小（通常为奇数）",initialvalue=3)
    if kernalSize and kernalSize%2==1:
        temp=[kernalSize*[1]]
        paraLists=kernalSize*temp
        paraW = paraWindow2(root,paraLists,"滤波核设置","请设置"+str(kernalSize)+"x"+str(kernalSize)+"滤波核的权重")
        paraLists=paraW.paraLists/np.sum(paraW.paraLists)#滤波核的权重和为1
        showinfo(title='提示', message="已调整滤波核，让平滑滤波核的权重和为1")
        img_result=SpatialFilter.smooth_filter(img_.copy(),paraLists)
        placePic2(img_result,"自定义平滑滤波")
    else:
        showwarning(title='警告', message='请先设置滤波核大小,且滤波核大小应为奇数')
#---------------------------------------------------------
def Constant_border():
    '''    图片扩展，常量法，边界补0    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    myWindow.hideFig1()  # 隐藏画布
    myWindow.explainText.set('''BORDER_CONSTANT：常量，增加的变量值为value色 [value][value] | abcdef | [value][value]
BORDER_REFLICATE:直接用边界的颜色填充， aaaaaa | abcdefg | gggg 
BORDER_REFLECT:镜像，abcdefg | gfedcbamn | nmabcd
BORDER_REFLECT_101:镜像，镜像时，会把边界空开，abcdefg | egfedcbamne | nmabcd
BORDER_WRAP:环绕 类似于这种方式abcdf | mmabcdf | mmabcd
''')
    placePic1(img_gray, "原灰度图像")
    borderSize = askinteger("设置边框宽度", "请输入增加的边框宽度", initialvalue=10)
    if borderSize:
        img_result = SpatialFilter.border_fill(img_gray.copy(), cv2.BORDER_CONSTANT, borderSize)
        placePic2(img_result)
        myWindow.resultText.set("边界补0后的图像")
    else:
        showwarning(title='警告', message='请先输入增加的边框宽度')

def Replicate_border():
    '''    图片扩展，复制法    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    myWindow.hideFig1()  # 隐藏画布
    myWindow.explainText.set('''BORDER_CONSTANT：常量，增加的变量值为value色 [value][value] | abcdef | [value][value]
BORDER_REFLICATE:直接用边界的颜色填充， aaaaaa | abcdefg | gggg 
BORDER_REFLECT:镜像，abcdefg | gfedcbamn | nmabcd
BORDER_REFLECT_101:镜像，镜像时，会把边界空开，abcdefg | egfedcbamne | nmabcd
BORDER_WRAP:环绕 类似于这种方式abcdf | mmabcdf | mmabcd
''')
    placePic1(img_gray, "原灰度图像")
    borderSize = askinteger("设置边框宽度", "请输入增加的边框宽度", initialvalue=10)
    if borderSize:
        img_result = SpatialFilter.border_fill(img_gray.copy(), cv2.BORDER_REPLICATE, borderSize)
        placePic2(img_result)
        myWindow.resultText.set("复制边界后的图像")
    else:
        showwarning(title='警告', message='请先输入增加的边框宽度')

def Reflect_border():
    '''    图片扩展，反射法    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    myWindow.hideFig1()  # 隐藏画布
    myWindow.explainText.set('''BORDER_CONSTANT：常量，增加的变量值为value色 [value][value] | abcdef | [value][value]
BORDER_REFLICATE:直接用边界的颜色填充， aaaaaa | abcdefg | gggg 
BORDER_REFLECT:镜像，abcdefg | gfedcbamn | nmabcd
BORDER_REFLECT_101:镜像，镜像时，会把边界空开，abcdefg | egfedcbamne | nmabcd
BORDER_WRAP:环绕 类似于这种方式abcdf | mmabcdf | mmabcd
''')
    placePic1(img_gray, "原灰度图像")
    borderSize = askinteger("设置边框宽度", "请输入增加的边框宽度", initialvalue=10)
    if borderSize:
        img_result = SpatialFilter.border_fill(img_gray.copy(), cv2.BORDER_REFLECT, borderSize)
        placePic2(img_result)
        myWindow.resultText.set("边界镜像处理")
    else:
        showwarning(title='警告', message='请先输入增加的边框宽度')

def Reflect_border_101():
    '''    图片扩展，反射法101    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    myWindow.hideFig1()  # 隐藏画布
    myWindow.explainText.set('''BORDER_CONSTANT：常量，增加的变量值为value色 [value][value] | abcdef | [value][value]
BORDER_REFLICATE:直接用边界的颜色填充， aaaaaa | abcdefg | gggg 
BORDER_REFLECT:镜像，abcdefg | gfedcbamn | nmabcd
BORDER_REFLECT_101:镜像，镜像时，会把边界空开，abcdefg | egfedcbamne | nmabcd
BORDER_WRAP:环绕 类似于这种方式abcdf | mmabcdf | mmabcd
''')
    placePic1(img_gray, "原灰度图像")
    borderSize = askinteger("设置边框宽度", "请输入增加的边框宽度", initialvalue=10)
    if borderSize:
        img_result = SpatialFilter.border_fill(img_gray.copy(), cv2.BORDER_REFLECT101, borderSize)
        placePic2(img_result)
        myWindow.resultText.set("边界镜像_101")
    else:
        showwarning(title='警告', message='请先输入增加的边框宽度')

def Wrap_border():
    '''    图片扩展，外包装法    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    myWindow.hideFig1()  # 隐藏画布
    myWindow.explainText.set('''BORDER_CONSTANT：常量，增加的变量值为value色 [value][value] | abcdef | [value][value]
BORDER_REFLICATE:直接用边界的颜色填充， aaaaaa | abcdefg | gggg 
BORDER_REFLECT:镜像，abcdefg | gfedcbamn | nmabcd
BORDER_REFLECT_101:镜像，镜像时，会把边界空开，abcdefg | egfedcbamne | nmabcd
BORDER_WRAP:环绕 类似于这种方式abcdf | mmabcdf | mmabcd
''')
    placePic1(img_gray, "原灰度图像")
    borderSize = askinteger("设置边框宽度", "请输入增加的边框宽度", initialvalue=10)
    if borderSize:
        img_result = SpatialFilter.border_fill(img_gray.copy(), cv2.BORDER_WRAP, borderSize)
        placePic2(img_result)
        myWindow.resultText.set("边界环绕处理")
    else:
        showwarning(title='警告', message='请先输入增加的边框宽度')
#---------------------------------------------------------
def Roberts_filter():
    '''    Roberts算子    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain=cv2.imread(r'.\explain\Roberts.jpg',0)
    text="Roberts算子能较好的增强正负45度的图像边缘。"
    picLists=[["Roberts算子模板",imgExplain]]
    myWindow.showExplain(text, picLists)
    placePic1(img_gray, "原灰度图像")
    img_result = SpatialFilter.roberts_filter(img_gray.copy())
    placePic2(img_result)
    myWindow.resultText.set("Roberts锐化")

def Sobel_filter():
    '''    sobel算子    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\sobel.jpg', 0)
    text="Sobel算子给靠近中心的像素以较大的权重，使锐化的同时还有一定的平滑作用。"
    picLists = [["Sobel算子模板", imgExplain]]
    myWindow.showExplain(text,picLists)
    placePic1(img_gray, "原灰度图像")
    img_result = SpatialFilter.sobel_filter(img_gray.copy())
    placePic2(img_result)
    myWindow.resultText.set("Sobel锐化")

def Scharr_filter():
    '''    Scharr算子    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\scharr.jpg', 0)
    text="Scharr算子是对Sobel算子差异性的增强，Scharr算子的主要思路是通过将模版中的权重系数放大来增大像素值间的差异"
    picLists = [["Scharr算子模板", imgExplain]]
    myWindow.showExplain(text,picLists)
    placePic1(img_gray, "原灰度图像")
    img_result = SpatialFilter.scharr_filter(img_gray.copy())
    placePic2(img_result)
    myWindow.resultText.set("Scharr锐化")

def Prewitt_filter():
    '''    Prewitt算子    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\prewitt.jpg', 0)
    text="Prewitt算子将模板大小从2x2扩大到3x3,使滤波操作在锐化图像边缘的同时能减少噪声的影响。"
    picLists = [["Prewitt算子模板", imgExplain]]
    myWindow.showExplain(text,picLists)
    placePic1(img_gray, "原灰度图像")
    img_result = SpatialFilter.prewitt_filter(img_gray.copy())
    placePic2(img_result)
    myWindow.resultText.set("Prewitt锐化")

def Laplacian_filter():
    '''    Laplacian算子    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain1 = cv2.imread(r'.\explain\laplacian1.jpg', 0)
    imgExplain2 = cv2.imread(r'.\explain\laplacian2.jpg', 0)
    picLists = [["Laplacian是二阶微分算子，以下为四邻域模板", imgExplain1],["Laplacian的八邻域模板具有方向同性", imgExplain2]]
    myWindow.showFig1(picLists)
    placePic1(img_gray, "原灰度图像")
    select=askinteger("模板选择","请选择1--四邻域,2--八邻域", initialvalue=1)
    if select==1:
        img_result = SpatialFilter.laplacian_filter(img_gray.copy(),1)
    elif select==2:
        img_result = SpatialFilter.laplacian_filter(img_gray.copy(), 2)
    else:
        showwarning(title='警告', message='必须在两个模板中选择一个！')
        return
    placePic2(img_result)
    myWindow.resultText.set("laplacian锐化")

def Log_filter():
    '''    log算子    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\log.jpg', 0)
    text="LOG算子实际上是把 Gauss滤波和Laplacian滤波结合了起来，先平滑掉噪声，再进行边缘检测。"
    picLists = [["5x5的LOG算子模板", imgExplain]]
    myWindow.showExplain(text,picLists)
    placePic1(img_gray, "原灰度图像")
    img_result = SpatialFilter.log_filter(img_gray.copy())
    placePic2(img_result,"LOG锐化")

def Canny_filter():
    '''    Canny算子    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    myWindow.hideFig1() #隐藏画布
    myWindow.explainText.set('''Canny边缘检测算子是一种多级检测算法。1986年由John　F.Canny提出，同时提出了边缘检测的三大准则：
•低错误率的边缘检测：检测算法应该精确地找到图像中的尽可能多的边缘，尽可能的减少漏检和误检。
•最优定位：检测的边缘点应该精确地定位于边缘的中心。
•图像中的任意边缘应该只被标记一次，同时图像噪声不应产生伪边缘。
''')
    placePic1(img_gray, "原灰度图像")
    img_result = SpatialFilter.canny_filter(img_gray.copy())
    placePic2(img_result)
    myWindow.resultText.set("canny锐化")
#---------------------------------------------------------
def Magnitude_spectrum0():
    '''    幅值谱    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain=cv2.imread(r'.\explain\magnitude.jpg',0)
    text="为了提高幅值谱的可视化程度，对幅值谱按公式y=20*log(1+|F(u,v)|)进行了非线性变换"
    picLists = [["幅值谱计算公式", imgExplain]]
    myWindow.showExplain(text,picLists)
    placePic1(img_gray, "原灰度图像")
    dft = Fourier.fft(img_gray)
    img_result=Fourier.fft_magnitude(dft)
    placePic2(img_result,"高频居中的幅值谱")

def Magnitude_spectrum1():
    '''    低频移中的幅值谱    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\translation.jpg', 0)
    text="图像傅里叶变换的高频区是在中心点附近，而低频区在四个角。\n" \
         "但人们通常习惯将低频信号集中在中心点附近，为此我们需要对傅里叶变换进行平移。" \
         "\n傅里叶变换的平移性：频域的平移相当于空域中乘以一个复指数\n" \
         "因此在空域将像素点乘以(-1)^(x+y),可以实现低频移中"
    picLists = [["傅里叶变换的平移性", imgExplain]]
    myWindow.showExplain(text,picLists)
    placePic1(img_gray, "原灰度图像")
    dft = Fourier.fft(img_gray,True)
    img_result=Fourier.fft_magnitude(dft)
    placePic2(img_result,"低频移中的幅值谱")

def Phase_spectrum0():
    '''    相位谱    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\phase.jpg', 0)
    text="傅里叶变换的幅值谱是对信号轮廓和形状的描述，而相位谱是对信号位置的描述。\n" \
         "不同位置相同形状的信号，其幅值谱相同，而相位谱不同"
    picLists = [["相位谱计算公式：", imgExplain]]
    myWindow.showExplain(text,picLists)
    placePic1(img_gray, "原灰度图像")
    dft = Fourier.fft(img_gray)
    img_result=Fourier.fft_phase(dft)
    placePic2(img_result,"高频居中的相位谱")

def Phase_spectrum1():
    '''    低频移中的相位谱    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\phase.jpg', 0)
    text="傅里叶变换的幅值谱是对信号轮廓和形状的描述，而相位谱是对信号位置的描述。\n" \
         "不同位置相同形状的信号，其幅值谱相同，而相位谱不同"
    picLists = [["相位谱计算公式：", imgExplain]]
    myWindow.showExplain(text,picLists)
    placePic1(img_gray, "原灰度图像")
    dft = Fourier.fft(img_gray,True)
    img_result=Fourier.fft_phase(dft)
    placePic2(img_result,"低频移中的相位谱")

def Inverse_fourier():
    '''    傅里叶反变换    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    placePic1(img_gray, "原灰度图像")
    dft=Fourier.fft(img_gray.copy())
    img_result=Fourier.ifft(dft) #　由于傅里叶变换期间，使用了最优尺寸，反变换时需要恢复原图像大小
    placePic2(img_result,"傅里叶反变换")
    myWindow.explainText.set("此操作是先对图像进行傅里叶变换，再进行逆变换。\n"\
                             "根据图像与其傅里叶变换之间的一一对应关系，重构的图像应该与原图像相同。\n"\
                             "但由于傅里叶变换期间，根据快速傅里叶变换要求，对图像大小进行了优化，\n"\
                             "反变换后生成图像大小可能会有微小变化")
def Property_fourier():
    '''    添加周期噪声    '''
    global picSize, img_gray, img_result
    initWindows()
    myWindow.explainText.set("所添加的噪声频点的位置决定了条纹的宽度和方向。\n" \
                             "噪声频点应该是圆点对称的，加圆圈标注。\n" \
                             "噪声频点越靠近中心，噪声频率越低，条纹越宽;反之条纹越细。\n" \
                             "若在某方向上添加了噪声频点，意味着在这个方向上灰度变化较大，条纹是垂直于这个方向。")
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    # 将图片大小变为奇数的，这样有图片中心
    m, n = img_gray.shape
    if m%2==0:
        m=m+1
    if n%2==0:
        n=n+1
    img = cv2.resize(img_gray, (m, n))
    F = np.fft.fft2(img)
    fshift0 = np.fft.fftshift(F)
    G = fshift0.copy()
    fmax = np.max(np.abs(fshift0))
    fmin = np.min(np.abs(fshift0))
    # 图片中心为u0,v0
    u0, v0 = (m - 1) // 2, (n - 1) // 2
    paraLists = [["孤立频点距离中心的水平距离", -10], ["孤立频点距离中心的垂直距离", 10]]
    paraW = paraWindow(root, "请设置孤立的噪声频点在频谱上的位置", paraLists)
    r1 = int(paraW.paraLists[0][1])
    r2 = int(paraW.paraLists[1][1])
    u1 = u0 + r1
    v1 = v0 + r2
    G[v1, u1] = fmax / 5
    u2 = m - 1 - u1
    v2 = n - 1 - v1
    G[v2, u2] = fmax / 5
    f1 = np.fft.ifftshift(G)
    img1 = abs(np.fft.ifft2(f1))  # 重构图像
    img0 = 20 * np.log(1 + abs(G))
    # img0=(abs(G)-fmin)/(fmax-fmin)*255
    cv2.circle(img0, (u1, v1), 3, (0, 0, 0), 1)
    cv2.circle(img0, (u2, v2), 3, (0, 0, 0), 1)
    placePic1(img0, "添加了孤立的噪声频点的幅值谱")
    # img1=img1.astype(np.uint8)
    # img2=cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    # font=cv2.QT_FONT_NORMAL
    # cv2.putText(img2, "注意条纹的宽度和方向！！！", (300, 300), font, 1.5, (255, 255, 255), 2)
    placePic2(img1, "带周期噪声的图像\n注意孤立的噪声频点与条纹的宽度、方向的关系！")

def Idle_low_pass():
    '''    理想低通滤波器    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain0 = cv2.imread(r'.\explain\ILPF0.jpg', 0)
    imgExplain1 = cv2.imread(r'.\explain\ILPF.jpg', 0)
    picLists = [["理想低通滤波器是将小于截止频率的分量完全\n无损地通过，而大于截止频率的分量完全衰减。", imgExplain0],["理想低通滤波器示意图", imgExplain1]]
    myWindow.showFig1(picLists)
    placePic1(img_gray, "原灰度图像")
    row,col=img_gray.shape
    maxD0=min(row,col)//2
    d0 = askinteger("理想低通滤波器参数设置","请设置截止频率，其值在["+str(1)+'~'+str(maxD0)+']之间',initialvalue=(1+maxD0)//2)
    if d0 and maxD0>=d0>=1:
        img_result = Fourier.lpfilter(img_gray, 0, d0)
        placePic2(img_result, "理想的低通滤波")
    else:
        showwarning("警告","截止频率的值应该在["+str(1)+'~'+str(maxD0)+']之间')

def Butterworth_low_pass():
    '''    巴特沃兹低通滤波器    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain1 = cv2.imread(r'.\explain\Butterworth0.jpg', 0)
    imgExplain2 = cv2.imread(r'.\explain\Butterworth1.jpg', 0)
    picLists = [["巴特沃斯低通滤波器的通带与阻带之间没有明显的\n不连续性。n阶巴特沃斯低通滤波器的传递函数为：", imgExplain1],["巴特沃斯低通滤波器的示意图", imgExplain2]]
    myWindow.showFig1(picLists)
    placePic1(img_gray, "原灰度图像")
    row,col=img_gray.shape
    maxD0=min(row,col)//2
    d0 = askinteger("巴特沃兹低通滤波器参数设置","请设置截止频率，其值在["+str(1)+'~'+str(maxD0)+']之间',initialvalue=(1+maxD0)//2)
    if d0 is None or d0>maxD0 or d0<10:
        showwarning("警告","截止频率的值应该在["+str(10)+'~'+str(maxD0)+']之间')
        return
    nj = askinteger("巴特沃兹低通滤波器参数设置","请设置巴特沃兹低通滤波器的阶数，其值在[1~20]之间的整数",initialvalue=1)
    if nj is None or nj>5 or nj<1:
        showwarning("警告","巴特沃兹低通滤波器阶数应该在[1~5]之间")
        return
    img_result = Fourier.lpfilter(img_gray, 1, d0,nj)
    placePic2(img_result, "巴特沃兹低通滤波")

def Gaussian_low_pass():
    '''    高斯低通滤波器    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain1 = cv2.imread(r'.\explain\Gaussian0.jpg', 0)
    imgExplain2 = cv2.imread(r'.\explain\Gaussian1.jpg', 0)
    picLists = [["高斯低通滤波器在截止频率较小时，具有较陡峭的频率特征，\n随着截止频率的增大，高斯低通滤波器趋于平缓。", imgExplain1],["高斯低通滤波器的示意图", imgExplain2]]
    myWindow.showFig1(picLists)
    placePic1(img_gray, "原灰度图像")
    row, col = img_gray.shape
    maxD0 = min(row, col) // 4
    d0 = askinteger("高斯低通滤波器参数设置", "请设置截止频率，其值在[" + str(10) + '~' + str(maxD0) + ']之间',initialvalue=(10+maxD0)//2)
    if d0 is None or d0 > maxD0 or d0 < 10:
        showwarning("警告", "截止频率的值应该在[" + str(10) + '~' + str(maxD0) + ']之间')
        return
    img_result = Fourier.lpfilter(img_gray, 2, d0)
    placePic2(img_result, "高斯低通滤波")

def Idle_high_pass():
    '''    理想高通滤波器    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain0=cv2.imread(r'.\explain\Ideal_High0.jpg',0)
    imgExplain1 = cv2.imread(r'.\explain\Ideal_High1.jpg', 0)
    picLists = [["理想高通滤波器是将大于截止频率的分量完全\n无损地通过，而小于截止频率的分量完全衰减。", imgExplain0],["理想高通滤波器示意图", imgExplain1]]
    myWindow.showFig1(picLists)
    placePic1(img_gray, "原灰度图像")
    row,col=img_gray.shape
    maxD0=min(row,col)//4
    d0 = askinteger("理想高通滤波器参数设置","请设置截止频率，其值在["+str(1)+'~'+str(maxD0)+']之间',initialvalue=(10+maxD0)//2)
    if d0 and maxD0>=d0>=1:
        img_result = Fourier.hpfilter(img_gray, 0, d0)
        placePic2(img_result, "理想的高通滤波")
    else:
        showwarning("警告","截止频率的值应该在["+str(1)+'~'+str(maxD0)+']之间')

def Butterworth_high_pass():
    '''    巴特沃兹高通滤波器    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain1 = cv2.imread(r'.\explain\Butterworth_high0.jpg', 0)
    imgExplain2 = cv2.imread(r'.\explain\Butterworth_high1.jpg', 0)
    picLists = [["巴特沃兹高通滤波器的传递函数为：", imgExplain1],["巴特沃斯高通滤波器的示意图", imgExplain2]]
    myWindow.showFig1(picLists)
    placePic1(img_gray, "原灰度图像")
    row,col=img_gray.shape
    maxD0=min(row,col)//4
    d0 = askinteger("巴特沃兹高通滤波器参数设置","请设置截止频率，其值在["+str(1)+'~'+str(maxD0)+']之间',initialvalue=(10+maxD0)//2)
    if d0 is None or d0>maxD0 or d0<1:
        showwarning("警告","截止频率的值应该在["+str(1)+'~'+str(maxD0)+']之间')
        return
    nj = askinteger("巴特沃兹高通滤波器参数设置","请设置巴特沃兹低通滤波器的阶数，其值在[1~5]之间",initialvalue=1)
    if nj is None or nj>5 or nj<1:
        showwarning("警告","巴特沃兹高通滤波器阶数应该在[1~5]之间")
        return
    img_result = Fourier.hpfilter(img_gray, 1, d0,nj)
    placePic2(img_result, "巴特沃兹高通滤波")

def Gaussian_high_pass():
    '''    高斯高通滤波器    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain1 = cv2.imread(r'.\explain\Gaussian_high0.jpg', 0)
    imgExplain2 = cv2.imread(r'.\explain\Gaussian_high1.jpg', 0)
    picLists = [["高斯高通滤波器传递函数为：", imgExplain1],["高斯高通滤波器的示意图", imgExplain2]]
    myWindow.showFig1(picLists)
    placePic1(img_gray, "原灰度图像")
    row, col = img_gray.shape
    maxD0 = min(row, col) // 4
    d0 = askinteger("高斯高通滤波器参数设置", "请设置截止频率，其值在[" + str(10) + '~' + str(maxD0) + ']之间',initialvalue=(10+maxD0)//2)
    if d0 is None or d0 > maxD0 or d0 < 10:
        showwarning("警告", "截止频率的值应该在[" + str(10) + '~' + str(maxD0) + ']之间')
        return
    img_result = Fourier.hpfilter(img_gray, 2, d0)
    placePic2(img_result, "高斯高通滤波")
#--------------------------------------------------------
def Move():
    '''    几何变换——图像平移    '''
    global picSize, img_open, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\move.jpg', 0)
    text="图像平移就是将图像中所有的像素点按照指定的平移量水平或者垂直移动。\n图像平移后，出现的空白区域可以统一设置为0或255。"
    picLists = [["平移变换矩阵", imgExplain]]
    myWindow.showExplain(text, picLists)
    placePic1(img_open, "原图像")
    paraLists=[["水平平移距离",10],["垂直平移距离",10]]
    paraW=paraWindow(root,"请设置水平和垂直方向的平移距离",paraLists)
    img_result=geometric.move(img_open.copy(),paraW.paraLists[0][1],paraW.paraLists[1][1])
    if len(img_result.shape)>2:
        img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)
    placePic2(img_result,"图像平移")

def Rotate():
    '''    几何变换——图像旋转    '''
    global picSize, img_open, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\rotate.jpg', 0)
    text="图像的旋转是以图像的中心为原点，旋转一定的角度，即将图像上的所有像素都旋转一个相同的角度。"
    picLists = [["旋转变换矩阵", imgExplain]]
    myWindow.showExplain(text, picLists)
    placePic1(img_open, "原图像")
    angle=askfloat("设置旋转角度","请输入旋转角度",initialvalue=45)
    img_result=geometric.rotate(img_open.copy(),angle)
    if len(img_result.shape)>2:
        img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)
    placePic2(img_result,"图像旋转")

def Reflect_x():
    '''    几何变换——图像垂直镜像    '''
    global picSize, img_open, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\v-reflect.jpg', 0)
    text="垂直镜像是将图像上半部分和下半部分以图像水平中轴线为中心轴进行对换。"
    picLists = [["垂直镜像变换矩阵", imgExplain]]
    myWindow.showExplain(text, picLists)
    placePic1(img_open, "原图像")
    img_result=geometric.reflect_x(img_open.copy())
    if len(img_result.shape)>2:
        img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)
    placePic2(img_result,"图像垂直镜像")

def Reflect_y():
    '''    几何变换——图像水平镜像    '''
    global picSize, img_open, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\h-reflect.jpg', 0)
    text="水平镜像即将图像左半部分和右半部分以图像垂直中轴线为中心轴进行对换"
    picLists = [["水平镜像变换矩阵", imgExplain]]
    myWindow.showExplain(text, picLists)
    placePic1(img_open, "原图像")
    img_result=geometric.reflect_y(img_open.copy())
    if len(img_result.shape)>2:
        img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)
    placePic2(img_result,"图像水平镜像")

def Zoom():
    '''    几何变换——图像缩放    '''
    global picSize, img_open, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    imgExplain = cv2.imread(r'.\explain\zoom.jpg', 0)
    text="图像缩放是指将给定的图像在x方向和y方向按不同的比例缩放"
    picLists = [["缩放变换矩阵", imgExplain]]
    myWindow.showExplain(text, picLists)
    placePic1(img_open, "原图像")
    paraLists=[["水平方向缩放因子",1],["垂直方向缩放因子",1]]
    paraW=paraWindow(root,"请设置水平和垂直方向的缩放因子",paraLists)
    img_result=geometric.zoom(img_open.copy(),paraW.paraLists[0][1],paraW.paraLists[1][1])
    if len(img_result.shape)>2:
        img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)
    placePic2(img_result,"图像缩放")

def Affine():
    '''    几何变换——图像仿射变换    '''
    global picSize, img_open, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    placePic1(img_open, "原图像")
    explainPic=cv2.imread(r'.\explain\affine.jpg')
    text="仿射变换是一种线性变换，它保持了二维图形的“平直性”。根据变换前后的三角形坐标，可确定仿射变换矩阵"
    picLists = [["仿射变换前后的三对坐标点", explainPic]]
    myWindow.showExplain(text,picLists)
    paraLists = [[50, 50, 10,100], [200, 50, 200, 50], [50, 200, 100, 250]]  # 初始默认为3段
    paraW=paraWindow2(root,paraLists,"仿射变换参数设置","变换前三点坐标     变换后的目标坐标")
    paraLists=paraW.paraLists
    pts1=np.float32([paraLists[0][0:2],paraLists[1][0:2],paraLists[2][0:2]])
    pts2=np.float32([paraLists[0][2:4],paraLists[1][2:4],paraLists[2][2:4]])
    img_result = geometric.affine(img_open.copy(),pts1,pts2)
    if len(img_result.shape)>2:
        img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)
    placePic2(img_result, "仿射变换")

def Perspective():
    '''    几何变换——图像透视变换    '''
    global picSize, img_open, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    placePic1(img_open, "原图像")
    explainPic=cv2.imread(r'.\explain\perspective.jpg')
    text="透视变换是将图片投影到一个新的视平面，也称作投影映射。图像中的直线经过后映射到另一幅图像上仍为直线，但平行关系基本不保持。根据变换前后的四点坐标，可确定透视变换矩阵"
    picLists = [["透视变换前后的四对坐标点", explainPic]]
    myWindow.showExplain(text,picLists)
    paraLists = [[50, 50, 50, 50], [100, 50, 100, 50], [50, 100, 40, 90],[100, 100, 90, 110]]  # 初始默认为3段
    paraW=paraWindow2(root,paraLists,"透视变换参数设置","变换前四点坐标     变换后的目标坐标")
    paraLists=paraW.paraLists
    pts1=np.float32([paraLists[0][0:2],paraLists[1][0:2],paraLists[2][0:2],paraLists[3][0:2]])
    pts2=np.float32([paraLists[0][2:4],paraLists[1][2:4],paraLists[2][2:4],paraLists[3][2:4]])
    img_result = geometric.perspective(img_open.copy(),pts1,pts2)
    if len(img_result.shape)>2:
        img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)
    placePic2(img_result, "透视变换")

def Interpolation():
    '''    图像灰度插值    '''
    global picSize, img_open, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img_=img_open.copy()
    myWindow.explainText.set("图像放大后的四种插值方法比较。\nNEAREST：最近邻插值法；\nLINEAR：双线性插值法（默认）；  \nCUBIC：基于4x4像素邻域的3次插值法；\nAREA：基于局部像素的重采样")
    placePic1(img_, "原图像")
    img_gray=cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    n,m=img_gray.shape[:2]
    paraLists = [["起点x坐标", 120], ["起点y坐标", 100]]
    paraW = paraWindow(root, "设置图像块的起点坐标", paraLists)
    x0=int(paraW.paraLists[0][1])
    y0=int(paraW.paraLists[1][1])
    dx=25
    if x0+dx>=m or y0+dx>n:
        showwarning(title='警告', message='图像大小为'+str(m)+'x'+str(n)+'设置的起点坐标超出图像范围范围')
        return
    cv2.rectangle(img_, (x0, y0), (x0+dx, y0+dx), (0, 0, 255), 2)
    placePic1(img_, "标注放大区域图像")
    img = img_gray[y0:y0+dx, x0:x0+dx]
    img1 = cv2.resize(img,None,fx=10,fy=10,interpolation=cv2.INTER_NEAREST)
    img2 = cv2.resize(img,None,fx=10,fy=10,interpolation=cv2.INTER_LINEAR)
    img3 = cv2.resize(img,None,fx=10,fy=10,interpolation=cv2.INTER_CUBIC)
    img4 = cv2.resize(img,None,fx=10,fy=10,interpolation=cv2.INTER_AREA)
    picLists1=[['NEAREST',img1],['LINEAR',img2]]
    picLists2=[['CUBIC', img3], ['AREA', img4]]
    combinedImg1 = myWindow.combine_images(picLists1, explain=True, fontSize=1)#合并两张图片
    combinedImg2 = myWindow.combine_images(picLists2, explain=True, fontSize=1)#合并两张图片
    picLists=[combinedImg1,combinedImg2]
    img_result=myWindow.combine_images(picLists)
    if len(img_result.shape)>2:
        img_result=cv2.cvtColor(img_result,cv2.COLOR_BGR2RGB)
    placePic2(img_result,"放大图像后的四种插值比较")

#---------------------------------------------------------
def Add_gaussian_noise():
    '''    添加高斯噪声    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_result is None:
        img_ = img_gray.copy()
    else:
        answer = askyesno("选择待退化图像", "Y-对打开图像进行退化，N-继续对已退化图像进行退化")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    placePic1(img_, '待处理图像')
    paraLists=[["均值",0],["方差",0.03]]
    paraW=paraWindow(root,"设置高斯噪声的均值与方差",paraLists)
    # paraW.show()
    img_result=spatialRestore.add_gaussian_noise(img_,paraW.paraLists[0][1],paraW.paraLists[1][1])
    histr = Histogram.gray_histogram(img_result)
    myWindow.showAndBarFig0(img_result,histr)
    myWindow.resultText.set("添加高斯噪声")
    # placePic2(img_result, "添加高斯噪声")
    # showinfo(title='提示', message='按任意键显示直方图，按回退键显示图像')
    # cv2.waitKey(0)
    # placePic1(img_result, "添加高斯噪声")
    explainPic = cv2.imread(r'.\explain\gaosi.jpg')
    picLists = [["高斯噪声是一种白噪声，高斯随机变量的PDF由下式给出：", explainPic]]
    myWindow.showFig1(picLists)

def Add_salt_and_pepper_noise():
    '''    添加椒盐噪声    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_result is None:
        img_ = img_gray.copy()
    else:
        answer = askyesno("选择待退化图像", "Y-对打开图像进行退化，N-继续对已退化图像进行退化")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    placePic1(img_, '待处理图像')
    proportion = askfloat("设置噪声比例", "请输入椒盐噪声的比例，介于【0，1】之间",initialvalue=0.05,minvalue=0,maxvalue=1)
    if proportion is None:
        showwarning(title='警告', message='请先设置噪声强度')
        return
    img_result=spatialRestore.add_salt_and_pepper_noise(img_,proportion)
    histr = Histogram.gray_histogram(img_result)
    myWindow.showAndBarFig0(img_result,histr)
    myWindow.resultText.set("添加椒盐噪声")
    # placePic2(img_result, "添加椒盐噪声")
    explainPic = cv2.imread(r'.\explain\jiaoyan.jpg')
    picLists = [["椒盐噪声是一种脉冲噪声，双极脉冲噪声的概率密度函数可以如下表示：", explainPic]]
    myWindow.showFig1(picLists)

def Add_mean_noise():
    '''    添加均匀噪声    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_result is None:
        img_ = img_gray.copy()
    else:
        answer = askyesno("选择待退化图像", "Y-对打开图像进行退化，N-继续对已退化图像进行退化")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    placePic1(img_, '待处理图像')
    paraLists=[["起始灰度值a",50],["终止灰度值b",100],["噪声比例",0.1]]
    paraW=paraWindow(root,"设置均值噪声的灰度范围【0.255】",paraLists)
    # paraW.show()
    img_result=spatialRestore.add_mean_noise(img_,paraW.paraLists[0][1],paraW.paraLists[1][1],paraW.paraLists[2][1])
    histr = Histogram.gray_histogram(img_result)
    myWindow.showAndBarFig0(img_result,histr)
    myWindow.resultText.set("添加均值噪声")
    # placePic2(img_result, "添加均值噪声")
    explainPic = cv2.imread(r'.\explain\junzhi.jpg')
    picLists = [["均值噪声分布的概率密度函数为", explainPic]]
    myWindow.showFig1(picLists)

def Add_rayleigh_noise():
    '''    添加瑞利噪声    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_result is None:
        img_ = img_gray.copy()
    else:
        answer = askyesno("选择待退化图像", "Y-对打开图像进行退化，N-继续对已退化图像进行退化")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    placePic1(img_, '待处理图像')
    paraLists=[["起始灰度值a",50],["平缓度b（越大分布越平缓）",100],["噪声比例",0.1]]
    paraW=paraWindow(root,"设置瑞利噪声的参数",paraLists)
    img_result=spatialRestore.add_rayleigh_noise(img_,paraW.paraLists[0][1],paraW.paraLists[1][1],paraW.paraLists[2][1])
    histr = Histogram.gray_histogram(img_result)
    myWindow.showAndBarFig0(img_result,histr)
    myWindow.resultText.set("添加瑞利噪声")
    # placePic2(img_result, "添加瑞利噪声")
    explainPic = cv2.imread(r'.\explain\ruili.jpg')
    picLists = [["瑞利噪声的概率密度函数由下式给出：", explainPic]]
    myWindow.showFig1(picLists)

def Add_erlang_noise():
    '''    添加伽马噪声    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_result is None:
        img_ = img_gray.copy()
    else:
        answer = askyesno("选择待退化图像", "Y-对打开图像进行退化，N-继续对已退化图像进行退化")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    placePic1(img_, '待处理图像')
    paraLists=[["起始灰度值a",50],["尖锐度b（越大分布越尖锐）",0.1],["噪声比例",0.1]]
    paraW=paraWindow(root,"设置伽马噪声的参数",paraLists)
    img_result=spatialRestore.add_erlang_noise(img_,paraW.paraLists[0][1],paraW.paraLists[1][1],paraW.paraLists[2][1])
    histr = Histogram.gray_histogram(img_result)
    myWindow.showAndBarFig0(img_result,histr)
    myWindow.resultText.set("添加伽马噪声")
    # placePic2(img_result, "添加伽马噪声")
    explainPic = cv2.imread(r'.\explain\gama.jpg')
    picLists = [["伽马噪声的概率密度函数由下式给出：", explainPic]]
    myWindow.showFig1(picLists)
#---------------------------------------------------------
def Motion_blur():
    '''    添加运动模糊    '''
    global img_gray, img_result,PSF
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    else:
        img_ = img_gray.copy()
    if img_result is not None:
        answer = askyesno("选择待退化图像", "Y-对打开图像进行退化，N-继续对已退化图像进行退化")
        if not answer:
            img_ = img_result.copy()
    placePic1(img_,'待处理图像')
    explainPic=cv2.imread(r'.\explain\motionBlur.jpg')
    picLists = [["运动模糊核是由单位矩阵旋转产生的，如下图所示是一个90度方向的运动模糊核", explainPic]]
    myWindow.showFig1(picLists)
    paraLists=[["运动模糊核大小",5],["运动方向（度）",45]]
    paraW=paraWindow(root,"请设置运动模糊的参数",paraLists)
    kernel_size=int(paraW.paraLists[0][1])
    angle=int(paraW.paraLists[1][1])
    PSF = Restore.motion_PSF(kernel_size, angle)  # 生成运动模糊核
    PSF = Restore.extension_PSF(img_, PSF)  # 在频域进行运动模糊，需要扩展PSF，使其与图像一样大小
    img_result = Restore.make_blurred(img_, PSF)
    placePic2(img_result, "运动模糊后的图像")

def Gaussian_blur():
    '''    添加高斯模糊    '''
    global picSize, img_gray, img_result, PSF
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    else:
        img_ = img_gray.copy()
    if img_result is not None:
        answer = askyesno("选择待退化图像", "Y-对打开图像进行退化，N-继续对已退化图像进行退化")
        if not answer:
            img_ = img_result.copy()
    placePic1(img_,'待处理图像')
    explainPic=cv2.imread(r'.\explain\gaussianBlur.jpg')
    picLists = [["二维高斯核可以根据下面的公式推导为两个一维高斯核的乘积", explainPic]]
    myWindow.showFig1(picLists)
    paraLists=[["高斯模糊核大小",5],["标准方差",0.1]]
    paraW=paraWindow(root,"请设置高斯模糊的参数",paraLists)
    kernel_size=int(paraW.paraLists[0][1])
    sigma=int(paraW.paraLists[1][1])
    PSF = Restore.Gaussian_PSF(kernel_size, sigma)  # 生成高斯模糊核
    PSF = Restore.extension_PSF(img_, PSF)  # 在频域进行运动模糊，需要扩展PSF，使其与图像一样大小
    img_result = Restore.make_blurred(img_, PSF)
    placePic2(img_result, "高斯模糊后的图像")

def Turbulence_blur():
    '''    添加大气湍流模糊    '''
    global picSize, img_gray, img_result, PSF
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    else:
        img_ = img_gray.copy()
    if img_result is not None:
        answer = askyesno("选择待退化图像", "Y-对打开图像进行退化，N-继续对已退化图像进行退化")
        if not answer:
            img_ = img_result.copy()
    placePic1(img_,'待处理图像')
    explainPic=cv2.imread(r'.\explain\turbulence.jpg')
    picLists = [["大气湍流模型的传递函数如下图所示，其中k值越大，模糊越严重", explainPic]]
    myWindow.showFig1(picLists)
    k=askfloat("大气湍流模型参数","请输入大气湍流的强度k",initialvalue=0.001)
    PSF = Restore.turbulence_PSF(img_,k)  # 生成模糊核
    PSF = Restore.extension_PSF(img_, PSF)  # 在频域进行运动模糊，需要扩展PSF，使其与图像一样大小
    img_result = Restore.make_blurred(img_, PSF)
    placePic2(img_result, "大气湍流模糊后的图像")

#---------------------------------------------------------
def Harmonic_filter():
    '''    谐波滤波    '''
    global picSize, img_gray, img_result, PSF
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    if img_result is not None:
        answer = askyesno("选择待滤波图像", "Y-对打开图像进行滤波，N-对处理后图像进行滤波")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()

    placePic1(img_, '待处理图像')
    kernelSize = askinteger("滤波核设置", "请输入滤波核大小", initialvalue=3)
    if kernelSize is None:
        showwarning("警告","请先确定滤波核大小！！！")
        return
    img_result = spatialRestore.harmonic_filter(img_,kernelSize)
    placePic2(img_result, "谐波滤波后的图像")
    imgExplain = cv2.imread(r'.\explain\harmonic.jpg', 0)
    picLists = [["谐波均值（Harmonic Mean）适合去除盐噪声，但不适合胡椒噪声，也适用于其他类型的噪声，比如高斯噪声。", imgExplain]]
    myWindow.showFig1(picLists)

def Contra_harmonic_filter():
    '''    逆谐波滤波    '''
    global picSize, img_gray, img_result, PSF
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    if img_result is not None:
        answer = askyesno("选择待滤波图像", "Y-对打开图像进行滤波，N-对处理后图像进行滤波")
        if not answer:
            img_ = img_result.copy()
        else:
            img_ = img_gray.copy()
    placePic1(img_, '待处理图像')
    paraLists = [["滤波核大小", 5], ["Q值", -1]]
    paraW = paraWindow(root, "请设置伪逆滤波的参数", paraLists)
    kernelSize = int(paraW.paraLists[0][1])
    Q = int(paraW.paraLists[1][1])
    img_result = spatialRestore.contra_harmonic_filter(img_,kernelSize,Q)
    placePic2(img_result, "逆谐波滤波后的图像")
    imgExplain = cv2.imread(r'.\explain\contraharmonic.jpg', 0)
    picLists = [["逆谐波均值，Q是滤波器的阶数。Q为负值时，可消除盐噪声。Q为正值时，可消除胡椒噪声。不能同时消除两者", imgExplain]]
    myWindow.showFig1(picLists)
#------------------------------------------------------------------
def Inverse():
    '''    逆滤波    '''
    global picSize, img_gray, img_result,PSF
    initWindows()
    if img_result is None or PSF is None:
        showwarning(title='警告', message='请先对图像进行退化！')
        return
    placePic1(img_result,"退化后图像")
    img_restore = Restore.inverse(img_result, PSF)
    # PSF=None
    placePic2(img_restore, "逆滤波后的图像")
    explainPic = cv2.imread(r'.\explain\nilvbo.jpg')
    picLists = [["有噪声的情况下，逆滤波复原的基本原理可以写成：", explainPic]]
    myWindow.showFig1(picLists)

def Improved_inverse():
    '''    伪逆滤波    '''
    global picSize, img_gray, img_result, PSF
    initWindows()
    if img_result is None or PSF is None:
        showwarning(title='警告', message='请先对图像进行退化！')
        return
    placePic1(img_result, "退化后图像")
    explainPic=cv2.imread(r'.\explain\pseudoInverse.jpg')
    picLists = [["伪逆滤波的传递函数是：低频部分(小于截止w)频率与逆滤波相同，高频部分采用一个指定的较小的常量k", explainPic]]
    myWindow.showFig1(picLists)
    paraLists=[["伪逆滤波的截止频率",70],["高频常数",0.01]]
    paraW=paraWindow(root,"请设置伪逆滤波的参数",paraLists)
    w=int(paraW.paraLists[0][1])
    k=int(paraW.paraLists[1][1])
    img_restore = Restore.improved_inverse(img_result, PSF,w,k)
    # PSF=None
    placePic2(img_restore, "伪逆滤波后的图像")
def Wiener():
    '''    维纳滤波    '''
    global picSize, img_gray, img_result, PSF
    initWindows()
    if img_result is None or PSF is None:
        showwarning(title='警告', message='请先对图像进行退化！')
        return
    placePic1(img_result, "退化后图像")
    explainPic = cv2.imread(r'.\explain\wiener.jpg')
    picLists = [["维纳滤波采用最小均方误差作为最佳过滤准则，使用时需要估计图像的噪信比，无噪声时，噪信比为0", explainPic]]
    myWindow.showFig1(picLists)
    k=askfloat("维纳滤波参数","请估计图像的噪信比",initialvalue=0.01)
    img_restore = Restore.wiener(img_result, PSF, k)
    # PSF=None
    placePic2(img_restore, "维纳滤波后的图像")

def Constrained_least_squares():
    '''    受限最小二乘滤波    '''
    global picSize, img_gray, img_result, PSF
    initWindows()
    if img_result is None or PSF is None:
        showwarning(title='警告', message='请先对图像进行退化！')
        return
    placePic1(img_result, "退化后图像")
    explainPic = cv2.imread(r'.\explain\constrained_least_squares.jpg')
    picLists = [["约束最小二乘滤波通过抑制高频成份，来获得平滑的复原图像，公式中Q是锐化算子，λ是权重参数，其值越大越平滑", explainPic]]
    myWindow.showFig1(picLists)
    k = askfloat("约束最小二乘滤波参数", "请输入λ", initialvalue=0.1)
    img_restore = Restore.constrained_least_squares(img_result, PSF, k)
    # PSF=None
    placePic2(img_restore, "约束最小二乘滤波后的图像")


# -------------------------------------------------------------
def Binary():
    '''    图像二值化    '''
    global img_open,img_binary,img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    placePic1(img_open, "原图像")
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_result=img_binary
    placePic2(img_result, "二值化图像")
    imgExplain = cv2.imread(r'.\explain\Image Binarization.jpg', 0)
    picLists = [["图像二值化处理，也就是阈值分割，它的目的就是求一个阈值T，并用T将图像f(x,y)分成对象物和背景两个领域。", imgExplain]]
    myWindow.showFig1(picLists)
def Dilate():
    '''    形态学处理-膨胀运算    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子大小设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形','十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.dilate(img_binary, kernel, iterations=1)
    placePic2(img_result, "膨胀后的图像")
    myWindow.explainText.set("膨胀运算是对二值化物体边界点扩充，将与物体接触的所以背景点合并到该物体中，使边界向外扩张。")

def Erode():
    '''    形态学处理-腐蚀运算    '''
    global img_open,img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row,col=img_binary.shape
    minrc=min(row,col)
    kernelSize=askinteger("结构算子设置","请输入结构算子大小",initialvalue=3,minvalue=3,maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.erode(img_binary,kernel,iterations = 1)
    placePic2(img_result,"腐蚀后的图像")
    imgExplain = cv2.imread(r'.\explain\Erosion.jpg',0)
    picLists = [["形态学腐蚀的定义：集合A被集合B腐蚀为：A○—B={x:B+x.A}",imgExplain]]
    myWindow.showFig1(picLists)

def Open_operation():
    '''    形态学处理-开运算    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    placePic2(img_result, "开运算后的图像")
    myWindow.explainText.set("开运算数学上是先腐蚀后膨胀的结果，物理结果为完全删除了不能包含结构元素的对象区域，平滑了对象的轮廓，断开了狭窄的连接，去掉了细小的突出部分。")

def Close_operation():
    '''    形态学处理-闭运算    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    placePic2(img_result, "闭运算后的图像")
    myWindow.explainText.set("闭运算在数学上是先膨胀再腐蚀的结果，物理结果也是会平滑对象的轮廓，但是与开运算不同的是，闭运算一般会将狭窄的缺口连接起来形成细长的弯口，并填充比结构元素小的洞。")

def Gradient_morphology():
    '''    形态学处理-形态学梯度    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_GRADIENT, kernel)
    placePic2(img_result, "形态学梯度图像")
    myWindow.explainText.set("形态学梯度是一幅图像膨胀与腐蚀的差别,结果看上去就像前景物体的轮廓。")

def Tophat():
    '''    形态学处理-礼帽运算    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_TOPHAT, kernel)
    placePic2(img_result, "形态学礼帽图像")
    myWindow.explainText.set("形态学礼帽是原始图像与进行开运算之后得到的图像的差")

def Blackhat():
    '''    形态学处理-黑帽运算    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_BLACKHAT, kernel)
    placePic2(img_result, "形态学黑帽图像")
    myWindow.explainText.set("形态学黑帽是进行闭运算之后得到的图像与原始图像的差。")

def Fill_holes():
    '''    形态学处理-孔洞填充运算    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    explainPic = cv2.imread(r'.\explain\fill_holes.jpg')
    text="孔洞填充是以带有白色边框的黑色图像Marker为起点，对其进行连续膨胀，并以原图像的补集作为遮罩Mask，来限制膨胀，直至膨胀收敛.最后对Marker取补即得到最终图像，与原图相减可得到填充图像。"
    picLists = [["形态学变换公式", explainPic]]
    myWindow.showExplain(text,picLists)
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = morphology.fill_holes(img_binary,kernel)
    placePic2(img_result, "填充孔洞图像")

def Skeleton():
    '''    形态学处理-骨架提取    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    img=img_open.copy()
    if len(img_open.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 0:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    elif structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_BLACKHAT, kernel)
    placePic2(img_result, "骨架提取")
    myWindow.explainText.set("图像骨架提取，实际上就是提取目标在图像上的中心像素轮廓。")
# 系统默认图片展示形态学效果------------------------------------------------------------
def Binary1():
    '''    图像二值化    '''
    global img_open,img_binary,img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    placePic1(img_, "原图像")
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_result=img_binary
    placePic2(img_result, "二值化图像")
    imgExplain = cv2.imread(r'.\explain\Image Binarization.jpg', 0)
    picLists = [["图像二值化处理，也就是阈值分割，它的目的就是求一个阈值T，并用T将图像f(x,y)分成对象物和背景两个领域。", imgExplain]]
    myWindow.showFig1(picLists)
def Dilate1():
    '''    形态学处理-膨胀运算    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子大小设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形','十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.dilate(img_binary, kernel, iterations=1)
    placePic2(img_result, "膨胀后的图像")
    myWindow.explainText.set("膨胀运算是对二值化物体边界点扩充，将与物体接触的所以背景点合并到该物体中，使边界向外扩张。")

def Erode1():
    '''    形态学处理-腐蚀运算    '''
    global img_open,img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row,col=img_binary.shape
    minrc=min(row,col)
    kernelSize=askinteger("结构算子设置","请输入结构算子大小",initialvalue=3,minvalue=3,maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.erode(img_binary,kernel,iterations = 1)
    placePic2(img_result,"腐蚀后的图像")
    imgExplain = cv2.imread(r'.\explain\Erosion.jpg',0)
    picLists = [["形态学腐蚀的定义：集合A被集合B腐蚀为：A○—B={x:B+x.A}",imgExplain]]
    myWindow.showFig1(picLists)

def Open_operation1():
    '''    形态学处理-开运算    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    placePic2(img_result, "开运算后的图像")
    myWindow.explainText.set("开运算数学上是先腐蚀后膨胀的结果，物理结果为完全删除了不能包含结构元素的对象区域，平滑了对象的轮廓，断开了狭窄的连接，去掉了细小的突出部分。")

def Close_operation1():
    '''    形态学处理-闭运算    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    placePic2(img_result, "闭运算后的图像")
    myWindow.explainText.set("闭运算在数学上是先膨胀再腐蚀的结果，物理结果也是会平滑对象的轮廓，但是与开运算不同的是，闭运算一般会将狭窄的缺口连接起来形成细长的弯口，并填充比结构元素小的洞。")

def Gradient_morphology1():
    '''    形态学处理-形态学梯度    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_GRADIENT, kernel)
    placePic2(img_result, "形态学梯度图像")
    myWindow.explainText.set("形态学梯度是一幅图像膨胀与腐蚀的差别,结果看上去就像前景物体的轮廓。")

def Tophat1():
    '''    形态学处理-礼帽运算    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=4, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_TOPHAT, kernel)
    placePic2(img_result, "形态学礼帽图像")
    myWindow.explainText.set("形态学礼帽是原始图像与进行开运算之后得到的图像的差")

def Blackhat1():
    '''    形态学处理-黑帽运算    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=4, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_BLACKHAT, kernel)
    placePic2(img_result, "形态学黑帽图像")
    myWindow.explainText.set("形态学黑帽是进行闭运算之后得到的图像与原始图像的差。")

def Fill_holes1():
    '''    形态学处理-孔洞填充运算    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    explainPic = cv2.imread(r'.\explain\fill_holes.jpg')
    text="孔洞填充是以带有白色边框的黑色图像Marker为起点，对其进行连续膨胀，并以原图像的补集作为遮罩Mask，来限制膨胀，直至膨胀收敛.最后对Marker取补即得到最终图像，与原图相减可得到填充图像。"
    picLists = [["形态学变换公式", explainPic]]
    myWindow.showExplain(text,picLists)
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=3, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    else:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = morphology.fill_holes(img_binary,kernel)
    placePic2(img_result, "填充孔洞图像")

def Skeleton1():
    '''    形态学处理-骨架提取    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\morphology.png', dtype=np.uint8), 0)
    _, img_binary = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placePic1(img_binary, "原二值图像")
    row, col = img_binary.shape
    minrc = min(row, col)
    kernelSize = askinteger("结构算子设置", "请输入结构算子大小", initialvalue=4, minvalue=3, maxvalue=minrc)
    paraLists = [["请选择结构算子形状", ['矩形', '十字形', '椭圆形']]]
    paraW = listWindow(root, "结构算子选项", paraLists)
    structType = paraW.returnValue
    if structType == 0:  # 矩形
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    elif structType == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(kernelSize,kernelSize))
    elif structType == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_BLACKHAT, kernel)
    placePic2(img_result, "骨架提取")
    myWindow.explainText.set("图像骨架提取，实际上就是提取目标在图像上的中心像素轮廓。")
#----------------------------------------
def Basic_global_thresholding():
    '''    基本全阈值分割    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    th=segmentation.basic_global_thresholding(img_gray)
    _,img_result=cv2.threshold(img_gray,th,255,cv2.THRESH_BINARY)
    placePic2(img_result, "基本全阈值分割图像")
    myWindow.explainText.set('''全局阈值分割就是将整个图像灰度值大于阈值（thresh）的像素调为白色，小于或等于阈值的调整为黑色，也可以反过来。 ''')

def Otsu_threshold():
    '''    OSTU全阈值分割    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    answer=askyesno("平滑选项","请选择在分割前是否使用高斯平滑")
    img=img_gray.copy()
    if answer:
        img = cv2.GaussianBlur(img,(5,5),0)
    _, img_result = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    placePic2(img_result, "单阈值OTSU算法分割图像")
    myWindow.explainText.set('''以最佳阈值将图像的灰度直方图分割成两部分，使两类之间的方差取最大值，即分离性最大。
    此算法利用了最小二乘法原理。''')

def Adaptive_mean_threshold():
    '''    基于邻域均值的自适应阈值分割    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # img_gray=cv2.equalizeHist(img_gray)
    maxSize=(min(img_gray.shape[:2]))//2
    kernalSize=askinteger("窗口设置","请输入邻域窗口大小",initialvalue=15,minvalue=3,maxvalue=maxSize)
    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_result = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernalSize, 20)
    placePic2(img_result, "基于邻域均值的自适应阈值分割图像")
    myWindow.explainText.set(
        "自适应阈值，则是根据像素的邻域块的像素值分布来确定该像素位置上的二值化阈值。")

def Adaptive_gaussian_threshold():
    '''    基于高斯滤波的自适应阈值分割    '''
    global picSize, img_gray, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    maxSize=(min(img_gray.shape[:2]))//2
    kernalSize=askinteger("窗口设置","请输入高斯窗口大小",initialvalue=15,minvalue=3,maxvalue=maxSize)
    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_result = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernalSize, 0)
    placePic2(img_result, "基于高斯滤波的自适应阈值分割图像")
    myWindow.explainText.set(
        "基于高斯滤波的自适应阈值分割，则是根据像素的邻域块的像素值分布来确定该像素位置上的二值化阈值。"
        "\n计算某个邻域(局部)的高斯加权平均(高斯滤波)来确定阈值。")

# def Moving_threshold():
#     '''    滑动平均阈值分割    '''
#     global picSize, img_gray, img_result
#     initWindows()
#     if img_gray is None:
#         showwarning(title='警告', message='请先在文件菜单下打开图片！')
#         return
#     maxSize = (min(img_gray.shape[:2])) // 2
#     kernalSize = askinteger("窗口设置", "请输入滑动平均窗口大小", initialvalue=15, minvalue=3, maxvalue=maxSize)
#     thresholdWeight = askfloat("阈值权重设置", "请输入分割权重比例，即黑色灰度值所占的比例", initialvalue=0.7, minvalue=0, maxvalue=1)
#     # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
#     img_result = segmentation.moving_threshold(img_gray, kernalSize,thresholdWeight)
#     placePic2(img_result, "滑动平均阈值分割图像")
#     myWindow.explainText.set('''滑动平均变阈值法：一般沿行、列或Z字形进行滑动平均，将像素值与滑动平均值进行比较，根据比较情况进行分割。''')

#系统默认图片的分割效果----------------------------------------
def Basic_global_thresholding1():
    '''    基本全阈值分割    '''
    global picSize, img_gray, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\paopao.jpg', dtype=np.uint8), 0)
    placePic1(img_,"待分割图像")
    th=segmentation.basic_global_thresholding(img_)
    _,img_result=cv2.threshold(img_,th,255,cv2.THRESH_BINARY)
    placePic2(img_result, "基本全阈值分割图像")
    myWindow.explainText.set('''全局阈值分割就是将整个图像灰度值大于阈值（thresh）的像素调为白色，小于或等于阈值的调整为黑色，也可以反过来。 ''')

def Otsu_threshold1():
    '''    OSTU全阈值分割    '''
    global picSize, img_gray, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\paopao.jpg', dtype=np.uint8), 0)
    placePic1(img_,"待分割图像")
    answer=askyesno("平滑选项","请选择在分割前是否使用高斯平滑")
    if answer:
        img_ = cv2.GaussianBlur(img_,(5,5),0)
    _, img_result = cv2.threshold(img_, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    placePic2(img_result, "单阈值OTSU算法分割图像")
    myWindow.explainText.set('''以最佳阈值将图像的灰度直方图分割成两部分，使两类之间的方差取最大值，即分离性最大。
    此算法利用了最小二乘法原理。''')

def Adaptive_mean_threshold1():
    '''    基于邻域均值的自适应阈值分割    '''
    global picSize, img_gray, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\light_bar.png', dtype=np.uint8), 0)
    placePic1(img_,"待分割图像")
    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # img_gray=cv2.equalizeHist(img_gray)
    maxSize=(min(img_.shape[:2]))//2
    kernalSize=askinteger("窗口设置","请输入邻域窗口大小",initialvalue=15,minvalue=3,maxvalue=maxSize)
    # img_ = cv2.GaussianBlur(img_, (5, 5), 0)
    if kernalSize%2==0:
        kernalSize=kernalSize-1
    img_result = cv2.adaptiveThreshold(img_, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernalSize, 20)
    placePic2(img_result, "基于邻域均值的自适应阈值分割图像")
    myWindow.explainText.set(
        "自适应阈值，则是根据像素的邻域块的像素值分布来确定该像素位置上的二值化阈值。")

def Adaptive_gaussian_threshold1():
    '''    基于高斯滤波的自适应阈值分割    '''
    global picSize, img_gray, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\light_bar.png', dtype=np.uint8), 0)
    placePic1(img_,"待分割图像")
    maxSize=(min(img_.shape[:2]))//2
    kernalSize=askinteger("窗口设置","请输入高斯窗口大小,必须为奇数",initialvalue=25,minvalue=3,maxvalue=maxSize)
    if kernalSize%2==0:
        kernalSize=kernalSize-1
    # img_ = cv2.GaussianBlur(img_, (5, 5), 0)
    img_result = cv2.adaptiveThreshold(img_, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernalSize, 18)
    placePic2(img_result, "基于高斯滤波的自适应阈值分割图像")
    myWindow.explainText.set(
        "基于高斯滤波的自适应阈值分割，则是根据像素的邻域块的像素值分布来确定该像素位置上的二值化阈值。"
        "\n计算某个邻域(局部)的高斯加权平均(高斯滤波)来确定阈值。")

# def Moving_threshold1():
#     '''    滑动平均阈值分割    '''
#     global picSize, img_gray, img_result
#     initWindows()
#     img_ = cv2.imdecode(np.fromfile(r'.\explain\light_bar.png', dtype=np.uint8), 0)
#     placePic1(img_,"待分割图像")
#     maxSize = (min(img_.shape[:2])) // 2
#     kernalSize = askinteger("窗口设置", "请输入滑动平均窗口大小", initialvalue=15, minvalue=3, maxvalue=maxSize)
#     if kernalSize%2==0:
#         kernalSize=kernalSize-1
#     thresholdWeight = askfloat("阈值权重设置", "请输入分割权重比例，即黑色灰度值所占的比例", initialvalue=0.7, minvalue=0, maxvalue=1)
#     # img_ = cv2.GaussianBlur(img_, (5, 5), 0)
#     img_result = segmentation.moving_threshold(img_, kernalSize,thresholdWeight)
#     placePic2(img_result, "滑动平均阈值分割图像")
#     myWindow.explainText.set('''滑动平均变阈值法：一般沿行、列或Z字形进行滑动平均，将像素值与滑动平均值进行比较，根据比较情况进行分割。''')


#----------------------------------------------------
def Draw_contours():
    '''    提取目标轮廓    '''
    global img_open, img_binary, img_result,getContours,contoursMinArea
    initWindows()
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    placePic1(img_open, "原图像")
    myWindow.explainText.set("OpenCV的findContours()函数来查找检测物体的轮廓。OpenCV的drawContours在图像上绘制轮廓。区域面积小于设置最小值的区域将会被忽略")
    row,col=img_open.shape[:2]
    maxV=row*col
    minArea=askfloat("最小区域设置","请输入最小区域外接矩形面积值",initialvalue=50,minvalue=1,maxvalue=maxV)
    if not minArea:
        showwarning("警告","请输入最小区域外接矩形面积值")
        return
    img_result=description.draw_contours(img_open,minArea)
    contoursMinArea=minArea
    placePic2(img_result, "添加了目标轮廓的图像")

def Get_region_description():
    '''
    获得区域描述，包括质心、方向、面积、周长、圆形度、矩形度
    '''
    global img_open, img_binary, img_result,contoursMinArea
    initWindows()
    if contoursMinArea is None:
        showwarning(title='警告', message='请重新获得目标轮廓！')
        return
    placePic1(img_result, "目标轮廓图像")
    img_result,contours=description.get_region_description(img_result,contoursMinArea)
    contoursMinArea=None
    placePic2(img_result, "添加了区域方向的图像")
    i=0
    descipStr=''
    for cnt in contours:
        i+=1
        descipStr+="第%d个区域:"%i
        descipStr+="质心为(%d,%d),"%(cnt[0][0],cnt[0][1])
        descipStr+="方向：%d度,"%cnt[1]
        descipStr+="面积：%d,"%cnt[2]
        descipStr+="周长：%d,"%cnt[3]
        descipStr+="圆形度：%0.2f,"%cnt[4]
        descipStr+= "矩形度：%0.2f"%cnt[5]
        descipStr +='\n'
    myWindow.explainText.set(descipStr)

def Moment_invariants():
    '''    图像经过多种几何变换后的不变矩    '''
    global img_open, img_binary, img_result
    initWindows()
    if img_gray is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    placePic1(img_gray, "原图像")
    imgLists,textLists,momentLists=description.moment_invariants(img_gray)
    i=0
    descipStr='图像几何变换后的7个不变矩\n'
    for moment in momentLists:
        descipStr+=textLists[i]+'\t'
        i+=1
        for mm in moment:
            descipStr+="%0.2f"%mm+'\t'
        descipStr +='\n'
    myWindow.explainText.set(descipStr)
    #合并几何变换后的图像
    # picLists=[]
    # for tt,img in zip(textLists,imgLists):
    #     picLists.append([tt, img])
    # combinedImg1 = myWindow.combine_images(picLists[:3], explain=False,fontSize=1)#合并两张图片
    # combinedImg2 = myWindow.combine_images(picLists[3:], explain=False,fontSize=1)#合并两张图片
    combinedImg1 = myWindow.combine_images(imgLists[:3])#合并三张图片
    combinedImg2 = myWindow.combine_images(imgLists[3:])#合并后几张图片
    picLists=[combinedImg1,combinedImg2]
    img_result=myWindow.combine_images(picLists)
    placePic2(img_result, "各种几何变换图像")

#系统默认图片展示图像描述效果---------------------------------------
def Draw_contours1():
    '''    提取目标轮廓    '''
    global img_open, img_binary, img_result,getContours,contoursMinArea
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\describe.png', dtype=np.uint8), 0)
    placePic1(img_, "原图像")
    myWindow.explainText.set("OpenCV的findContours()函数来查找检测物体的轮廓。OpenCV的drawContours在图像上绘制轮廓。区域面积小于设置最小值的区域将会被忽略")
    row,col=img_.shape[:2]
    maxV=row*col
    minArea=askfloat("最小区域设置","请输入最小区域外接矩形面积值",initialvalue=50,minvalue=1,maxvalue=maxV)
    if not minArea:
        showwarning("警告","请输入最小区域外接矩形面积值")
        return
    img_result=description.draw_contours(img_,minArea)
    contoursMinArea=minArea
    placePic2(img_result, "添加了目标轮廓的图像")

def Get_region_description1():
    '''
    获得区域描述，包括质心、方向、面积、周长、圆形度、矩形度
    '''
    global img_open, img_binary, img_result,contoursMinArea
    initWindows()
    if contoursMinArea is None:
        showwarning(title='警告', message='请重新获得目标轮廓！')
        return
    placePic1(img_result, "目标轮廓图像")
    img_result,contours=description.get_region_description(img_result,contoursMinArea)
    contoursMinArea=None
    placePic2(img_result, "添加了区域方向的图像")
    i=0
    descipStr=''
    for cnt in contours:
        i+=1
        descipStr+="第%d个区域:"%i
        descipStr+="质心为(%d,%d),"%(cnt[0][0],cnt[0][1])
        descipStr+="方向：%d度,"%cnt[1]
        descipStr+="面积：%d,"%cnt[2]
        descipStr+="周长：%d,"%cnt[3]
        descipStr+="圆形度：%0.2f,"%cnt[4]
        descipStr+= "矩形度：%0.2f"%cnt[5]
        descipStr +='\n'
    myWindow.explainText.set(descipStr)

def Moment_invariants1():
    '''    图像经过多种几何变换后的不变矩    '''
    global img_open, img_binary, img_result
    initWindows()
    img_ = cv2.imdecode(np.fromfile(r'.\explain\lung.jpg', dtype=np.uint8), 0)
    placePic1(img_, "原图像")
    imgLists,textLists,momentLists=description.moment_invariants(img_)
    i=0
    descipStr='图像几何变换后的7个不变矩\n'
    for moment in momentLists:
        descipStr+=textLists[i]+'\t'
        i+=1
        for mm in moment:
            descipStr+="%0.2f"%mm+'\t'
        descipStr +='\n'
    myWindow.explainText.set(descipStr)
    #合并几何变换后的图像
    picLists=[]
    for tt,img in zip(textLists,imgLists):
        picLists.append([tt,img])
    combinedImg1 = myWindow.combine_images(picLists[:3], explain=True,fontSize=0.6)#合并两张图片
    combinedImg2 = myWindow.combine_images(picLists[3:], explain=True,fontSize=0.6)#合并两张图片
    picLists=[combinedImg1,combinedImg2]
    img_result=myWindow.combine_images(picLists)
    placePic2(img_result, "各种几何变换图像")


#---------------------------------------------------
def Car_license_train():
    '''    训练车牌识别模型    '''
    initWindows()
    myWindow.explainText.set("车牌识别系统通常包括图像预处理、车牌检测定位、字符分割、特征提取和字符识别等部分。"
                             "\n1.图像预处理就是去除图像中的噪声、模糊、光照不均匀、遮挡等问题。"
                             "\n2.车牌检测定位是从复杂背景的汽车图像中检测并定位车牌的位置。"
                             "\n3.字符分割是将车牌区域进一步分割成若干个单字符图像。"
                             "\n4.特征提取是提取易于区分字符的特征，用于字符识别。"
                             "\n5.字符识别包括模型训练和车牌识别两个阶段。")
    predictor = predict.CardPredictor(myWindow)   #创建识别模型
    flag=True    #是否重新训练的标记
    if os.path.exists(r".\carLicense\svm.dat") and os.path.exists(r".\carLicense\svmchinese.dat"):
         flag= askyesno("选择是否重新训练模型", "模型已经训练过了，训练模型需要花费一段时间，是否重新训练模型？")
         if not flag:
            return
    predictor.train_svm()                 #训练模型

def Car_license_recognition():
    '''    利用训练好的模型，对车牌图像进行识别    '''
    global picSize, img_gray, img_result
    initWindows()
    myWindow.explainText.set("车牌识别系统通常包括图像预处理、车牌检测定位、字符分割、特征提取和字符识别等部分。"
                             "\n1.图像预处理就是去除图像中的噪声、模糊、光照不均匀、遮挡等问题。"
                             "\n2.车牌检测定位是从复杂背景的汽车图像中检测并定位车牌的位置。"
                             "\n3.字符分割是将车牌区域进一步分割成若干个单字符图像。"
                             "\n4.特征提取是提取易于区分字符的特征，用于字符识别。"
                             "\n5.字符识别包括模型训练和车牌识别两个阶段。")
    if img_open is None:
        showwarning(title='警告', message='请先在文件菜单下打开图片！')
        return
    predictor = predict.CardPredictor(myWindow)  # 创建识别模型
    predictor.stepFlag = askyesno("选择输出结果方式", "系统提供了分步显示识别数据和显示最终结果两种显示方式，请选择显示方式：是——分步显示，否——仅显示最终结果")
    if predictor.stepFlag:
        myWindow.setVisibleRight0()
    r,roi,color = predictor.predict(img_open)  #用训练的模型进行识别
    myWindow.setVisibleRight1()
    if r :
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
        roi = Image.fromarray(roi)
        imgtk_roi = ImageTk.PhotoImage(image=roi)
        myWindow.roi_ctl.configure(image=imgtk_roi, state='normal')
        myWindow.roi_ctl.image = imgtk_roi
        myWindow.r_ctl.configure(text=str(r))
        myWindow.updataTime = time.time()
        try:
            c = myWindow.colorTransform[color]
            myWindow.color_ctl.configure(text=c[0], background=c[1], state='normal')
        except Exception as e:
            print(e)
            myWindow.color_ctl.configure(state='disabled')
    elif   myWindow.updataTime + 8 < time.time():
            myWindow.roi_ctl.configure(state='disabled')
            myWindow.r_ctl.configure(text="非车牌图片")
            myWindow.color_ctl.configure(state='disabled')

#------------------------------------------------------------
class MyApp:
    '''    主窗体类，对于显示系统界面    '''
    global picSize
    colorTransform = {"green": ("绿", "#55ff55"), "yello": ("黄", "#ffff00"), "blue": ("蓝", "#6666ff")}
    updataTime = 0
    def __init__(self, root):
        '''
        构造方法
        :param root: 主窗体的父类
        '''
        self.root = root
        self.root.title("基于python的图像处理算法演示系统v1.0")

        # 界面布局
        self.init_menu(self.root)  #初始化菜单
        self.init_widget(self.root)#创建显示控件
        self.setVisibleLeft()      #显示左窗格
        self.setVisibleRight0()    #显示右窗格
        self.hideRight1()          #隐藏车牌识别显示
        self.setVisibleBottom()    #显示下窗格
        self.hideFig0()            #隐藏右窗格的画布myCanvas0
        self.hideFig1()            #隐藏下窗格的画布myCanvas1
        self.hideExplain()         #隐藏下窗格的画布myCanvas2

    def init_widget(self,master):
        self.originalText = tkinter.StringVar()
        self.label1 = tkinter.Label(master, textvariable=self.originalText, font=("song", 12))
        self.originalText.set("原始图像")
        self.label3 = tkinter.Label(master, bg='gray86')  #用于显示原始图像
        self.resultText = tkinter.StringVar()
        self.label2 = tkinter.Label(master, textvariable=self.resultText, font=("song", 12))
        self.resultText.set("处理结果图")
        self.label4 = tkinter.Label(master, bg='gray86')  #用于显示处理结果
        self.regionLabel = tkinter.Label(master, text='检测的车牌区域：', font=("song", 12))
        self.roi_ctl = tkinter.Label(master, font=("song", 12))
        self.recogResult = tkinter.Label(master, text='识别结果：', font=("song", 12))
        self.r_ctl = tkinter.Label(master,font=("song", 12),wraplength=400)
        self.color_ctl = tkinter.Label(master)
        self.from_pic_ctl = tkinter.Button(master, text="识别图片", width=20, command=Car_license_recognition, font=("song", 12))
        self.explainText = tkinter.StringVar()
        #用于显示算法注释说明
        self.label5 = tkinter.Label(master, textvariable=self.explainText, font=("song", 12),bg='gray86',wraplength=850,justify ='left')
        self.explainText.set("图像处理教学演示软件实现了图像处理中的许多经典算法。\n使用此软件需要先点文件-打开命令,打开一个图像，"
                             "原始图片将显示在左边窗格内;\n然后从菜单中调用某算法处理图片，处理的结果将显示在右边窗格内。")
        #myfig0在右边结果窗格的画布中显示matplot图片
        self.myFig0 = plt.figure(figsize=(4, 4), facecolor='#dddddd')
        self.myCanvas0 = FigureCanvasTkAgg(self.myFig0, master)
        #myfig1在下边窗格的画布中显示多张包含图片的注释说明
        self.myFig1 = plt.figure(figsize=(9, 1.8), facecolor='lightgray')
        #myfig2在下边窗格中显示说明与单张图片
        self.myFig2 = plt.figure(figsize=(4.5, 1.8), facecolor='lightgray')
        self.myCanvas1 = FigureCanvasTkAgg(self.myFig1, master)
        self.myCanvas2 = FigureCanvasTkAgg(self.myFig2, master)
    def setVisibleLeft(self):
        self.label1.place(relx=0.25, rely=0.04, anchor=tkinter.CENTER)  # 设置相对座标
        self.label3.place(relx=0.25, rely=0.40, width=picSize, height=picSize,anchor=tkinter.CENTER)  # 设置绝对座标
    def setVisibleRight0(self):
        '''
        右边窗格重叠的显示之一（right0）
        right0一般算法结果显示窗格
        right1-用于显示车牌识别结果
        myFig0-使用mayplot显示运行结果图片
        '''
        self.label2.place(relx=0.75, rely=0.04, anchor=tkinter.CENTER)
        self.label4.place(relx=0.75, rely=0.40, width=picSize, height=picSize,anchor=tkinter.CENTER)
        self.hideRight1()
        self.hideFig0()
    def hideRight0(self):
        self.label2.place_forget()
        self.label4.place_forget()

    def setVisibleRight1(self):
        '''
        右边窗格重叠的显示之一（right1）
        right0一般算法结果显示窗格
        right1-用于显示车牌识别结果
        myFig0-使用mayplot显示运行结果图片
        '''
        self.regionLabel.place(relx=0.75, rely=0.1, anchor=tkinter.CENTER)
        self.roi_ctl.place(relx=0.75, rely=0.18, width=picSize, height=50,anchor=tkinter.CENTER)
        self.recogResult.place(relx=0.75, rely=0.26, anchor=tkinter.CENTER)
        self.r_ctl.place(relx=0.75, rely=0.34, width=picSize, height=50,anchor=tkinter.CENTER)
        self.color_ctl.place(relx=0.75, rely=0.42, width=picSize, height=30,anchor=tkinter.CENTER)
        self.from_pic_ctl.place(relx=0.75, rely=0.67,anchor=tkinter.CENTER)
        self.hideRight0()
        self.hideFig0()
    def hideRight1(self):
        self.regionLabel.place_forget()
        self.roi_ctl.place_forget()
        self.recogResult.place_forget()
        self.r_ctl.place_forget()
        self.color_ctl.place_forget()
        self.from_pic_ctl.place_forget()
    def setVisibleFig0(self):
        '''
        右边窗格重叠的显示之一（myFig0）
        right0一般算法结果显示窗格
        right1-用于显示车牌识别结果
        myFig0-使用mayplot显示运行结果图片
        '''
        self.myCanvas0.get_tk_widget().place(relx=0.75, rely=0.40, anchor=tkinter.CENTER)
        self.myCanvas0.draw()
    def hideFig0(self):
        self.myCanvas0.get_tk_widget().place_forget()  # 隐藏画布self

    def setVisibleBottom(self):
        '''
        下边窗格的重叠显示之一（label5)
        label5-仅显示文字说明
        myFig1-用matplot显示多个文字图片
        explain-左边是label5,右边是myFig2
        '''
        self.label5.place(relx=0.50, rely=0.86, width=900, height=180,anchor=tkinter.CENTER)
        self.hideFig1()
        self.hideExplain()

    def setVisibleFig1(self):
        '''
        下边窗格的重叠显示之一（myFig1)
        label5-仅显示文字说明
        myFig1-用matplot显示多个文字图片
        explain-左边是label5,右边是myFig2
        '''
        self.myCanvas1.get_tk_widget().place(relx=0.5, rely=0.86, anchor=tkinter.CENTER)
        self.myCanvas1.draw()
    def hideFig1(self):
        self.myCanvas1.get_tk_widget().place_forget()  # 隐藏画布self

    def setVisibleExplain(self):
        '''
        下边窗格的重叠显示之一（explain)
        label5-仅显示文字说明
        myFig1-用matplot显示多个文字图片
        explain-左边是label5,右边是myFig2
        '''
        self.label5.place(relx=0.275, rely=0.86, width=450, height=180,anchor=tkinter.CENTER)
        self.label5.config(wraplength=430)
        self.myCanvas2.get_tk_widget().place(relx=0.725, rely=0.86, anchor=tkinter.CENTER)
        self.myCanvas2.draw()
    def hideExplain(self):
        self.label5.config(wraplength=850)
        self.myCanvas2.get_tk_widget().place_forget()  # 隐藏画布self
    def init_menu(self,master):
        '''创建菜单'''

        menubar = tkinter.Menu(master)# 创建菜单对象
        # 创建菜单项
        fmenu_file = tkinter.Menu(master)  # 文件
        fmenu_file.add_command(label='打开', command=Choosepic)
        fmenu_file.add_command(label='重载', command=Reload)
        fmenu_file.add_command(label='转换为GRAY', command=Convert_gray)
        fmenu_file.add_command(label='转换为HSV', command=Convert_HSV)
        fmenu_file.add_command(label='转换为BGR', command=Convert_BGR)
        fmenu_file.add_command(label='保存处理结果', command=Savepic)
        fmenu_file.add_separator()  # 添加一条分隔线
        fmenu_file.add_command(label='退出', command=quit)  # 用tkinter里面自带的quit()函数

        fmenu_point = tkinter.Menu(master)  # 灰度点运算
        fmenu_point.add_command(label='图像取反', command=Negative)
        fmenu_point.add_command(label='全局灰度线性变换', command=Global_linear_transmation)
        fmenu_point.add_command(label='分段线性灰度变换', command=Piecewise_linear_transformation)
        fmenu_point.add_command(label='对数变换', command=Logarithmic_transformations)
        fmenu_point.add_command(label='幂次（伽马）变换', command=Power_law_transformations)
        submenu_point_1 = tkinter.Menu(fmenu_point)
        fmenu_point.add_cascade(label='直方图处理', menu=submenu_point_1)
        submenu_point_1.add_command(label='灰度直方图', command=Gray_histogram)
        submenu_point_1.add_command(label='掩模直方图', command=Mask_histogram)
        # submenu2_2.add_command(label='局部直方图', command=Locality_histogram)
        submenu_point_1.add_command(label='直方图均衡化', command=Equalization_histogram)
        submenu_point_11 = tkinter.Menu(submenu_point_1)
        submenu_point_1.add_cascade(label='直方图规定化', menu=submenu_point_11)
        submenu_point_11.add_command(label='SML单映射', command=SML)
        submenu_point_11.add_command(label='GML组映射', command=GML)

        fmenu_spatial = tkinter.Menu(master)  # 空域滤波
        submenu_spatial_2 = tkinter.Menu(fmenu_point)  #空域平滑
        submenu_spatial_3 = tkinter.Menu(fmenu_point)  #边界处理
        submenu_spatial_4 = tkinter.Menu(fmenu_point)   #空域锐化

        fmenu_spatial.add_cascade(label='空域平滑', menu=submenu_spatial_2)
        submenu_spatial_2.add_command(label='高斯平滑滤波', command=Gaussian_filter)
        submenu_spatial_2.add_command(label='中值滤波', command=Median_filter)
        submenu_spatial_2.add_command(label='最大值滤波', command=Max_filter)
        submenu_spatial_2.add_command(label='最小值滤波', command=Min_filter)
        submenu_spatial_2.add_command(label='自定义平滑滤波', command=Smooth_filter)
        fmenu_spatial.add_cascade(label='扩展边界', menu=submenu_spatial_3)
        submenu_spatial_3.add_command(label='常量（补0）法（BORDER_CONSTANT）', command=Constant_border)
        submenu_spatial_3.add_command(label='复制法（BORDER_REFLICATE）', command=Replicate_border)
        submenu_spatial_3.add_command(label='反射法（BORDER_REFLECT）', command=Reflect_border)
        submenu_spatial_3.add_command(label='反射法1（BORDER_REFLECT_101）', command=Reflect_border_101)
        submenu_spatial_3.add_command(label='外包装法（BORDER_WRAP）', command=Wrap_border)
        fmenu_spatial.add_cascade(label='空域锐化', menu=submenu_spatial_4)
        submenu_spatial_4.add_command(label='Roberts算子', command=Roberts_filter)
        submenu_spatial_4.add_command(label='Prewitt算子', command=Prewitt_filter)
        submenu_spatial_4.add_command(label='Sobel算子', command=Sobel_filter)
        submenu_spatial_4.add_command(label='Scharr算子', command=Scharr_filter)
        submenu_spatial_4.add_command(label='laplacian算子', command=Laplacian_filter)
        submenu_spatial_4.add_command(label='LOG算子', command=Log_filter)
        submenu_spatial_4.add_command(label='canny算子', command=Canny_filter)

        fmenu_spectrum = tkinter.Menu(master)  # 频域增强
        submenu_spectrum_1 = tkinter.Menu(fmenu_spectrum)
        submenu_spectrum_2 = tkinter.Menu(fmenu_spectrum)
        submenu_spectrum_3 = tkinter.Menu(fmenu_spectrum)
        fmenu_spectrum.add_cascade(label='傅里叶正变换', menu=submenu_spectrum_1)
        submenu_spectrum_1.add_command(label='幅值谱', command=Magnitude_spectrum0)
        submenu_spectrum_1.add_command(label='低频移中的幅值谱', command=Magnitude_spectrum1)
        submenu_spectrum_1.add_command(label='相位谱', command=Phase_spectrum0)
        submenu_spectrum_1.add_command(label='低频移中的相位谱', command=Phase_spectrum1)
        fmenu_spectrum.add_command(label='傅里叶反变换', command=Inverse_fourier)
        fmenu_spectrum.add_cascade(label='低通滤波器', menu=submenu_spectrum_2)
        submenu_spectrum_2.add_command(label='理想低通滤波器', command=Idle_low_pass)
        submenu_spectrum_2.add_command(label='巴特沃兹低通滤波器', command=Butterworth_low_pass)
        submenu_spectrum_2.add_command(label='高斯低通滤波器', command=Gaussian_low_pass)
        fmenu_spectrum.add_cascade(label='高通滤波器', menu=submenu_spectrum_3)
        submenu_spectrum_3.add_command(label='理想高通滤波器', command=Idle_high_pass)
        submenu_spectrum_3.add_command(label='巴特沃斯高通滤波器', command=Butterworth_high_pass)
        submenu_spectrum_3.add_command(label='高斯高通滤波器', command=Gaussian_high_pass)
        fmenu_spectrum.add_command(label='孤立频点与周期噪声', command=Property_fourier)

        fmenu_restore = tkinter.Menu(master)
        submenu_restore_1 = tkinter.Menu(fmenu_restore) # 图像去噪
        submenu_restore_2 = tkinter.Menu(fmenu_restore)  #退化模型
        submenu_restore_3 = tkinter.Menu(fmenu_restore)  # 空域去除特定噪声
        # submenu_restore_4 = tkinter.Menu(fmenu_restore)  # 频域去除周期噪声
        submenu_restore_5 = tkinter.Menu(fmenu_restore)  # 无约束图像复原
        submenu_restore_6 = tkinter.Menu(fmenu_restore)  #有约束图像复原
        fmenu_restore.add_cascade(label='添加噪声', menu=submenu_restore_1)
        submenu_restore_1.add_command(label='添加高斯噪声', command=Add_gaussian_noise)
        submenu_restore_1.add_command(label='添加椒盐噪声', command=Add_salt_and_pepper_noise)
        submenu_restore_1.add_command(label='添加均值噪声', command=Add_mean_noise)
        submenu_restore_1.add_command(label='添加瑞利噪声', command=Add_rayleigh_noise)
        submenu_restore_1.add_command(label='添加伽马噪声', command=Add_erlang_noise)

        # fmenu_restore.add_cascade(label='系统默认图片的噪声分布比较', menu=submenu_restore_1)
        # submenu_restore_1.add_command(label='添加高斯噪声', command=Add_gaussian_noise1)
        # submenu_restore_1.add_command(label='添加椒盐噪声', command=Add_salt_and_pepper_noise1)
        # submenu_restore_1.add_command(label='添加均值噪声', command=Add_mean_noise1)
        # submenu_restore_1.add_command(label='添加瑞利噪声', command=Add_rayleigh_noise1)
        # submenu_restore_1.add_command(label='添加伽马噪声', command=Add_erlang_noise1)

        fmenu_restore.add_cascade(label='退化模型', menu=submenu_restore_2)
        submenu_restore_2.add_command(label='运动模糊', command=Motion_blur)
        submenu_restore_2.add_command(label='高斯模糊', command=Gaussian_blur)
        submenu_restore_2.add_command(label='大气湍流模糊', command=Turbulence_blur)
        fmenu_restore.add_cascade(label='去除噪声', menu=submenu_restore_3)
        submenu_restore_3.add_command(label='中值滤波', command=Median_filter)
        submenu_restore_3.add_command(label='最大值滤波', command=Max_filter)
        submenu_restore_3.add_command(label='最小值滤波', command=Min_filter)
        submenu_restore_3.add_command(label='谐波滤波', command=Harmonic_filter)
        submenu_restore_3.add_command(label='逆谐波滤波', command=Contra_harmonic_filter)
        # submenu_restore_3.add_command(label='Alpha剪枝滤波', command=Median_filter)
        # submenu_restore_3.add_command(label='自适应滤波', command=Median_filter)
        # fmenu_restore.add_cascade(label='频域去除周期噪声', menu=submenu_restore_4)
        # submenu_restore_4.add_command(label='带阻滤波器', command=quit)
        # submenu_restore_4.add_command(label='带通滤波器', command=quit)
        # submenu_restore_4.add_command(label='陷波带阻滤波器', command=quit)
        fmenu_restore.add_cascade(label='无约束图像复原', menu=submenu_restore_5)
        submenu_restore_5.add_command(label='逆滤波', command=Inverse)
        submenu_restore_5.add_command(label='伪逆滤波', command=Improved_inverse)
        fmenu_restore.add_cascade(label='有约束图像复原', menu=submenu_restore_6)
        submenu_restore_6.add_command(label='维纳滤波器', command=Wiener)
        submenu_restore_6.add_command(label='最小二乘滤波', command=Constrained_least_squares)

        fmenu_geometric = tkinter.Menu(master)  # 几何变换
        fmenu_geometric.add_command(label='平移', command=Move)
        fmenu_geometric.add_command(label='旋转', command=Rotate)
        fmenu_geometric.add_command(label='垂直镜像', command=Reflect_x)
        fmenu_geometric.add_command(label='水平镜像', command=Reflect_y)
        fmenu_geometric.add_command(label='缩放', command=Zoom)
        fmenu_geometric.add_command(label='仿射变换', command=Affine)
        fmenu_geometric.add_command(label='透视变换', command=Perspective)
        fmenu_geometric.add_command(label='插值方法', command=Interpolation)

        fmenu_morphology = tkinter.Menu(master)  # 形态学处理
        submenu_morphology1= tkinter.Menu(fmenu_morphology)
        fmenu_morphology.add_cascade(label='系统默认图像的形态学处理', menu=submenu_morphology1)
        submenu_morphology1.add_command(label='二值化处理', command=Binary1)
        submenu_morphology1.add_command(label='膨胀运算', command=Dilate1)
        submenu_morphology1.add_command(label='腐蚀运算', command=Erode1)
        submenu_morphology1.add_command(label='开运算', command=Open_operation1)
        submenu_morphology1.add_command(label='闭运算', command=Close_operation1)
        submenu_morphology1.add_command(label='形态学梯度', command=Gradient_morphology1)
        submenu_morphology1.add_command(label='礼帽', command=Tophat1)
        submenu_morphology1.add_command(label='黑帽', command=Blackhat1)
        submenu_morphology1.add_command(label='孔洞填充', command=Fill_holes1)
        submenu_morphology1.add_command(label='骨架提取', command=Skeleton1)

        fmenu_morphology.add_command(label='二值化处理', command=Binary)
        fmenu_morphology.add_command(label='膨胀运算', command=Dilate)
        fmenu_morphology.add_command(label='腐蚀运算', command=Erode)
        fmenu_morphology.add_command(label='开运算', command=Open_operation)
        fmenu_morphology.add_command(label='闭运算', command=Close_operation)
        fmenu_morphology.add_command(label='形态学梯度', command=Gradient_morphology)
        fmenu_morphology.add_command(label='礼帽', command=Tophat)
        fmenu_morphology.add_command(label='黑帽', command=Blackhat)
        fmenu_morphology.add_command(label='孔洞填充', command=Fill_holes)
        fmenu_morphology.add_command(label='骨架提取', command=Skeleton)


        fmenu_segment = tkinter.Menu(master)  # 图像分割
        submenu_segment_1 = tkinter.Menu(fmenu_segment)
        submenu_segment_2=tkinter.Menu(fmenu_segment)
        fmenu_segment.add_cascade(label='系统默认图像的阈值分割', menu=submenu_segment_1)
        submenu_segment_1.add_command(label='基本全阈值分割', command=Basic_global_thresholding1)
        submenu_segment_1.add_command(label='最大类间方差OTSU分割', command=Otsu_threshold1)
        submenu_segment_1.add_command(label='邻域均值自适应阈值分割',command=Adaptive_mean_threshold1)
        submenu_segment_1.add_command(label='高斯窗口自适应阈值分割',command=Adaptive_gaussian_threshold1)
        # submenu_segment_1.add_command(label='局部滑动平均阈值分割', command=Moving_threshold1)

        # fmenu_segment.add_command(label='阈值估计', command=quit)
        fmenu_segment.add_command(label='基本全阈值分割', command=Basic_global_thresholding)
        fmenu_segment.add_command(label='最大类间方差OTSU分割', command=Otsu_threshold)
        fmenu_segment.add_cascade(label='自适应局部可变阈值分割', menu=submenu_segment_2)
        submenu_segment_2.add_command(label='邻域均值自适应阈值分割',command=Adaptive_mean_threshold)
        submenu_segment_2.add_command(label='高斯窗口自适应阈值分割',command=Adaptive_gaussian_threshold)
        # fmenu_segment.add_command(label='局部滑动平均阈值分割', command=Moving_threshold)

        fmenu_description = tkinter.Menu(master)  # 图像描述
        submenu_description1= tkinter.Menu(fmenu_description)
        fmenu_description.add_cascade(label='系统默认图像的描述', menu=submenu_description1)
        submenu_description1.add_command(label='获取目标轮廓', command=Draw_contours1)
        submenu_description1.add_command(label='区域描述子', command=Get_region_description1)
        submenu_description1.add_command(label='几何变换后的不变矩', command=Moment_invariants1)

        fmenu_description.add_command(label='获取目标轮廓', command=Draw_contours)
        fmenu_description.add_command(label='区域描述子', command=Get_region_description)
        fmenu_description.add_command(label='几何变换后的不变矩', command=Moment_invariants)

        fmenu_license = tkinter.Menu(master)  # 车牌识别系统
        fmenu_license.add_command(label='训练识别模型', command=Car_license_train)
        fmenu_license.add_command(label='车牌识别', command=Car_license_recognition)

        menubar.add_cascade(label="文件", menu=fmenu_file)
        menubar.add_cascade(label="灰度点运算", menu=fmenu_point)
        menubar.add_cascade(label="空域滤波", menu=fmenu_spatial)
        menubar.add_cascade(label="频域增强", menu=fmenu_spectrum)
        menubar.add_cascade(label="图像复原", menu=fmenu_restore)
        menubar.add_cascade(label="几何变换", menu=fmenu_geometric)
        menubar.add_cascade(label="形态学处理", menu=fmenu_morphology)
        menubar.add_cascade(label="图像分割", menu=fmenu_segment)
        menubar.add_cascade(label="图像描述", menu=fmenu_description)
        menubar.add_cascade(label="车牌识别系统", menu=fmenu_license)

        master.config(menu=menubar)

    def combine_images(self,picLists,axis=None,explain=False,fontSize=1):
        '''
        将图像合并为一个图片，并可以为图像添加文字说明
        :param picLists:[图像说明,图像](图像成员的维数必须相同)
        :param axis:合并方向。
        axis=None 按照图片的形状，自动选择合并方向
        axis=0时，图像垂直合并;
        axis = 1 时， 图像水平合并。
        :param explain:是否添加文字说明的标记
        :param fontSize:文字大小
        :return:合并后的图像
        '''
        if explain:
            images=[]
            # font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
            font=cv2.QT_FONT_NORMAL
            # fontSize = 12
            for i in range(len(picLists)):
                # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                images.append(cv2.putText(picLists[i][1], picLists[i][0], (5, 25), font, fontSize, (255,255, 0),2))
        else:   #无文字说明
            images=picLists
        # ndim = images[0].shape
        shapes = np.array([mat.shape for mat in images])
        # assert np.all(map(lambda e: len(e) == ndim, shapes)), 'all picLists should be same ndim.'
        if axis is None:
            if shapes[0, 0] < shapes[0, 1]:  # 根据图像的长宽比决定是合并方向，让合并后图像尽量是方形
                axis = 0
            else:
                axis = 1
        if axis == 0:  # 垂直方向合并图像
            cols = np.max(shapes[:, 1])
            # 扩展各图像 cols大小，使得 cols一致
            copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                            cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
            # 垂直方向合并
            return np.vstack(copy_imgs)
        else:  # 水平方向合并图像
            rows = np.max(shapes[:, 0])
            # 扩展各图像rows大小，使得 rows一致
            copy_imgs = [cv2.copyMakeBorder(img, 0, rows - img.shape[0], 0, 0,
                                            cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
            # 水平方向合并
            return np.hstack(copy_imgs)

    def showFig0(self,picLists,type="show"):
        '''
        在右边窗格的画布上用matplot显示多张结果图片
        :param picLists: 需要显示的图片及说明
        :param type: 显示的类型可以是imshow或bar
        '''
        self.myFig0.clear()
        plt.subplots_adjust(top=0.7)
        num=len(picLists)
        for i in range(num):
            f_plot = self.myFig1.add_subplot(1, num, i + 1)
            if type == 'show':
                f_plot.imshow(picLists[i][1], cmap='gray')
            elif type == 'bar':
                f_plot.bar(np.arange(256), picLists[i][1])
            plt.title(picLists[i][0],wrap=True)
        self.setVisibleFig0()

    def showAndBarFig0(self,pic,histr):
        '''
        在右边窗格用matplotlib显示图像及其直方图
        :param pic: 需要显示的图片
        :param histr: 需要显示的直方图
        '''
        self.myFig0.clear()
        height, width = pic.shape
        if width >= height:
            f_plot1 = self.myFig0.add_subplot(211)
            f_plot2 = self.myFig0.add_subplot(212)
        else:
            f_plot1 = self.myFig0.add_subplot(121)
            f_plot2 = self.myFig0.add_subplot(122)
        f_plot1.imshow(pic, cmap='gray')
        f_plot2.bar(np.arange(256), histr.flatten(), color='blue')
        self.setVisibleFig0()

    def showFig1(self,picLists,type='show'):
        '''
        此函数用于显示算法说明，包括多个文字和图片说明
        picLists[i][0]存放的是文字说明，picLists[i][1]存放的是图片
        type：是图片显示方式, type='show'是通过imshow显示，type='bar'是通过plt.bar显示，
        '''
        self.myFig1.clear()
        self.setVisibleFig1()
        # plt.subplots_adjust(top=0.7)
        num=len(picLists)
        for i in range(num):
            f_plot = self.myFig1.add_subplot(1, num, i+1)
            if type=='show':
                f_plot.imshow(picLists[i][1], cmap='gray')
                f_plot.xaxis.set_major_locator(plt.NullLocator())  #不显示坐标轴
                f_plot.yaxis.set_major_locator(plt.NullLocator())
                f_plot.spines['top'].set_visible(False)  #不显示边框
                f_plot.spines['right'].set_visible(False)
                f_plot.spines['bottom'].set_visible(False)
                f_plot.spines['left'].set_visible(False)
            elif type=='bar':
                f_plot.bar(np.arange(256), picLists[i][1])
                f_plot.xaxis.set_major_locator(plt.NullLocator())  # 不显示坐标轴
                f_plot.yaxis.set_major_locator(plt.NullLocator())
                f_plot.spines['top'].set_visible(False)  # 不显示边框
                f_plot.spines['right'].set_visible(False)
                f_plot.spines['bottom'].set_visible(False)
                f_plot.spines['left'].set_visible(False)
            f_plot.set_title(picLists[i][0],verticalalignment='center')
        self.setVisibleFig1()

    def showExplain(self, explain,picLists,type='show'):
        '''
        在下边窗格显示文字和图片说明 ,文字和图片各占一半
        :param explain:需要显示的文字
        :param picLists: 文字和图片
        :param type: 图片显示方式imshow和bar
        '''
        self.setVisibleExplain()
        self.explainText.set(explain)
        self.myFig2.clear()
        # plt.subplots_adjust(top=0.7)
        num = len(picLists)
        for i in range(num):
            f_plot = self.myFig2.add_subplot(1, num, i + 1)
            if type == 'show':
                if len(picLists[i][1].shape)==2:
                    f_plot.imshow(picLists[i][1], cmap=plt.cm.gray)
                else:
                    f_plot.imshow(picLists[i][1], cmap= plt.cm.jet)
                f_plot.xaxis.set_major_locator(plt.NullLocator())  # 不显示坐标轴
                f_plot.yaxis.set_major_locator(plt.NullLocator())
                f_plot.spines['top'].set_visible(False)  # 不显示边框
                f_plot.spines['right'].set_visible(False)
                f_plot.spines['bottom'].set_visible(False)
                f_plot.spines['left'].set_visible(False)
            elif type == 'bar':
                f_plot.bar(np.arange(256), picLists[i][1])
                f_plot.xaxis.set_major_locator(plt.NullLocator())  # 不显示坐标轴
                f_plot.yaxis.set_major_locator(plt.NullLocator())
                f_plot.spines['top'].set_visible(False)  # 不显示边框
                f_plot.spines['right'].set_visible(False)
                f_plot.spines['bottom'].set_visible(False)
                f_plot.spines['left'].set_visible(False)
            f_plot.set_title(picLists[i][0],verticalalignment='center')
        self.setVisibleExplain()

    def hide(self):
        '''隐藏窗体'''
        self.root.withdraw()

    def show(self):
        '''显示窗体'''
        self.root.update()
        self.root.deiconify()

# --------------------------------------
class paraWindow(tkinter.Toplevel):
    '''
    多行一列的参数设置窗口
    '''
    def __init__(self, root, title = None, paraLists=[]):
        """Constructor"""
        self.root = root
        self.paraLists=paraLists
        self.names = locals()   #动态组件
        tkinter.Toplevel.__init__(self,root)
        if title:
            self.title(title)
            # 创建对话框的主体内容
        frame = tkinter.Frame(self)
        # 调用init_widgets方法来初始化对话框界面
        self.initial_focus = self.init_widgets(frame)
        frame.pack(padx=5, pady=5)
        # 根据modal选项设置是否为模式对话框
        self.grab_set()   #重要，必须是模式对话框
        # 为"WM_DELETE_WINDOW"协议使用self.cancel_click事件处理方法
        self.protocol("WM_DELETE_WINDOW", self.cancel_click)
        # 根据父窗口来设置对话框的位置
        self.geometry("+%d+%d" % (root.winfo_rootx() + 100, root.winfo_rooty() + 100))
        # 让对话框获取焦点
        self.initial_focus.focus_set()
        self.wait_window(self)

    def init_widgets(self,master):
        '''创建自定义对话框的内容'''
        nrow=0
        for i in range(len(self.paraLists)):
            str0 = self.paraLists[i][0]
            nrow = i + 1
            labelMessage = tkinter.Label(master, text=str0, font=("song", 12))
            labelMessage.grid(row=nrow, column=0)
            self.names['paraV' + str(i)] = tkinter.StringVar()
            self.names['paraE' + str(i)] = tkinter.Entry(master, textvariable=self.names['paraV' + str(i)], width=20)
            self.names['paraV' + str(i)].set(self.paraLists[i][1])
            self.names['paraE' + str(i)].grid(row=nrow, column=1)#控件按列排列
        b1 = tkinter.Button(master, text='确定退出', command=self.setPara)
        b1.grid(row=nrow + 1, column=1)
        self.bind("<Return>", self.setPara)
        self.bind("<Escape>", self.cancel_click)
        return self.names['paraE0']

    def cancel_click(self, event=None):
        showwarning(title='警告', message='必须先设置参数')
        self.initial_focus.focus_set()

    def setPara(self):
        '''通过对话框设置参数'''
        for i in range(len(self.paraLists)):
            text0 = self.names['paraV' + str(i)].get()
            if not self.on_validate(text0):          # 如果不能通过校验，让用户重新输入
                showwarning(title='警告', message='必须输入数字')
                self.names['paraV' + str(i)].set(self.paraLists[i][1])
                self.names['paraE'+ str(i)].focus_set()
                return
            else:
                self.paraLists[i][1] = float(text0)
        self.hide()

    def on_validate(self,content):
        '''该方法可对用户输入的数据进行校验，保证输入的是数字'''
        for i in range(len(content) - 1, -1, -1):
            if not (48 <= ord(content[i]) <= 57 or content[i] == "." or content[i] =="+" or content[i] =="-" ):
                return False
        return True

    def hide(self):
        '''销毁对话框'''
        self.withdraw()
        self.update_idletasks()
        # 将焦点返回给父窗口
        self.root.focus_set()
        # 销毁自己
        self.destroy()
        self.root.update()
        self.root.deiconify()
        self.root.focus_set()
# ----------------------------------------------------------------------
class paraWindow2(tkinter.Toplevel):
    '''多行多列参数设置窗口    '''
    def __init__(self, root, paraLists,title,explain):
        """Constructor"""
        self.root = root
        self.names=locals()
        self.paraLists=paraLists
        self.explain=explain
        tkinter.Toplevel.__init__(self,root)
        # self.geometry("400x300")
        self.title(title)
        # 调用init_widgets方法来初始化对话框界面
        self.initial_focus = self.init_widgets()
        # 根据modal选项设置是否为模式对话框
        self.grab_set()
        # 为"WM_DELETE_WINDOW"协议使用self.cancel_click事件处理方法
        self.protocol("WM_DELETE_WINDOW", self.cancel_click)
        # 根据父窗口来设置对话框的位置
        self.geometry("+%d+%d" % (root.winfo_rootx() + 100, root.winfo_rooty() + 100))
        # print(self.initial_focus)
        # 让对话框获取焦点
        self.initial_focus.focus_set()
        self.wait_window(self)

    def init_widgets(self):
        frame1 = tkinter.Frame(self)
        explainLabel = tkinter.Label(frame1, text=self.explain, font=("song", 12))
        explainLabel.pack(padx=50, pady=5)
        frame1.pack(padx=5, pady=5)
        frame2 = tkinter.Frame(self)
        nrow=1
        for i in range(len(self.paraLists)):
            nrow = i + 3
            for j in range(len(self.paraLists[i])):
                self.names['ev' + str(i) + str(j)] = tkinter.StringVar()
                self.names['e' + str(i) + str(j)] = tkinter.Entry(frame2, textvariable=self.names['ev' + str(i) + str(j)],width=10)
                self.names['ev' + str(i) + str(j)].set(self.paraLists[i][j])
                self.names['e' + str(i) + str(j)].grid(row=nrow, column=j)
        b1 = tkinter.Button(frame2, text='确定退出', command=self.setPara)
        b1.grid(row=nrow + 1, column=1)
        frame2.pack(padx=5, pady=6)
        return b1

    def setPara(self):
        '''通过对话框给参数赋值'''
        for i in range(len(self.paraLists)):
            for j in range(len(self.paraLists[i])):
                text0 = self.names['ev' + str(i) + str(j)].get()
                if not self.on_validate(text0):  # 如果不能通过校验，让用户重新输入
                    showwarning(title='警告', message='必须输入数字')
                    self.names['ev' + str(i) + str(j)].set(self.paraLists[i][j])
                    self.names['e' + str(i) + str(j)].focus_set()
                    return
                else:
                    self.paraLists[i][j] = float(text0)
        self.hide()

    def on_validate(self,content):
        '''该方法可对用户输入的数据进行校验，保证输入的是数字'''
        for i in range(len(content) - 1, -1, -1):
            if not (48 <= ord(content[i]) <= 57 or content[i] == "." or content[i] == "+" or content[i] == "-"):
                return False
        return True
    def cancel_click(self, event=None):
        showwarning(title='警告', message='必须先设置参数')
        self.initial_focus.focus_set()
    def hide(self):
        """"""
        self.withdraw()
        self.update_idletasks()
        # 将焦点返回给父窗口
        self.root.focus_set()
        # 销毁自己
        self.destroy()
        self.root.update()
        self.root.deiconify()
        self.root.focus_set()
# ----------------------------------------------------------------------
class listWindow(tkinter.Toplevel):
    '''下拉列表窗口，用对参数设置'''
    def __init__(self, root, title = None, paraLists=[]):
        """Constructor"""
        self.root = root
        self.paraLists=paraLists
        tkinter.Toplevel.__init__(self,root)
        if title:
            self.title(title)
            # 创建对话框的主体内容
        # 调用init_widgets方法来初始化对话框界面
        self.initial_focus = self.init_widgets(self)
       # 根据modal选项设置是否为模式对话框
        self.grab_set()
        # 为"WM_DELETE_WINDOW"协议使用self.cancel_click事件处理方法
        self.protocol("WM_DELETE_WINDOW", self.cancel_click)
        # 根据父窗口来设置对话框的位置
        self.geometry("+%d+%d" % (root.winfo_rootx() + 100, root.winfo_rooty() + 100))
        # print(self.initial_focus)
        # 让对话框获取焦点
        self.initial_focus.focus_set()
        self.wait_window(self)

    def init_widgets(self,master):
        str0 = self.paraLists[0][0]
        labelMessage = tkinter.Label(master, text=self.paraLists[0][0], font=("song", 12))
        labelMessage.grid(row=0, column=0)
        self.listBox = tkinter.Listbox(master,selectmode=tkinter.SINGLE)
        self.listBox.grid(row=1, column=0)
        lists=self.paraLists[0][1]
        for item in lists:
            self.listBox.insert(tkinter.END, item)
        self.listBox.select_set(0)
        b1 = tkinter.Button(master, text='确定退出', command=self.setPara)
        b1.grid(row=2, column=0)
        self.bind("<Return>", self.setPara)
        self.bind("<Escape>", self.cancel_click)
        return self.listBox

    def cancel_click(self, event=None):
        showwarning(title='警告', message='必须先设置参数')
        self.initial_focus.focus_set()
    def setPara(self):
        self.returnValue=self.listBox.curselection()[0]
        self.hide()

    def hide(self):
        """"""
        self.withdraw()
        self.update_idletasks()
        # 将焦点返回给父窗口
        self.root.focus_set()
        # 销毁自己
        self.destroy()
        self.root.update()
        self.root.deiconify()
        self.root.focus_set()

#-------------------------------------------
if __name__ == "__main__":
    root = tkinter.Tk()
    root.geometry("1000x640+150+5")  # 界面大小,以相对屏幕的坐标
    root.resizable(0, 0)
    myWindow = MyApp(root)
    root.mainloop()