from globalObject import *
from math import *
import math
from matplotlib import pyplot as plt
from tkinter.messagebox import showinfo, showwarning, showerror

#图像取反
def negative(im,L=255):
    [m,n]=im.shape
    img=im.copy()
    for i in range(m):
        for j in range(n):
            img[i,j]=L-im[i,j]
    return img
# 全局灰度线性变换（也可对彩色图像进行线性变换）
# 全局灰度线性变换（也可对彩色图像进行线性变换）
def global_linear_transmation(im,c=0,d=255):
    img=im.copy()
    maxV = img.max()
    minV = img.min()
    if maxV==minV:
        return np.uint8(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = ((d-c) / (maxV - minV)) * (img[i, j] - minV)+c#img[i,j]代表的是某像素点三通道的值
    return np.uint8(img)

# 对img_result进行分段线性灰度变换
def piecewise_linear_transformation(im,lists): #lists存放着各段分段前后的灰度范围
    global img_result,img_empty
    try:
        img=im.copy()
        for list in lists:
            a = int(list[0])
            b = int(list[1])
            c = int(list[2])
            d = int(list[3])
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if(img[i,j]>=a and img[i,j]<=b):
                        img[i, j] = ((d- c) / (b-a)) * (im[i, j] - a)+c
    except:
        showerror("错误提示","灰度值设置不合理，起始灰度值不能与终止值相同")
        img = img_empty
    return img

#位平面分割
def Bit_Plane_Slicing(im):
    img=im.copy()
    BP=np.zeros([8,img.shape[0],img.shape[1]])
    for n in range(8):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if(img[i,j]>=2**(7-n)):
                    BP[n][i,j]=2**(7-n)
                    img[i,j]=img[i,j]-2**(7-n)
                else:
                    BP[n][i,j]=0
    return BP

#对数变换，默认不改变像素点的范围（0-255）
def logarithmic_transformations(im,c=1):
    img=np.zeros([im.shape[0],im.shape[1]])
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            img[i,j]=c*math.log(1+im[i,j])
    return img

#幂次（伽马）变换
def power_law_transformations(im,r=0.45,c=1):
    img=np.zeros([im.shape[0],im.shape[1]])
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            img[i,j]=c*255.0*(im[i,j]/255.0)**r
    return np.uint8(img)




if __name__=="__main__":
    im0 = cv2.imread(r'.\img\test.jpg')
    im0=cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)
    im1=cv2.cvtColor(im0,cv2.COLOR_RGB2GRAY)
    im1=piecewise_linear_transformation(im1, [[[0, 100], [0, 80]]])
    plt.subplot(121), plt.imshow(im0, cmap='gray')
    plt.title('Original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(im1, cmap='gray')
    plt.title('Piecewise linear gray enhancementt'), plt.xticks([]), plt.yticks([])
    plt.show()

'''
#傅里叶频谱的对数变换

im0=cv2.imread('2.jpg')
im0=cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)

im0=abs(np.fft.fftshift(np.fft.fft2(im0)))
im1=Logarithmic_Transformations(im0,1,exp(1))

plt.subplot(121), plt.imshow(im0, cmap='gray')
plt.title('Original spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(im1, cmap='gray')
plt.title('log Transformed Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

#加法运算去除“叠加性”随机噪音


def SaltAndPepper(src, percentage):
    #NoiseImg = src         #使用此语句传递的是地址，程序会出错
    NoiseImg = src.copy()   #在此要使用copy函数，否则src和主程序中的img都会跟着改变
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)  #产生[0, src.shape[0] - 1]之间随机整数
        randY = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg

im0=cv2.imread('2.jpg')
im0=cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)

[m,n]=im0.shape

im=np.zeros([100,m,n])

for i in range(100):
    im[i]=SaltAndPepper(im0,0.1)

im2=add(im)

plt.subplot(131), plt.imshow(im0, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(im[0], cmap='gray')
plt.title('0.1 Salt and pepper noise'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(im2, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.show()

#差影法的应用


im0=cv2.imread('2.jpg')
im0=cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
im1=cv2.imread('2.jpg')
im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

im2=im0-im1

plt.subplot(131), plt.imshow(im0, cmap='gray')
plt.title('image before'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(im1, cmap='gray')
plt.title('image after'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(im2, cmap='gray')
plt.title('image difference'), plt.xticks([]), plt.yticks([])
plt.show()

#用逻辑运算提取子图像


img=cv2.imread('2.jpg')
im0=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
im0=Thresholding(im0)
im=Bit_Plane_Slicing(im0)
im1=im[0]
im1=Negative(Thresholding(im1,1,1,1),1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))
im1 = cv2.morphologyEx(im1, cv2.MORPH_CLOSE, kernel)
im1 = cv2.morphologyEx(im1, cv2.MORPH_OPEN, kernel)

im2=img_and(im0,im1)

plt.subplot(131), plt.imshow(im0,cmap='gray')
plt.title('Original Two valued image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(im1,cmap='gray')
plt.title('intersection Two valued image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(im2,cmap='gray')
plt.title('Sub image'), plt.xticks([]), plt.yticks([])
plt.show()

#用乘法运算提取局部图像

def GetStructuringElement(path):
    global img, im1, im2
    img = cv2.imread(path)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    im0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = Bit_Plane_Slicing(im0)
    im1 = im[0]
    im1 = Negative(Thresholding(im1, 1, 1, 1), 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    im1 = cv2.morphologyEx(im1, cv2.MORPH_CLOSE, kernel)
    im1 = cv2.morphologyEx(im1, cv2.MORPH_OPEN, kernel)

    r = multiply(r, im1)
    g = multiply(g, im1)
    b = multiply(b, im1)
    im2 = cv2.merge([r, g, b])
    return im2
if __name__=="__main__":
    GetStructuringElement('2.jpg')
    plt.subplot(131), plt.imshow(img)
    plt.title('Original image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(im1, cmap='gray')
    plt.title('Two valued image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(im2)
    plt.title('Local image'), plt.xticks([]), plt.yticks([])
    plt.show()

'''
