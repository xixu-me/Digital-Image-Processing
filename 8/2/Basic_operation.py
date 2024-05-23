
# 统计各灰度值的像素个数
def histogram(image):
    (row, col) = image.shape
    hist = [0]*256
    for i in range(row):
        for j in range(col):
            hist[image[i, j]] += 1
    return hist

#MODE=0自适应阈值（算术平均法）二值化，MODE=ture，自定义阈值二值化
def thresholding(im,T=128,L=255,MODE=0):
    [m,n]=im.shape
    img=im.copy()
    if MODE==0:
        list = histogram(img)
        for i in range(len(list)):
            list[i]=list[i]/(m*n)
        dT=1
        while dT>=0.5:
            T1=0
            T2=0
            t=T
            for i in range(floor(T)):
               T1+=list[i]*i
            for i in range(floor(T)+1,256,1):
               T2+=list[i]*i
            T=(T1+T2)/2
            dT=abs(T-t)
    for i in range(m):
        for j in range(n):
            if img[i,j]>=T:
                img[i,j]=L
            else:
                img[i,j]=0
    return img



#多图像加运算并取平均
def add(list):#list=[im1,im2,im3,...,imn],list为三维列表
    img=np.zeros([list.shape[1],list.shape[2]])
    for i in range(list.shape[0]):
        img+=list[i]
    img=img/list.shape[0]
    return img

#减运算im1-im2
def reduce(im1,im2):
    img=im1-im2
    return img


#乘运算
def multiply(im1,im2):
    img=im1.copy()
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            img[i,j]=im1[i,j]*im2[i,j]
    return img

#除运算，im1/im2
def divide(im1,im2):
    img=im1/im2
    return img

#与运算
def img_and(im1,im2,L=255):
    img=np.zeros([im1.shape[0],im1.shape[1]])
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            if(im1[i,j] and im2[i,j]):
                img[i,j] = L
            else:
                img[i,j] = 0
    return img

#或运算
def img_or(im1,im2,L=255):
    img=np.zeros([im1.shape[0],im1.shape[1]])
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            if(im1[i,j] or im2[i,j]):
                img[i,j] = L
            else:
                img[i,j] = 0
    return  img

