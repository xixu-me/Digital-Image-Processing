#位平面分割
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
#位平面切片
import cv2
img0 = cv2.imread(r"..\img\bitSlice.jpg",0)
imgSlices=Bit_Plane_Slicing(img0)
images=[img0]
images.extend(imgSlices[:2])
shapes = np.array([mat.shape for mat in images])
cols = np.max(shapes[:, 1])
# 扩展各图像 cols大小，使得 cols一致
copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
# 水平方向合并
img1=np.hstack(copy_imgs)
images=imgSlices[2:5]
shapes = np.array([mat.shape for mat in images])
cols = np.max(shapes[:, 1])
# 扩展各图像 cols大小，使得 cols一致
copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]

img2=np.hstack(copy_imgs)
images=imgSlices[5:8]
shapes = np.array([mat.shape for mat in images])
cols = np.max(shapes[:, 1])
# 扩展各图像 cols大小，使得 cols一致
copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
img3=np.hstack(copy_imgs)
images=[img1]
images.extend([img2])
images.extend([img3])
shapes = np.array([mat.shape for mat in images])
cols = np.max(shapes[:, 1])
# 扩展各图像 cols大小，使得 cols一致
copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
imgAll=np.vstack(copy_imgs)
plt.figure(figsize=(8,8))
plt.imshow(imgAll,cmap='gray')
plt.figure(figsize=(10,10))
plt.subplot(331)
plt.imshow(img0,cmap='gray')
plt.title("原图")
plt.axis('off')  #不显示坐标轴
for i in range(8):
    plt.subplot(3,3,i+2)
    plt.imshow(imgSlices[i],cmap='gray')
    plt.title("Bit-plane "+str(7-i))
    plt.axis('off')  #不显示坐标轴
    plt.xticks([])
    plt.yticks([])

# plt.show()
plt.savefig("ch03-10.jpg")