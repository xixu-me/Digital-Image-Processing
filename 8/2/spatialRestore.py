import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt

#给图像添加椒盐噪声
def add_salt_and_pepper_noise(src, percentage=0.1):
    #NoiseImg = src         #使用此语句传递的是地址，程序会出错
    NoiseImg = src.copy()   #在此要使用copy函数，否则src和主程序中的img都会跟着改变
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = np.random.randint(0, src.shape[0] - 1)  #产生[0, src.shape[0] - 1]之间随机整数
        randY = np.random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg
# 添加高斯噪声
def add_gaussian_noise(src,mu,sigma):
    NoiseImg=src.copy()
    NoiseImg=NoiseImg/NoiseImg.max()
    rows=NoiseImg.shape[0]
    cols=NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            NoiseImg[i,j]=NoiseImg[i,j]+np.random.normal(mu,sigma)
            if  NoiseImg[i,j]< 0:
                 NoiseImg[i,j]=0
            elif  NoiseImg[i,j]>1:
                 NoiseImg[i,j]=1
    NoiseImg=np.uint8(NoiseImg*255)
    return NoiseImg
# 添加Rayleigh噪声
# def add_rayleigh_noise(src,scale):
#     NoiseImg=src.copy()
#     NoiseImg=NoiseImg/NoiseImg.max()
#     rows=NoiseImg.shape[0]
#     cols=NoiseImg.shape[1]
#     for i in range(rows):
#         for j in range(cols):
#             NoiseImg[i,j]=NoiseImg[i,j]+np.random.rayleigh(scale)
#             if  NoiseImg[i,j]< 0:
#                  NoiseImg[i,j]=0
#             elif  NoiseImg[i,j]>1:
#                  NoiseImg[i,j]=1
#     NoiseImg=np.uint8(NoiseImg*255)
#     return NoiseImg

def add_mean_noise(img, a=50, b=150, percentage=0.5):
    '''添加均匀分布噪声'''
    NoiseImg = img.copy()
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        NoiseImg[randX, randY] = np.random.randint(a, b)
    return NoiseImg

def add_rayleigh_noise(img, a=50, b=150, percentage=1):
    '''添加瑞利噪声'''
    NoiseImg = img.copy()
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        NoiseImg[randX, randY] = a + (-b * np.log(1 - np.random.rand())) ** 0.5
    return NoiseImg

def add_erlang_noise(img, a=50, b=0.1, percentage=1):
    '''添加伽马噪声'''
    NoiseImg = img.copy()
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        NoiseImg[randX, randY] = a- np.log(1 - np.random.rand()) / b
    return NoiseImg


'''openCV的五种滤波器
3种线性滤波：
方框滤波boxFilter(g_srcImage, g_dstImage1, -1, Size(g_nBoxFilterValue+1,g_nBoxFilterValue+1));
均值滤波blur(g_srcImage,g_dstImage2,Size(g_nMeanBlurValue+1,g_nMeanBlurValue+1),Point(-1,-1));
高斯滤波 GaussianBlur(g_srcImage, g_dstImage3, Size(g_nGaussianBlurValue*2+1,g_nGaussianBlurValue*2+1), 0, 0);

2种非线性滤波：
中值滤波medianBlur(g_srcImage, g_dstImage4, g_nMedianBlurValue*2+1);
双边滤波bilateralFilter(g_srcImage, g_dstImage5, g_nBilateralFilterValue, g_nBilateralFilterValue*2, g_nBilateralFilterValue/2);
'''
def mean_filter(input_image, filter_size, title=''):
    '''均值滤波'''
    input_image_cp = np.copy(input_image)
    filter_template = np.ones((filter_size, filter_size))
    pad_num = int((filter_size - 1) / 2)
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)
    m, n = input_image_cp.shape
    output_image = np.copy(input_image_cp)
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_image[i, j] = np.mean(
                filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1])
    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]
    return output_image

def geometric_filter(input_image, filter_size, title=''):
    '''几何滤波'''
    input_image_cp = np.copy(input_image)
    filter_template = np.ones((filter_size, filter_size))
    pad_num = int((filter_size - 1) / 2)
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)
    m, n = input_image_cp.shape
    output_image = np.copy(input_image_cp)
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_image[i, j] = \
            np.cumprod(filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1])[
                -1] ** (1 / (filter_size * filter_size))
    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]

    return output_image


def median_filter(pic, filt_size, title=''):
    '''中值滤波'''
    pic = np.copy(pic)
    shape_x, shape_y = pic.shape
    pad_num = (filt_size - 1) // 2
    new_pic = np.zeros([shape_x + 2 * pad_num, shape_y + 2 * pad_num])
    new_pic[pad_num: pad_num + shape_x, pad_num: pad_num + shape_y] = pic
    for i in range(0, shape_x):
        for j in range(0, shape_y):
            mean = sorted(new_pic[i + pad_num - 1: i + filt_size, j: j + filt_size].reshape(-1))[filt_size**2 // 2]
            pic[i][j] = mean
    return pic

def max_filter(pic, filt_size, title=''):
    '''最大值滤波'''
    pic = np.copy(pic)
    shape_x, shape_y = pic.shape
    pad_num = (filt_size - 1) // 2
    new_pic = np.zeros([shape_x + 2 * pad_num, shape_y + 2 * pad_num])
    new_pic[pad_num: pad_num + shape_x, pad_num: pad_num + shape_y] = pic
    for i in range(0, shape_x):
        for j in range(0, shape_y):
            mean = np.max(new_pic[i: i + filt_size, j: j + filt_size])
            pic[i][j] = mean
    #     plt.imshow(pic, cmap='gray')
    #     plt.title(f'max_filt-{title}')
    #     plt.show()
    return pic

def min_filter(pic, filt_size, title=''):
    '''最小值滤波'''
    pic = np.copy(pic)
    shape_x, shape_y = pic.shape
    pad_num = (filt_size - 1) // 2
    new_pic = np.zeros([shape_x + 2 * pad_num, shape_y + 2 * pad_num])
    new_pic[pad_num: pad_num + shape_x, pad_num: pad_num + shape_y] = pic
    for i in range(0, shape_x):
        for j in range(0, shape_y):
            mean = np.min(new_pic[i: i + filt_size, j: j + filt_size])
            pic[i][j] = mean
    #     plt.imshow(pic, cmap='gray')
    #     plt.title(f'min_filt-{title}')
    #     plt.show()
    return pic

def harmonic_filter(input_image, filter_size, title=''):
    '''谐波滤波'''
    input_image_cp = np.copy(input_image)
    filter_template = np.ones((filter_size, filter_size))
    pad_num = int((filter_size - 1) / 2)
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)
    m, n = input_image_cp.shape
    output_image = np.copy(input_image_cp)
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_image[i, j] = 1 / np.mean(
                1 / (filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]))
    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]
    #     plt.imshow(output_image, cmap='gray')
    #     plt.title(f'harmonic_filter{title}')
    #     plt.show()
    return output_image

def contra_harmonic_filter(input_image, filter_size, Q, title=''):
    '''逆谐波滤波'''
    input_image_cp = np.copy(input_image)
    filter_template = np.ones((filter_size, filter_size))
    pad_num = int((filter_size - 1) / 2)
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)
    m, n = input_image_cp.shape
    output_image = np.copy(input_image_cp)
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_image[i, j] = np.sum(
                np.power((filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]) + 0.0001,
                         Q + 1)) / np.sum(
                np.power((filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]) + 0.0001,
                         Q))
    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]
    #     plt.imshow(output_image, cmap='gray')
    #     plt.title(f'contra_harmonic_filter-{title}')
    #     plt.show()
    return output_image

def medpoint_filt(pic, filt_size, title=''):
    '''中点滤波'''
    pic = np.copy(pic)
    shape_x, shape_y = pic.shape
    pad_num = (filt_size - 1) // 2
    new_pic = np.zeros([shape_x + 2 * pad_num, shape_y + 2 * pad_num])
    new_pic[pad_num: pad_num + shape_x, pad_num: pad_num + shape_y] = pic
    for i in range(0, shape_x):
        for j in range(0, shape_y):
            mean = (np.max(new_pic[i: i + filt_size, j: j + filt_size]) + np.min(
                new_pic[i: i + filt_size, j: j + filt_size])) / 2.
            pic[i][j] = mean
    #     plt.imshow(pic, cmap='gray')
    #     plt.title(f'mepoint_filt-{title}')
    #     plt.show()
    return pic

def alpha_filter(input_image, filter_size, d, title=''):
    '''修改的alpha滤波'''
    input_image_cp = np.copy(input_image)#深拷贝
    filter_template = np.ones((filter_size, filter_size))#创建滤波核
    pad_num = int((filter_size - 1) / 2)#给图像周围添加元素值
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)
    m, n = input_image_cp.shape
    output_image = np.copy(input_image_cp)
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            output_image[i, j] = np.sum(np.sort((filter_template * input_image_cp[i - pad_num:i + pad_num + 1,
                                       j - pad_num:j + pad_num + 1]).reshape(1, -1) )[0][d // 2:- (d // 2)] ) / (filter_size ** 2 - d)
    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]
    #     plt.imshow(output_image, cmap='gray')
    #     plt.title(f'alpha_filter-{title}')
    #     plt.show()
    return output_image

def adaptive_median_filter(pic, max_size):
    pic = np.copy(pic)
    shape_x, shape_y = pic.shape
    pad_num = (max_size - 1) // 2
    new_pic = np.zeros([shape_x + 2 * pad_num, shape_y + 2 * pad_num])
    new_pic[pad_num: pad_num + shape_x, pad_num: pad_num + shape_y] = pic
    for i in range(0, shape_x):
        for j in range(0, shape_y):
            temp_pad = 1
            while True:
                temp = new_pic[i + pad_num - temp_pad: i + pad_num + temp_pad + 1,
                       j + pad_num - temp_pad: j + pad_num + temp_pad + 1].reshape(-1)
                min_ = np.min(temp)
                max_ = np.max(temp)
                med_ = sorted(temp)[((temp_pad * 2 + 1) ** 2 - 1) // 2]
                a1 = med_ - min_
                a2 = med_ - max_
                if a1 > 0 and a2 < 0:
                    b1 = new_pic[i + 1, j + 1] - min_
                    b2 = new_pic[i + 1, j + 1] - max_
                    if b1 > 0 and b2 < 0:
                        pic[i][j] = new_pic[i + 1, j + 1]
                        break
                    else:
                        pic[i][j] = med_
                        break
                else:
                    temp_pad += 1
                    if temp_pad * 2 + 1 > max_size:
                        pic[i][j] = med_
                        break
    return pic


def cross_plot(func, img, sizes, name=''):
    plt.figure(figsize=[15, 5])
    imgs = []
    for i in range(len(sizes)):
        plt.subplot(131 + i)
        result = func(img, sizes[i])
        imgs.append(result)
        plt.imshow(result, cmap='gray')
        plt.title(f'{name}-filtsize:{sizes[i]}')
    plt.savefig(f'img_/{name}')
    plt.show()
    return imgs


def cross_contra_plot(func, img, sizes, name=''):
    plt.figure(figsize=[15, 5])
    imgs = []
    for i in range(len(sizes)):
        plt.subplot(131 + i)
        result = func(img, sizes[i], 1.5)
        imgs.append(result)
        plt.imshow(result, cmap='gray')
        plt.title(f'{name}-filtsize:{sizes[i]}')
    plt.savefig(f'img_/{name}')
    plt.show()
    return imgs


def cross_alpha_plot(func, img, ds, name=''):
    plt.figure(figsize=[15, 5])
    imgs = []
    for i in range(len(ds)):
        plt.subplot(131 + i)
        result = func(img, 5, ds[i])
        imgs.append(result)
        plt.imshow(result, cmap='gray')
        plt.title(f'{name}-d:{ds[i]}')
    plt.savefig(f'img_/{name}')
    plt.show()
    return imgs


def cross_autofit_meanplot(func, img, maxsizes, name=''):
    plt.figure(figsize=[15, 5])
    imgs = []
    for i in range(len(maxsizes)):
        plt.subplot(131 + i)
        result = func(img, maxsizes[i])
        imgs.append(result)
        plt.imshow(result, cmap='gray')
        plt.title(f'{name}-max:{maxsizes[i]}')
    plt.savefig(f'img_/{name}')
    plt.show()
    return imgs

def count_dB(img_a, img_b):
    '''计算信噪比 单位dB
        img_a:原图
        img_b:滤波之后的图像
    '''
    img_c = img_a - img_b
    img_c = img_c.reshape(-1)
    var_a = np.sum(img_a ** 2)
    var_c = np.sum(img_c ** 2)
    snr = var_a / var_c
    return np.log10(snr) * 10


def count_mse(img_a, img_b):
    '''计算均方差
        img_a:原图
        img_b:滤波之后的图像
    '''
    img_ = img_a - img_b
    result = np.mean(img_ ** 2) * (1 / img_.shape[0]) * (1 / img_.shape[1])
    return result


def count_feek_dB(img_a, img_b):
    '''峰值信噪比
        img_a:原图
        img_b:滤波之后的图像
    '''
    mse_ = count_mse(img_a, img_b)
    return 10 * np.log10(255 ** 2 / mse_)


def plot_dB(xs, img_, name, func=count_dB):
    dBs = []
    for img__ in img_:
        dB = func(img, img__)
        dBs.append(dB)
    plt.plot(xs, dBs, '-*')
    plt.title(name)
    temp = ''
    if func == count_mse:
        temp = 'mse'
    elif func == count_feek_dB:
        temp = 'pnsr'
    elif func == count_dB:
        temp = 'nsr'
    plt.savefig(f'img_/{temp}-{name}')
    plt.show()

if __name__=='__main__':
    # 原图像
    img = cv2.imread(r'.\img\lenna.png', 0)

    # 噪声图像
    '''per = .006
    gassi_img = add_gaussian_noise(img, var=40, percentage=per)  # 高斯噪声
    salt_pepper_img = add_salt_and_pepper_noise(img, per)  # 椒盐噪声
    mean_img = add_mean_noise(img, 0, 252, percentage=per)  # 均值噪声
    rayleigh_img = add_rayleigh_noise(img, 0, 5000, percentage=per)  # 瑞利噪声
    erlang_img = add_erlang_noise(img, 0.03, percentage=per)  # 伽马噪声
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.savefig('./img_/Lenna.png')
    plt.subplot(232)
    plt.imshow(gassi_img, cmap='gray')
    plt.savefig('./img_/gassi_img.png')
    plt.subplot(233)
    plt.imshow(salt_pepper_img, cmap='gray')
    plt.savefig('./img_/salt_pepper_img.png')
    plt.subplot(234)
    plt.imshow(rayleigh_img, cmap='gray')
    plt.savefig('./img_/rayleigh_img.png')
    plt.subplot(235)
    plt.imshow(mean_img, cmap='gray')
    plt.savefig('./img_/mean_img.png')
    plt.subplot(236)
    plt.imshow(erlang_img, cmap='gray')
    plt.savefig('./img_/erlang_img.png')
    plt.show()'''

    salt_result_lst = []
    papper_result_lst = []
    Qs = np.linspace(-3, 3, 13)
    for metric in zip([count_dB, count_feek_dB, count_mse], ['dB', 'feek_dB', 'mse']):
        for i in Qs:
            result_1 = contra_harmonic_filter(salt_img, 3, i)
            salt_result_lst.append(metric[0](img, result_1))
            plt.show()
            result_2 = contra_harmonic_filter(papper_img, 3, i)
            papper_result_lst.append(metric[0](img, result_2))
        plt.plot(Qs, salt_result_lst, '-*', label='salt')
        plt.plot(Qs, papper_result_lst, '-*', label='papper')
        plt.legend()
        plt.title('{}-{}'.format(metric[1], i))
        plt.show()
        salt_result_lst = []
        papper_result_lst = []

    # 均值滤波处理
    mean_gass_img = cross_plot(mean_filter, gassi_img, [3, 5, 7], 'mean-gassi')
    mean_salt_pepper_img = cross_plot(mean_filter, salt_pepper_img, [3, 5, 7], 'mean-salt_pepper')
    mean_mean_img = cross_plot(mean_filter, mean_img, [3, 5, 7], 'mean-mean')
    mean_rayleigh_img = cross_plot(mean_filter, rayleigh_img, [3, 5, 7], 'mean-rayleigh')
    mean_erlang_img = cross_plot(mean_filter, erlang_img, [3, 5, 7], 'mean-erlang')
    # 几何滤波处理
    geometri_gass_img = cross_plot(geometric_filter, gassi_img, [3, 5, 7], 'geometric-gassi')
    geometri_pepper_img = cross_plot(geometric_filter, salt_pepper_img, [3, 5, 7], 'geometric-salt_pepper')
    geometri_mean_img = cross_plot(geometric_filter, mean_img, [3, 5, 7], 'geometric-mean')
    geometri_rayleigh_img = cross_plot(geometric_filter, rayleigh_img, [3, 5, 7], 'geometric-rayleigh')
    geometri_erlang_img = cross_plot(geometric_filter, erlang_img, [3, 5, 7], 'geometric-erlang')
    # 谐波滤波处理
    harmonic_gass_img = cross_plot(harmonic_filter, gassi_img, [3, 5, 7], 'harmonic-gassi')
    harmonic_pepper_img = cross_plot(harmonic_filter, salt_pepper_img, [3, 5, 7], 'harmonic-salt_pepper')
    harmonic_mean_img = cross_plot(harmonic_filter, mean_img, [3, 5, 7], 'harmonic-mean')
    harmonic_rayleigh_img = cross_plot(harmonic_filter, rayleigh_img, [3, 5, 7], 'harmonic-rayleigh')
    harmonic_erlang_img = cross_plot(harmonic_filter, erlang_img, [3, 5, 7], 'harmonic-erlang')
    # 逆谐波滤波处理
    contra_gass_img = cross_contra_plot(contra_harmonic_filter, gassi_img, [3, 5, 7], 'contra_harmonic-gassi')
    contra_pepper_img = cross_contra_plot(contra_harmonic_filter, salt_pepper_img, [3, 5, 7], 'contra_harmonic-salt_pepper')
    contra_mean_img = cross_contra_plot(contra_harmonic_filter, mean_img, [3, 5, 7], 'contra_harmonic-mean')
    contra_rayleigh_img = cross_contra_plot(contra_harmonic_filter, rayleigh_img, [3, 5, 7], 'contra_harmonic-rayleigh')
    contra_erlang_img = cross_contra_plot(contra_harmonic_filter, erlang_img, [3, 5, 7], 'contra_harmonic-erlang')

    # 最大值滤波处理
    max_gass_img = cross_plot(max_filt, gassi_img, [3, 5, 7], 'max-gassi')
    max_salt_pepper_img = cross_plot(max_filt, salt_pepper_img, [3, 5, 7], 'max-salt_pepper')
    max_mean_img = cross_plot(max_filt, mean_img, [3, 5, 7], 'max-mean')
    max_rayleigh_img = cross_plot(max_filt, rayleigh_img, [3, 5, 7], 'max-rayleigh')
    max_erlang_img = cross_plot(max_filt, erlang_img, [3, 5, 7], 'max-erlang')
    # 最小值滤波处理
    min_gass_img = cross_plot(min_filt, gassi_img, [3, 5, 7], 'min-gassi')
    min_pepper_img = cross_plot(min_filt, salt_pepper_img, [3, 5, 7], 'min-salt_pepper')
    min_mean_img = cross_plot(min_filt, mean_img, [3, 5, 7], 'min-mean')
    min_rayleigh_img = cross_plot(min_filt, rayleigh_img, [3, 5, 7], 'min-rayleigh')
    min_erlang_img = cross_plot(min_filt, erlang_img, [3, 5, 7], 'min-erlang')
    # 中值滤波处理
    media_gass_img = cross_plot(media_filt, gassi_img, [3, 5, 7], 'media-gassi')
    media_pepper_img = cross_plot(media_filt, salt_pepper_img, [3, 5, 7], 'media-salt_pepper')
    media_mean_img = cross_plot(media_filt, mean_img, [3, 5, 7], 'media-mean')
    media_rayleigh_img = cross_plot(media_filt, rayleigh_img, [3, 5, 7], 'media-rayleigh')
    media_erlang_img = cross_plot(media_filt, erlang_img, [3, 5, 7], 'media-erlang')
    # 中点滤波处理
    medpoint_gass_img = cross_plot(medpoint_filt, gassi_img, [3, 5, 7], 'medpoint-gassi')
    medpoint_pepper_img = cross_plot(medpoint_filt, salt_pepper_img, [3, 5, 7], 'medpoint-salt_pepper')
    medpoint_mean_img = cross_plot(medpoint_filt, mean_img, [3, 5, 7], 'medpoint-mean')
    medpoint_rayleigh_img = cross_plot(medpoint_filt, rayleigh_img, [3, 5, 7], 'medpoint-rayleigh')
    medpoint_erlang_img = cross_plot(medpoint_filt, erlang_img, [3, 5, 7], 'medpoint-erlang')
    # 自适应均值滤波处理
    autofit_gassi_img = cross_autofit_meanplot(autofit_mean_filt, gassi_img, [3, 5, 7], 'autofit-gassi')
    autofit_salt_pepper_img = cross_autofit_meanplot(autofit_mean_filt, salt_pepper_img, [3, 5, 7], 'autofit-salt_pepper')
    autofit_mean_img = cross_autofit_meanplot(autofit_mean_filt, mean_img, [3, 5, 7], 'autofit-mean')
    autofit_rayleigh_img = cross_autofit_meanplot(autofit_mean_filt, rayleigh_img, [3, 5, 7], 'autofit-rayleigh')
    autofit_erlang_img = cross_autofit_meanplot(autofit_mean_filt, erlang_img, [3, 5, 7], 'autofit-erlang')
    # alpha滤波处理
    alpha_gassi_img = cross_alpha_plot(alpha_filter, gassi_img, [2, 4, 6], 'alpha-gassi')
    alpha_salt_pepper_img = cross_alpha_plot(alpha_filter, salt_pepper_img, [2, 4, 6], 'alpha-salt_pepper')
    alpha_mean_img = cross_alpha_plot(alpha_filter, mean_img, [2, 4, 6], 'alpha-mean')
    alpha_rayleigh_img = cross_alpha_plot(alpha_filter, rayleigh_img, [2, 4, 6], 'alpha-rayleigh')
    alpha_erlang_img = cross_alpha_plot(alpha_filter, erlang_img, [2, 4, 6], 'alpha-erlang')




    print('====================nsr=================')
    plot_dB([3, 5, 7], mean_gass_img, 'mean_gass-filtisize', func=count_mse)
    plot_dB([3, 5, 7], mean_salt_pepper_img, 'mean_salt_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], mean_mean_img, 'mean_mean-filtsize', func=count_mse)
    plot_dB([3, 5, 7], mean_rayleigh_img, 'mean_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], mean_erlang_img, 'mean_erlang-filtsize', func=count_mse)

    plot_dB([3, 5, 7], geometri_gass_img, 'geometri_gass-filtsize', func=count_mse)
    plot_dB([3, 5, 7], geometri_pepper_img, 'geometri_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], geometri_mean_img, 'geometri_mean-fitlsize', func=count_mse)
    plot_dB([3, 5, 7], geometri_rayleigh_img, 'geometri_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], geometri_erlang_img, 'geometri_erlang-filtsize', func=count_mse)

    plot_dB([3, 5, 7], harmonic_gass_img, 'harmonic_gass-filtsize', func=count_mse)
    plot_dB([3, 5, 7], harmonic_pepper_img, 'harmonic_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], harmonic_mean_img, 'harmonic_mean-filtsize', func=count_mse)
    plot_dB([3, 5, 7], harmonic_rayleigh_img, 'harmonic_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], harmonic_erlang_img, 'harmonic_erlang-filtsize', func=count_mse)

    plot_dB([3, 5, 7], contra_gass_img, 'contra_gass-filtsize', func=count_mse)
    plot_dB([3, 5, 7], contra_pepper_img, 'contra_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], contra_mean_img, 'contra_mean-filtsize', func=count_mse)
    plot_dB([3, 5, 7], contra_rayleigh_img, 'contra_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], contra_erlang_img, 'contra_erlang-filtsize', func=count_mse)

    plot_dB([3, 5, 7], max_gass_img, 'max_gass-filtsize', func=count_mse)
    plot_dB([3, 5, 7], max_salt_pepper_img, 'max_salt_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], max_mean_img, 'max_mean-filtsize', func=count_mse)
    plot_dB([3, 5, 7], max_rayleigh_img, 'max_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], max_erlang_img, 'max_erlang-filtsize', func=count_mse)

    plot_dB([3, 5, 7], min_gass_img, 'min_gass-filtsize', func=count_mse)
    plot_dB([3, 5, 7], min_pepper_img, 'min_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], min_mean_img, 'min_mean-filtsize', func=count_mse)
    plot_dB([3, 5, 7], min_rayleigh_img, 'min_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], min_erlang_img, 'min_erlang-filtsize', func=count_mse)

    plot_dB([3, 5, 7], media_gass_img, 'media_gass-filtsize', func=count_mse)
    plot_dB([3, 5, 7], media_pepper_img, 'media_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], media_mean_img, 'media_mean-filtsize', func=count_mse)
    plot_dB([3, 5, 7], media_rayleigh_img, 'media_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], media_erlang_img, 'media_erlang-filtsize', func=count_mse)

    plot_dB([3, 5, 7], medpoint_gass_img, 'medpoint_gass-filtsize', func=count_mse)
    plot_dB([3, 5, 7], medpoint_pepper_img, 'medpoint_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], medpoint_mean_img, 'medpoint_mean-filtsize', func=count_mse)
    plot_dB([3, 5, 7], medpoint_rayleigh_img, 'medpoint_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], medpoint_erlang_img, 'medpoint_erlang-filtsize', func=count_mse)

    plot_dB([2, 4, 6], alpha_gassi_img, 'alpha_gass-filtsize', func=count_mse)
    plot_dB([2, 4, 6], alpha_salt_pepper_img, 'alpha_pepper-filtsize', func=count_mse)
    plot_dB([2, 4, 6], alpha_mean_img, 'alpha_mean-filtsize', func=count_mse)
    plot_dB([2, 4, 6], alpha_rayleigh_img, 'alpha_rayleigh-filtsize', func=count_mse)
    plot_dB([2, 4, 6], alpha_erlang_img, 'alpha_erlang-filtsize', func=count_mse)

    plot_dB([3, 5, 7], autofit_gassi_img, 'autofit_gass-filtsize', func=count_mse)
    plot_dB([3, 5, 7], autofit_salt_pepper_img, 'autofit_pepper-filtsize', func=count_mse)
    plot_dB([3, 5, 7], autofit_mean_img, 'autofit_mean-filtsize', func=count_mse)
    plot_dB([3, 5, 7], autofit_rayleigh_img, 'autofit_rayleigh-filtsize', func=count_mse)
    plot_dB([3, 5, 7], autofit_erlang_img, 'autofit_erlang-filtsize', func=count_mse)

    print('===================nsr==================')
    plot_dB([3, 5, 7], mean_gass_img, 'mean_gass-filtisize')
    plot_dB([3, 5, 7], mean_salt_pepper_img, 'mean_salt_pepper-filtsize')
    plot_dB([3, 5, 7], mean_mean_img, 'mean_mean-filtsize')
    plot_dB([3, 5, 7], mean_rayleigh_img, 'mean_rayleigh-filtsize')
    plot_dB([3, 5, 7], mean_erlang_img, 'mean_erlang-filtsize')

    plot_dB([3, 5, 7], geometri_gass_img, 'geometri_gass-filtsize')
    plot_dB([3, 5, 7], geometri_pepper_img, 'geometri_pepper-filtsize')
    plot_dB([3, 5, 7], geometri_mean_img, 'geometri_mean-fitlsize')
    plot_dB([3, 5, 7], geometri_rayleigh_img, 'geometri_rayleigh-filtsize')
    plot_dB([3, 5, 7], geometri_erlang_img, 'geometri_erlang-filtsize')

    plot_dB([3, 5, 7], harmonic_gass_img, 'harmonic_gass-filtsize')
    plot_dB([3, 5, 7], harmonic_pepper_img, 'harmonic_pepper-filtsize')
    plot_dB([3, 5, 7], harmonic_mean_img, 'harmonic_mean-filtsize')
    plot_dB([3, 5, 7], harmonic_rayleigh_img, 'harmonic_rayleigh-filtsize')
    plot_dB([3, 5, 7], harmonic_erlang_img, 'harmonic_erlang-filtsize')

    plot_dB([3, 5, 7], contra_gass_img, 'contra_gass-filtsize')
    plot_dB([3, 5, 7], contra_pepper_img, 'contra_pepper-filtsize')
    plot_dB([3, 5, 7], contra_mean_img, 'contra_mean-filtsize')
    plot_dB([3, 5, 7], contra_rayleigh_img, 'contra_rayleigh-filtsize')
    plot_dB([3, 5, 7], contra_erlang_img, 'contra_erlang-filtsize')

    plot_dB([3, 5, 7], max_gass_img, 'max_gass-filtsize')
    plot_dB([3, 5, 7], max_salt_pepper_img, 'max_salt_pepper-filtsize')
    plot_dB([3, 5, 7], max_mean_img, 'max_mean-filtsize')
    plot_dB([3, 5, 7], max_rayleigh_img, 'max_rayleigh-filtsize')
    plot_dB([3, 5, 7], max_erlang_img, 'max_erlang-filtsize')

    plot_dB([3, 5, 7], min_gass_img, 'min_gass-filtsize')
    plot_dB([3, 5, 7], min_pepper_img, 'min_pepper-filtsize')
    plot_dB([3, 5, 7], min_mean_img, 'min_mean-filtsize')
    plot_dB([3, 5, 7], min_rayleigh_img, 'min_rayleigh-filtsize')
    plot_dB([3, 5, 7], min_erlang_img, 'min_erlang-filtsize')

    plot_dB([3, 5, 7], media_gass_img, 'media_gass-filtsize')
    plot_dB([3, 5, 7], media_pepper_img, 'media_pepper-filtsize')
    plot_dB([3, 5, 7], media_mean_img, 'media_mean-filtsize')
    plot_dB([3, 5, 7], media_rayleigh_img, 'media_rayleigh-filtsize')
    plot_dB([3, 5, 7], media_erlang_img, 'media_erlang-filtsize')

    plot_dB([3, 5, 7], medpoint_gass_img, 'medpoint_gass-filtsize')
    plot_dB([3, 5, 7], medpoint_pepper_img, 'medpoint_pepper-filtsize')
    plot_dB([3, 5, 7], medpoint_mean_img, 'medpoint_mean-filtsize')
    plot_dB([3, 5, 7], medpoint_rayleigh_img, 'medpoint_rayleigh-filtsize')
    plot_dB([3, 5, 7], medpoint_erlang_img, 'medpoint_erlang-filtsize')

    plot_dB([2, 4, 6], alpha_gassi_img, 'alpha_gass-filtsize')
    plot_dB([2, 4, 6], alpha_salt_pepper_img, 'alpha_pepper-filtsize')
    plot_dB([2, 4, 6], alpha_mean_img, 'alpha_mean-filtsize')
    plot_dB([2, 4, 6], alpha_rayleigh_img, 'alpha_rayleigh-filtsize')
    plot_dB([2, 4, 6], alpha_erlang_img, 'alpha_erlang-filtsize')

    plot_dB([3, 5, 7], autofit_gassi_img, 'autofit_gass-filtsize')
    plot_dB([3, 5, 7], autofit_salt_pepper_img, 'autofit_pepper-filtsize')
    plot_dB([3, 5, 7], autofit_mean_img, 'autofit_mean-filtsize')
    plot_dB([3, 5, 7], autofit_rayleigh_img, 'autofit_rayleigh-filtsize')
    plot_dB([3, 5, 7], autofit_erlang_img, 'autofit_erlang-filtsize')

    print('=================mse===================')
    plot_dB([3, 5, 7], mean_gass_img, 'mean_gass-filtisize', func=count_feek_dB)
    plot_dB([3, 5, 7], mean_salt_pepper_img, 'mean_salt_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], mean_mean_img, 'mean_mean-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], mean_rayleigh_img, 'mean_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], mean_erlang_img, 'mean_erlang-filtsize', func=count_feek_dB)

    plot_dB([3, 5, 7], geometri_gass_img, 'geometri_gass-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], geometri_pepper_img, 'geometri_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], geometri_mean_img, 'geometri_mean-fitlsize', func=count_feek_dB)
    plot_dB([3, 5, 7], geometri_rayleigh_img, 'geometri_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], geometri_erlang_img, 'geometri_erlang-filtsize', func=count_feek_dB)

    plot_dB([3, 5, 7], harmonic_gass_img, 'harmonic_gass-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], harmonic_pepper_img, 'harmonic_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], harmonic_mean_img, 'harmonic_mean-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], harmonic_rayleigh_img, 'harmonic_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], harmonic_erlang_img, 'harmonic_erlang-filtsize', func=count_feek_dB)

    plot_dB([3, 5, 7], contra_gass_img, 'contra_gass-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], contra_pepper_img, 'contra_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], contra_mean_img, 'contra_mean-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], contra_rayleigh_img, 'contra_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], contra_erlang_img, 'contra_erlang-filtsize', func=count_feek_dB)

    plot_dB([3, 5, 7], max_gass_img, 'max_gass-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], max_salt_pepper_img, 'max_salt_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], max_mean_img, 'max_mean-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], max_rayleigh_img, 'max_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], max_erlang_img, 'max_erlang-filtsize', func=count_feek_dB)

    plot_dB([3, 5, 7], min_gass_img, 'min_gass-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], min_pepper_img, 'min_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], min_mean_img, 'min_mean-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], min_rayleigh_img, 'min_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], min_erlang_img, 'min_erlang-filtsize', func=count_feek_dB)

    plot_dB([3, 5, 7], media_gass_img, 'media_gass-filtsize, func=count_feek_dB')
    plot_dB([3, 5, 7], media_pepper_img, 'media_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], media_mean_img, 'media_mean-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], media_rayleigh_img, 'media_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], media_erlang_img, 'media_erlang-filtsize', func=count_feek_dB)

    plot_dB([3, 5, 7], medpoint_gass_img, 'medpoint_gass-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], medpoint_pepper_img, 'medpoint_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], medpoint_mean_img, 'medpoint_mean-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], medpoint_rayleigh_img, 'medpoint_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], medpoint_erlang_img, 'medpoint_erlang-filtsize', func=count_feek_dB)

    plot_dB([2, 4, 6], alpha_gassi_img, 'alpha_gass-filtsize', func=count_feek_dB)
    plot_dB([2, 4, 6], alpha_salt_pepper_img, 'alpha_pepper-filtsize', func=count_feek_dB)
    plot_dB([2, 4, 6], alpha_mean_img, 'alpha_mean-filtsize', func=count_feek_dB)
    plot_dB([2, 4, 6], alpha_rayleigh_img, 'alpha_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([2, 4, 6], alpha_erlang_img, 'alpha_erlang-filtsize', func=count_feek_dB)

    plot_dB([3, 5, 7], autofit_gassi_img, 'autofit_gass-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], autofit_salt_pepper_img, 'autofit_pepper-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], autofit_mean_img, 'autofit_mean-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], autofit_rayleigh_img, 'autofit_rayleigh-filtsize', func=count_feek_dB)
    plot_dB([3, 5, 7], autofit_erlang_img, 'autofit_erlang-filtsize', func=count_feek_dB)
