import cv2
import numpy as np

def combine_images(images):
    '''
    合并图像。
    @param images: 图像列表(图像成员的维数必须相同)
    @param axis: 合并方向。
    axis=0时，图像垂直合并;
    axis = 1 时， 图像水平合并。
    @return 合并后的图像
    '''
    ndim = images[0].ndim
    shapes = np.array([mat.shape for mat in images])
    assert np.all(map(lambda e: len(e) == ndim, shapes)
                  ), 'all images should be same ndim.'
    if shapes[0, 0] < shapes[0, 1]:  # 根据图像的长宽比决定是合并方向，让合并后图像尽量是方形
        axis = 0
    else:
        axis = 1
    if axis == 0:  # 垂直方向合并图像
        # 合并图像的 cols
        cols = np.max(shapes[:, 1])
        # 扩展各图像 cols大小，使得 cols一致
        copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        # 垂直方向合并
        return np.vstack(copy_imgs)
    else:  # 水平方向合并图像
        # 合并图像的 rows
        rows = np.max(shapes[:, 0])
        # 扩展各图像rows大小，使得 rows一致
        copy_imgs = [cv2.copyMakeBorder(img, 0, rows - img.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        # 水平方向合并
        return np.hstack(copy_imgs)

def fft(img,shift=False):
    '''
    对图像进行傅立叶变换
    :param img: 待处理图片
    :param shift: 是否移中，True低频移中
    :return: 频率矩阵
    '''
    assert img.ndim == 2, 'img should be gray.'
    rows, cols = img.shape[:2]
    # 计算最优尺寸
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    # 根据新尺寸，建立新变换图像
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img
    # 傅立叶变换
    fft_mat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 换位，低频部分移到中间，高频部分移到四周
    if shift:
        return np.fft.fftshift(fft_mat)
    else:
        return fft_mat

def fft_magnitude(fft_mat):
    '''
    傅里叶变换的幅值谱
    :param fft_mat:频率矩阵
    :return: 幅值谱图像
    '''
    # log函数中加1，避免log(0)出现.
    log_mat = cv2.log(1 + cv2.magnitude(fft_mat[:, :, 0], fft_mat[:, :, 1]))
    # 标准化到0~255之间
    cv2.normalize(log_mat, log_mat, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(log_mat))

def fft_phase(fft_mat):
    '''获得相位谱'''
    phase = cv2.phase(fft_mat[:, :, 0], fft_mat[:, :, 1])
    # 标准化到0~255之间
    cv2.normalize(phase, phase, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(phase))

def ifft(fft_mat,shift=False):
    '''
    傅里叶反变换
    :param fft_mat: 频率矩阵
    :param shift: 是否低频移中
    :return: 返回反变换图像
    '''
    # 反换位，低频部分移到四周，高频部分移到中间
    if shift:
        f_ishift_mat = np.fft.ifftshift(fft_mat)
    else:
        f_ishift_mat=fft_mat
    # 傅立叶反变换
    img = cv2.idft(f_ishift_mat)
    # 将复数转换为幅度, sqrt(re^2 + im^2)
    img_back = cv2.magnitude(*cv2.split(img))  #*的意思是将任意个参数导入
    # 标准化到0~255之间
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(img_back))

def fft_distances(m, n):
    '''
    计算m,n矩阵每一点距离中心的距离,傅里叶变换已低频移中
    '''
    u = np.array([m/2-i if i <= m / 2 else i-m/2 for i in range(m)],
                 dtype=np.float32)
    v = np.array([n/2-i if i <= n / 2 else i-n/2 for i in range(n)],
                 dtype=np.float32)
    u.shape = m, 1
    # 每点距离矩阵左上角的距离
    ret = np.sqrt(u * u + v * v)
    ret.shape = (m,n)
    # 每点距离矩阵中心的距离
    return ret

def lpfilter(img_original, flag=0, d0=20, n=1):
    '''
    低通滤波器
    :param img_original: 待处理图像
    :param flag: 滤波器类型
    :param d0: 截止频率
    :param n: 滤波器的阶次
    :return: 返回滤波后的图像与滤波器的幅值谱
    '''
    fft_mat = fft(img_original,True)
    filter_mat = None
    rows,cols=fft_mat.shape[:2]
    # 理想低通滤波
    if flag == 0:
        filter_mat = np.zeros((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (cols // 2,rows // 2),
                   d0, (1, 1, 1), thickness=-1)#(1,1,1)是颜色 thickness=-1是实心
    # 巴特沃兹低通滤波
    elif flag == 1:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    # 高斯低通滤波
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = np.exp(-(duv * duv) / (2 * d0 * d0))
        filter_mat = cv2.merge((filter_mat, filter_mat))
    filtered_mat = filter_mat * fft_mat
    # 反变换
    img_back = ifft(filtered_mat, True)
    combined=combine_images([img_back, fft_magnitude(filter_mat)])
    return combined

def hpfilter(img_original,flag=0, d0=20, n=1):
    '''高通滤波器
    @param flag: 滤波器类型
    0 - 理想高通滤波
    1 - 巴特沃兹高通滤波
    2 - 高斯高通滤波
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param d0: 滤波器大小 D0
    @param n: 巴特沃兹高通滤波的阶数
    @return 滤波器矩阵
    '''
    assert d0 > 0, 'd0 should be more than 0.'
    fft_mat = fft(img_original, True)
    filter_mat = None
    rows, cols = fft_mat.shape[:2]
    # 理想高通滤波
    if flag == 0:
        filter_mat = np.ones((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (cols // 2, rows // 2), d0, (0, 0, 0), thickness=-1)
    # 巴特沃兹高通滤波
    elif flag == 1:
        duv = fft_distances(rows, cols)
        # duv有 0 值(中心距离中心为0)， 为避免除以0，设中心为 0.000001
        duv[rows // 2, cols // 2] = 0.000001
        filter_mat = 1 / (1 + np.power(d0 / duv, 2 * n))
        filter_mat = cv2.merge((filter_mat, filter_mat))
    # 高斯高通滤波
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 - np.exp(-(duv * duv) / (2 * d0 * d0))
        filter_mat = cv2.merge((filter_mat, filter_mat))
    filtered_mat = filter_mat * fft_mat
    # 反变换
    img_back = ifft(filtered_mat, True)
    combined = combine_images([img_back, fft_magnitude(filter_mat)])
    return combined

def standardization(img):
    #将0-255范围的图像转化为0-1，将彩色图像转换为灰度图像
    if len(img.shape)>2 :#判断img.shape元组的长度
        [m, n, k] = img.shape
        if k!=1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else :              #img.shape长度为2，则是灰度图像
        [m ,n] = img.shape
    for i in range(m):
        for j in range(n):
            if(img[i,j]!=0):
                if(img[i,j]>1):
                    img = img / 255
                    break
    return img

if __name__ == '__main__':
    img_original = cv2.imread(r'.\img\kenny.jpg', 0)
    rows, cols = img_original.shape[:2]
    # 滤波器窗口名称
    # filter_win = 'Filter Parameters'
    # # 图像窗口名称
    # image_win = 'Filtered Image'
    # cv2.namedWindow(filter_win)
    # cv2.namedWindow(image_win)
    # # 创建d0 tracker, d0为过滤器大小
    # cv2.createTrackbar('d0', filter_win, 20, min(rows, cols) // 4, on_change)
    # # 创建flag tracker,
    # # flag=0时，为理想滤波
    # # flag=1时，为巴特沃兹滤波
    # # flag=2时，为高斯滤波
    # cv2.createTrackbar('flag', filter_win, 0, 2, on_change)
    # # 创建n tracker
    # # n 为巴特沃兹滤波的阶数
    # cv2.createTrackbar('n', filter_win, 1, 5, on_change)
    # # 创建lh tracker
    # # lh: 滤波器是低通还是高通， 0 为低通， 1为高通
    # cv2.createTrackbar('lh', filter_win, 0, 1, on_change)
    # lpfilter(img_original)
    # cv2.resizeWindow(filter_win, 512, 20)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
