# from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import os, math, cv2, struct
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负


# 读取MNIST数据
# path是数据文件夹的路径，labelfile是图像标注文件名,datafile是数据文件名
def load_mnist(path, labelfile, datafile):  # 读取数据函数
    # Load MNIST data from path
    labels_path = os.path.join(path, labelfile)
    images_path = os.path.join(path, datafile)

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


# 读取训练集    features是图片数据,labels是对应的标注
features, labels = load_mnist(
    r"mnist", "train-labels-idx1-ubyte", "train-images-idx3-ubyte"
)
print(features.shape, labels.shape)
# print(type(features),type(labels))


# 显示训练数据
features = features[0:6000, :]
labels = labels[0:6000]
print("训练集行数: %d, 列数: %d" % (features.shape[0], features.shape[1]))
x = np.array(features[0, :])  # 提取第一行数据
x = x.reshape([28, 28])
plt.figure(figsize=(12, 4))
# plt.subplots_adjust(top=0,bottom=-1)
plt.imshow(x, cmap="Greys")
plt.show()


# # 读取测试集
testfeatures, testlabels = load_mnist(
    r"mnist", "t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte"
)
print("测试集行数: %d, 列数: %d" % (testfeatures.shape[0], testfeatures.shape[1]))


# 提取训练集HOG特征
list_hog_fd = []
for feature in features:
    fd = hog(
        feature.reshape((28, 28)),  # hog 特征
        orientations=9,
        pixels_per_cell=(14, 14),
        cells_per_block=(1, 1),
        visualize=False,
    )
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, "float64")  # 训练集的HOG特征
print(hog_features.shape)
# 提取测试集HOG特征
list_hog_fd = []
for feature in testfeatures:
    fd = hog(
        feature.reshape((28, 28)),  # hog 特征
        orientations=9,
        pixels_per_cell=(14, 14),
        cells_per_block=(1, 1),
        visualize=False,
    )
    list_hog_fd.append(fd)
hog_testfeatures = np.array(list_hog_fd, "float64")  # 测试集的HOG
print(hog_testfeatures.shape)


# 使用SVM支持向量机分类器进行分类
from sklearn.svm import LinearSVC

# 建立并训练SVM模型
svm_model = LinearSVC()  # 建立SVM模型
svm_model.fit(hog_features, labels)  # 训练
# joblib.dump(clf, "digits_cls.pkl", compress=3)   # 模型保存

# #测试SVM模型
# svm_model = joblib.load("digits_cls.pkl")
test_est = svm_model.predict(hog_testfeatures)  # 测试单个图像，输出结果为图像所属类别
SVMscore = svm_model.score(hog_testfeatures, testlabels)
print("测试集第一个图像数字是：", testlabels[0])
print("HOG+SVM分类精度：", SVMscore)


import sklearn.metrics as metrics

print("--------混淆矩阵-----------")
print(
    metrics.confusion_matrix(
        testlabels, test_est, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
)  # 混淆矩阵
print("--------分类评估报告-----------")
print(metrics.classification_report(testlabels, test_est))
