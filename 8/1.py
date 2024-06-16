from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import os, math, cv2, struct
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


def load_mnist(path, labelfile, datafile):
    labels_path = os.path.join(path, labelfile)
    images_path = os.path.join(path, datafile)
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


features, labels = load_mnist(
    "mnist", "train-labels-idx1-ubyte", "train-images-idx3-ubyte"
)
print(features.shape, labels.shape)

features = features[0:6000, :]
labels = labels[0:6000]
print("训练集行数: %d, 列数: %d" % (features.shape[0], features.shape[1]))
x = np.array(features[0, :])
x = x.reshape([28, 28])
plt.figure(figsize=(12, 4))
plt.imshow(x, cmap="Greys")
plt.show()

testfeatures, testlabels = load_mnist(
    "mnist", "t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte"
)
print("测试集行数: %d, 列数: %d" % (testfeatures.shape[0], testfeatures.shape[1]))

list_hog_fd = []
for feature in features:
    fd = hog(
        feature.reshape((28, 28)),
        orientations=9,
        pixels_per_cell=(14, 14),
        cells_per_block=(1, 1),
        visualize=False,
    )
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, "float64")
print(hog_features.shape)
list_hog_fd = []
for feature in testfeatures:
    fd = hog(
        feature.reshape((28, 28)),
        orientations=9,
        pixels_per_cell=(14, 14),
        cells_per_block=(1, 1),
        visualize=False,
    )
    list_hog_fd.append(fd)
hog_testfeatures = np.array(list_hog_fd, "float64")
print(hog_testfeatures.shape)

from sklearn.neighbors import KNeighborsClassifier

k = 5
kNN_model = KNeighborsClassifier(n_neighbors=k)
kNN_model.fit(hog_features, labels)
testnumber = kNN_model.predict(hog_testfeatures)
KNNscore = kNN_model.score(hog_testfeatures, testlabels)
print("测试集第一个图像数字是：", testnumber[0])
print("分类精度：", KNNscore)

g_mapping = [
    0,
    1,
    2,
    3,
    4,
    58,
    5,
    6,
    7,
    58,
    58,
    58,
    8,
    58,
    9,
    10,
    11,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    12,
    58,
    58,
    58,
    13,
    58,
    14,
    15,
    16,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    17,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    18,
    58,
    58,
    58,
    19,
    58,
    20,
    21,
    22,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    23,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    24,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    25,
    58,
    58,
    58,
    26,
    58,
    27,
    28,
    29,
    30,
    58,
    31,
    58,
    58,
    58,
    32,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    33,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    34,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    35,
    36,
    37,
    58,
    38,
    58,
    58,
    58,
    39,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    40,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    41,
    42,
    43,
    58,
    44,
    58,
    58,
    58,
    45,
    58,
    58,
    58,
    58,
    58,
    58,
    58,
    46,
    47,
    48,
    58,
    49,
    58,
    58,
    58,
    50,
    51,
    52,
    58,
    53,
    54,
    55,
    56,
    57,
]


def LBP(I, radius=2, count=8):
    dh = np.round([radius * math.sin(i * 2 * math.pi / count) for i in range(count)])
    dw = np.round([radius * math.cos(i * 2 * math.pi / count) for i in range(count)])
    I = I.reshape(28, 28)
    height, width = I.shape
    lbp = np.zeros(I.shape, dtype=int)
    I1 = np.pad(I, radius, "edge")
    for k in range(count):
        h, w = int(radius + dh[k]), int(radius + dw[k])
        lbp += (I > I1[h : h + height, w : w + width]) << k
    return lbp


def calLbpHistogram(lbp, hCount=7, wCount=5, maxLbpValue=255):
    height, width = lbp.shape
    res = np.zeros((hCount * wCount, max(g_mapping) + 1), dtype=float)
    assert maxLbpValue + 1 == len(g_mapping)

    for h in range(hCount):
        for w in range(wCount):
            blk = lbp[
                height * h // hCount : height * (h + 1) // hCount,
                width * w // wCount : width * (w + 1) // wCount,
            ]
            hist1 = np.bincount(blk.ravel(), minlength=maxLbpValue)
            hist = res[h * wCount + w, :]
            for v, k in zip(hist1, g_mapping):
                hist[k] += v
            hist /= hist.sum()
    feature = res.reshape(res.shape[0] * res.shape[1], 1)
    return res


lbp_features = np.array([calLbpHistogram(LBP(d)).ravel() for d in features])
lbp_testfeatures = np.array([calLbpHistogram(LBP(d)).ravel() for d in testfeatures])

from sklearn.svm import LinearSVC

svm_model = LinearSVC(dual=False)
svm_model.fit(hog_features, labels)

test_est = svm_model.predict(hog_testfeatures)
SVMscore = svm_model.score(hog_testfeatures, testlabels)
print("测试集第一个图像数字是：", testlabels[0])
print("HOG+SVM分类精度：", SVMscore)

import sklearn.metrics as metrics

print("--------混淆矩阵-----------")
print(
    metrics.confusion_matrix(
        testlabels, test_est, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
)
print("--------分类评估报告-----------")
print(metrics.classification_report(testlabels, test_est))

print(testlabels.shape)

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(hog_features, labels)
testnumber = rf_model.predict(hog_testfeatures)
rf_score = rf_model.score(hog_testfeatures, testlabels)
print("第一个图像数字是：", testnumber[0])
print("分类精度：", rf_score)


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(hog_features, labels)
testnumber = lr_model.predict(hog_testfeatures)
lr_score = lr_model.score(hog_testfeatures, testlabels)
print("第一个图像数字是：", testnumber[0])
print("分类精度：", lr_score)
