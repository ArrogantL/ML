import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

from DataGenerator import generateTwoDimensionalData,generateThiDimensionalData

def gaussianMixtureModel(XX, k, minincrement=10 ** -5,tag=True):
    """
    使用kmeans方法将一组向量XX聚为k类。
    :param XX:样本
    :param k:聚类数量
    :param minincrement:E步最小增量，小于则停止迭代
    :return: pk, means, vars    即 混合概率，各聚类高斯均值，各聚类高斯协方差阵
    """
    #  convert list X to column vector X
    XX = np.array(XX)
    len_X = len(XX[0])
    # start: random model:P(K) means vars,Attention：mean[i]!=mean[j] for any i!=j,
    # otherwise they will get the same parameters during the iteration.
    # labels, means_array = kmeans(XX.tolist(), k)

    means_array=np.tile(np.random.random_sample(k), (len_X, 1)).T

    vars_array = np.array([np.eye(len_X)] * k)
    assert vars_array.shape == (k, len_X, len_X)
    pk_array = np.ones(k) * (1.0 / k)
    # P(K|X) rowX columnK
    pkx_array = np.zeros(len(XX) * k).reshape(len(XX), k)
    # start EM process
    while True:
        # E: estimate P(K|X)
        tmp = pkx_array.copy()
        pkx_array = recalPKX(pk_array, means_array, vars_array, XX)
        # increment is less than the threshold,return result
        increment = np.sum(np.abs(pkx_array - tmp))
        print(increment)
        if increment < minincrement:
            break
        # M: max likehood with respect model:P(K) means vars
        NK_array = np.sum(pkx_array, axis=0)
        for j in range(k):
            means_array[j] = np.sum(np.tile(pkx_array[:, j], (len_X, 1)).T * XX, axis=0) / NK_array[j]
        for j in range(k):
            tmp = np.array([pkx_array[i, j] * (np.mat(XX[i] - means_array[j]).T * np.mat(XX[i] - means_array[j])).A for i in
                            range(len(XX))])
            assert tmp.shape == (len(XX), len_X, len_X)
            vars_array[j] = np.sum(tmp, axis=0) / NK_array[j]
            assert vars_array[j].shape == (len_X, len_X)
    return pk_array, means_array, vars_array


def recalPKX(pk, means, vars, XX):
    """
    E步重新计算类后验概率矩阵P（Y|X）
    :param pk:
    :param means:
    :param vars:
    :param XX:
    :return: pkx_array
    """
    logpxi_array = np.zeros(len(XX) * len(pk)).reshape(len(XX), len(pk))
    pkx_array = np.zeros(len(XX) * len(pk)).reshape(len(XX), len(pk))
    for j in range(len(XX)):
        for i in range(len(pk)):
            try:
                logpxi_array[j, i] = multivariate_normal.logpdf(XX[j], mean=means[i], cov=vars[i]) + np.log(pk[i])
            except:
                print("Singular Matrix!!")
    for k in range(len(pk)):
        pkx_array[:, k] = np.sum(np.exp(logpxi_array - np.tile(logpxi_array[:, k], (len(pk), 1)).T), axis=1)
    pkx_array = 1.0 / pkx_array
    return pkx_array


def twoDimensionalClusterDisplay():
    clusternum = 3
    # 二维划分展示
    XX, Y = generateTwoDimensionalData(num=200)
    XX = np.array(XX)
    XX_train_array = XX[len(XX) // 2:]
    XX_test_array = XX[len(XX) // 2:]
    # XX_test_array = XX[:len(XX) // 2]
    # 使用一半样本来训练
    pk, means, vars = gaussianMixtureModel(XX_train_array, clusternum)
    # 绘制散点图
    colors = ['blue', 'red', 'black', 'green']
    pkx_array = recalPKX(pk, means, vars, XX_test_array)
    labels = np.argmax(pkx_array, axis=1)
    assert len(labels) == len(XX_test_array)
    plt.scatter(XX_test_array[:, 0], XX_test_array[:, 1], s=75, c=[colors[i] for i in labels], alpha=0.5)
    plt.show()
def thiDimensionalClusterDisplay():
    clusternum = 4
    # 二维划分展示
    XX, Y = generateThiDimensionalData(num=200,featurenum=clusternum)
    XX = np.array(XX)
    XX_train_array = XX[len(XX) // 2:]
    XX_test_array = XX[len(XX) // 2:]
    # XX_test_array = XX[:len(XX) // 2]
    # 使用一半样本来训练
    pk, means, vars = gaussianMixtureModel(XX_train_array, clusternum)
    # 绘制散点图
    colors = ['blue', 'red', 'black', 'green']
    pkx_array = recalPKX(pk, means, vars, XX_test_array)
    labels = np.argmax(pkx_array, axis=1)
    assert len(labels) == len(XX_test_array)

    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(XX_test_array[:, 0], XX_test_array[:, 1],XX_test_array[:, 2], s=75, c=[colors[i] for i in labels], alpha=0.5 ) # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()




def GMMandAnalyze(X_train, X_test, y_test, k):
    """
    train model with X_train, test model with X_test:y_test
    :param X_train:
    :param X_test:
    :param y_test:
    :param k:
    :return: pk, means, vars, TP(correct clustering in X_test:y_test)
    """
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    y_test = np.array(y_test)
    pk, means, vars = gaussianMixtureModel(X_train, k)
    pkx_array = recalPKX(pk, means, vars, X_test)
    labels = np.argmax(pkx_array, axis=1)
    TP = 0
    cluster={}
    for i in range(len(labels)):
        cluster.setdefault(labels[i],[])
        cluster[labels[i]].append(i)
    for v in cluster.values():
        vy=y_test[v]
        right=sorted(vy)[0]
        for i in vy:
            if i ==right:
                TP+=1


    return pk, means, vars, TP


def processUCIheart():
    """
    process UCIiris data with GMM, give analyze result.
    """
    data = pd.read_csv("data/motified_data.csv")
    Y = data["5"].values.tolist()
    data = data.drop("5", axis=1)
    XX = data.iloc[:, 1:].values.tolist()
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.25, random_state=0)  # 随机选择25%作为测试集，剩余作为训练集

    pk, means, vars, TPtest = GMMandAnalyze(X_train, X_test, y_test, 4)
    pk, means, vars, TPtrain = GMMandAnalyze(X_train, X_train, y_train, 4)
    print("TPtrain,TPtest,len(X_train),len(X_test)=", TPtrain, TPtest, len(X_train), len(X_test))
    # TPtrain,TPtest,len(X_train),len(X_test)= 80 35 112 38

def preprocessData():
    """
    data-preprocess.
    """
    data = pd.read_csv("data/iris.csv")
    data.to_csv("data/motified_data.csv")


if __name__ == '__main__':
    # 测试说明：下面三条分别展示UCI数据、二维、三维数据的GMM效果
    # processUCIheart()
    # twoDimensionalClusterDisplay()
    thiDimensionalClusterDisplay()
