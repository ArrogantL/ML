from numpy import *
from numpy.linalg import det
from sklearn.model_selection import train_test_split
from matplotlib import *
from numpy import *
import pandas as pd
from sklearn.model_selection import train_test_split

from DataGenerator import generateTwoDimensionalData
import matplotlib.pyplot as plt
"""
该版本以及已经弃用，是未使用矩阵操作优化的版本。
"""

def gaussianMixtureModel(XX, k):
    """
    使用kmeans方法讲一组向量XX聚为k类。
    :param XX:样本
    :param k:聚类数量
    :return: 混合概率，各聚类高斯均值，各聚类高斯协方差阵=pk, means, vars
    """
    #  convert list X to column vector X
    # 该版本已弃用
    assert False
    tmp = []

    for X in XX:
        tmp.append(mat(X).T)
    XX = tmp
    D = len(XX[0])
    # start: random model:P(K) means vars,Attention：mean[i]!=mean[j] for any i!=j,
    # otherwise they will get the same parameters during the iteration.
    means = [mat([i]*D).T for i in range(k)]
    vars = [mat(eye(D))] * k
    pk = [1 / k] * k
    # P(K|X) rowX columnK
    pkx = [[0] * k] * len(XX)

    # start EM process
    count=0
    while True:
        count+=1
        # E: estimate P(K|X)
        tmp = pkx
        increment = 0
        pkx = []
        for i in range(len(XX)):
            pkx.append([])
            for j in range(k):
                v = calPKX(pk, means, vars, XX[i], j)
                pkx[i].append(v)
                increment += abs(v - tmp[i][j])
        # increment is less than the threshold,return result
        if increment<10**-5:
            break

        # M: max likehood with respect model:P(K) means vars
        tmpmeans = []
        tmpvars = []
        tmppk = []
        for j in range(k):
            Nk = 0
            mean = mat(zeros(D)).T
            var = mat(zeros((D, D)))
            for i in range(len(XX)):
                v = pkx[i][j][0, 0]
                mean += v * XX[i]
                Nk += v
            for i in range(len(XX)):
                delt = XX[i] - mean / Nk
                var += pkx[i][j][0, 0] * delt * delt.T
            tmpmeans.append(mean / Nk)
            tmpvars.append(var / Nk)
            tmppk.append(Nk / len(XX))
        means = tmpmeans
        vars = tmpvars
        pk = tmppk

    return pk, means, vars


def calPKX(pk, means, vars, X, k):
    """
    calculating P(K=k|X)
    :param pk:blend weight
    :param means:gaussians' means
    :param vars:gaussians' variances
    :param X:target sample
    :param k:target class
    :return:[[P(k|X)]]
    """
    # 该版本已弃用
    assert False
    assert k < len(pk)
    pkx = 0
    lgpxk = 0
    lgpxilist = []
    for i in range(len(pk)):
        mean = means[i]
        var = vars[i]
        try:
            # use log to ease overflow
            assert linalg.det(var) >= 0
            lgpxi = log(pk[i] * power((2 * math.pi), -len(X) / 2) * power(det(var), -1 / 2)) + (-1 / 2 * (X - mean).T * var.I * (X - mean))
            lgpxilist.append(lgpxi)
        except numpy.linalg.linalg.LinAlgError:
            # matrix without inverse matrix
            continue
        if i == k :
            lgpxk = lgpxi

    for lgpxi in lgpxilist:
        try:
            pkx += exp(lgpxi - lgpxk)
        except:
            continue

    pkx = 1 / pkx
    if pkx[0, 0] == nan:
        # 仍然可能存在溢出，这样的样本选择抛弃
        pkx[0, 0] = 0
    return pkx


def twoDimensionalClusterDisplay():
    # 该版本已弃用
    assert False
    clusternum=4
    # 二维划分展示
    XX, Y = generateTwoDimensionalData(num=200)
    # 使用一半样本来训练
    pk, means, vars = gaussianMixtureModel(XX[:len(XX) // 2], clusternum)
    # 绘制散点图
    colors = ['blue','red', 'black', 'green']
    labels = {}
    for X in XX[len(XX) // 2:]:
        X=mat(X).T
        p, k = max((calPKX(pk, means, vars, X, k)[0, 0], k) for k in range(clusternum))
        labels.setdefault(k, [])
        labels[k].append(X)
    for k in labels:
        for X in labels[k]:
            plt.scatter(X[0,0], X[1,0], s=75, c=colors[k], alpha=0.5)
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
    # 该版本已弃用
    assert False
    pk, means, vars = gaussianMixtureModel(X_train, k)
    labels = {}
    for X, y in zip(X_test, y_test):
        X = mat(X).T
        p, m = max((calPKX(pk, means, vars, X, m)[0, 0], m) for m in range(k))
        labels.setdefault(m, [])
        labels[m].append((X, y))
    TP = 0
    for list in labels.values():
        nums = {}
        for X, y in list:
            nums.setdefault(y, 0)
            nums[y] += 1
        num, y = max((nums[y], y) for y in nums)
        TP += num

    return pk, means, vars, TP


def processUCIheart():
    """
    process UCIiris data with GMM, give analyze result.
    """
    # 该版本已弃用
    assert False
    data = pd.read_csv("data/motified_data.csv")
    Y = data["5"].values.tolist()
    data = data.drop("5", axis=1)
    XX = data.iloc[:, 1:].values.tolist()
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.25, random_state=0)  # 随机选择25%作为测试集，剩余作为训练集

    pk, means, vars, TPtest = GMMandAnalyze(X_train, X_test, y_test, 3)
    pk, means, vars, TPtrain = GMMandAnalyze(X_train, X_train, y_train, 3)
    print("TPtrain,TPtest,len(X_train),len(X_test)=",TPtrain,TPtest,len(X_train),len(X_test))
    # TPtrain,TPtest,len(X_train),len(X_test)= 98 35 112 38

def preprocessData():
    """
    data-preprocess.
    """
    data = pd.read_csv("data/iris.csv")
    data.to_csv("data/motified_data.csv")

if __name__ == '__main__':
    # 该版本已弃用
    assert False
    # preprocessData()
    processUCIheart()
    # twoDimensionalClusterDisplay()
