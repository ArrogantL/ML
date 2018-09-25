import sys

import numpy
from numpy import logspace
from numpy import mat, polyval

from DataGenerator import generateData
from Visualization import visualResultAndSampleAndTarget


def gradientDescent(n, X, T, lr, batch=1, maxItrTimes=10,targetAverageRSS=5*10**-4):
    """
    梯度下降法多项式拟合
    :param n:
    :param X:
    :param T:
    :param lr:
    :param batch:
    :param maxItrTimes: 最大迭代次数，默认为sys.maxsize
    :param targetAverageRSS: 目标RSS均值，到达时停止迭代
    :return: W，次数由低到高
    """
    wSize = n + 1
    count = 0
    W = [5 for i in range(wSize)]
    mins = 1000
    batchX = []
    batchT = []

    for i in range(maxItrTimes):
        for x, t in zip(X, T):
            batchX.append(x)
            batchT.append(t)
            count += 1
            if count % batch == 0:
                gradient = getGradient(batchT, W, batchX)
                W = [w + lr * g / batch for w, g in zip(W, gradient)]
                rss = RSS(T, X, W, range=10, isaverage=True)

                if rss <= targetAverageRSS :
                    return rss, W

                print("%d %f %e" % (count, lr, rss))
                batchX = []
                batchT = []
        # visualPoly(*[W, type], isShow=True)
    return rss, W


def getGradient(T, W, X):
    W.reverse()
    rW = [0 for w in W]
    for t, x in zip(T, X):
        val = polyval(W, x)
        k = t - val
        rW = [rw + k * m for rw, m in zip(rW, logspace(0, len(W) - 1, len(W), base=x))]
    W.reverse()

    return rW


def standgetGradient(T, W, X):
    """
    样例梯度下降，测试代码正确性用，不属于实验内容
    :param T:
    :param W:
    :param X:
    :return:
    """
    XX = mat([[x ** i for i in range(len(W))] for x in X])
    XXT = XX.T
    vectorW = mat(W).T
    vectorT = mat(T).T
    hypothesis = XX * vectorW
    loss = vectorT - hypothesis
    gradient = XXT * loss
    for i in range(len(W)):
        assert abs(getGradient(T, W, X)[i] - standgetGradient.T.tolist()[0][i]) < 0.0001

    return gradient.T.tolist()[0]


def RSS(T, X, W, range=-1, isaverage=False):
    sum = 0
    W.reverse()
    count = 0
    for t, x in zip(T, X):
        sum += numpy.square((t - polyval(W, x)))
        count += 1
        if range > 0 and count == range:
            break
    sum /= 2
    W.reverse()
    if isaverage:
        sum /= count
    return sum


if __name__ == '__main__':
    dataNum = 5
    n = 9
    lr = 1.1
    maxItrTimes = sys.maxsize
    batch = 5

    X, T = generateData(dataNum)

    e, resultW = gradientDescent(n, X, T, lr, batch=batch,
                                 maxItrTimes=maxItrTimes)
    visualResultAndSampleAndTarget(resultW, X, T)
