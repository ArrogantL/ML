import numpy as np
from numpy import *

from DataGenerator import *
from matplotlib import *
import matplotlib.pyplot as plt


def quasiNewtonLogisticRegress(XX, y, t):
    assert len(XX) == len(y)
    # 初始化W
    W = ones((1, len(XX[0]))).T
    tmp = sys.maxsize
    flag = 1
    while True:
        gradient = gradientOfLogisticRegres(W, XX, y)
        # hessianI = hessianOfLogisticRegress(W, XX).I
        if flag == 1:
            hessianI = eye(len(W))
            flag = 0
        else:
            hessianI = quasiHessianI(dg, delta, hessianI)
        delta = -hessianI * gradient
        loss = -gradient.T * hessianI * gradient

        W += t * delta
        dg = gradientOfLogisticRegres(W, XX, y) - gradient

        print(loss)
        if abs(loss[0, 0] / 2) < 0.1:
            break

    return W, loss


def quasiHessianI(dg, dW, B):
    return B + (dg - B * dW) * dW.T / (dW.T * dW)


def newtonLogisticRegress(XX, y, t):
    """
    XX中的X要求第一个维度是1,为方便运算。每行一个X
    :param XX:
    :param y:
    :return:
    """
    assert len(XX) == len(y)
    # 初始化W
    W = ones((1, len(XX[0]))).T
    tmp = sys.maxsize
    while True:
        gradient = gradientOfLogisticRegres(W, XX, y)
        hessianI = hessianOfLogisticRegress(W, XX).I
        delta = -hessianI * gradient

        loss = -gradient.T * hessianI * gradient

        print(loss)
        if abs(loss[0, 0] / 2) < 0.1:
            break
        W += t * delta
    return W, loss


def hessianOfLogisticRegress(W, XX):
    assert len(W) == len(XX[0])
    len_W = len(W)
    hessian = mat(zeros((len_W, len_W)))
    mat_W = mat(W)
    for i in range(len_W):
        for j in range(len_W):
            sum = 0
            for l in range(len(XX)):
                e = exp(dot(mat(XX[l]), mat_W))
                sum -= XX[l][i] * XX[l][j] * e / (1 + e) ** 2
            hessian[i, j] = sum
    return hessian


def gradientOfLogisticRegres(W, XX, y):
    assert len(W) == len(XX[0])
    assert len(y) == len(XX)
    len_W = len(W)
    gradient = mat(zeros((1, len_W)))
    mat_W = mat(W)
    for i in range(len_W):
        sum = 0
        for l in range(len(XX)):
            e = exp(dot(mat(XX[l]), mat_W))
            sum += y[l] * XX[l][i] - XX[l][i] * e / (1 + e)
        gradient[0, i] = sum
    return gradient.T


def analyze(W, XX, Y):
    sum = 0
    for i in range(len(Y)):
        if (dot(XX[i], W) > 0 and Y[i] == 1) or (dot(XX[i], W) < 0 and Y[i] == 0):
            sum += 1
    return sum / len(Y)


if __name__ == '__main__':
    # W = [1, 5, 4]
    # XX = [[1, 2, 5], [1, 9, 3]]
    # hessian = hessianOfLogisticRegress(W, X)
    # W2,loss=newtonLogisticRegress(XX,[0,1],0.00001)

    XX, Y = generateData(100, 5)
    W, loss = quasiNewtonLogisticRegress(XX, Y, 0.0000001)
    """
    for i in range(len(Y)):

        if Y[i] == 1:

            plt.scatter(XX[i][0], XX[i][1], s=75, c='G', alpha=.5)
        else:
            plt.scatter(XX[i][0], XX[i][1], s=75, c='R', alpha=.5)
    X = numpy.arange(-2, 10, 0.01)
    plt.scatter(X,[W[0]*x+W[1] for x in X] )
    plt.show()"""
    print(analyze(W, XX, Y))
