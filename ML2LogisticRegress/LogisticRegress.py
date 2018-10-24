import numpy as np
from numpy import *

from DataGenerator import *
from matplotlib import *
import matplotlib.pyplot as plt


def quasiNewtonLogisticRegress(XX, y):
    XX = appendOne(XX)

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

        W += delta
        dg = gradientOfLogisticRegres(W, XX, y) - gradient

        print(len(W), W.T)
        if abs(loss[0, 0] / 2) < 0.1:
            break

    return W, loss


def quasiHessianI(dg, dW, B):
    return B + (dg - B * dW) * dW.T / (dW.T * dW)


def newtonLogisticRegress(XX, y, punishment=0):
    """
    XX中的X要求第一个维度是1,为方便运算。每行一个X
    :param XX:
    :param y:
    :return:
    """
    XX=appendOne(XX)
    assert len(XX) == len(y)
    # 初始化W
    W = zeros((1, len(XX[0]))).T
    tmp = sys.maxsize
    i=0
    while True:
        gradient = gradientOfLogisticRegres(W, XX, y, punishment=punishment)
        hessianI = hessianOfLogisticRegress(W, XX, punishment=punishment).I
        delta = -hessianI * gradient

        loss = -gradient.T * hessianI * gradient

        print("Round Loss %d : %f"%(i,loss))
        if abs(loss[0, 0] / 2) < 10**-10:
            break
        W += delta
        i+=1
    return W, loss


def hessianOfLogisticRegress(W, XX, punishment):
    assert len(W) == len(XX[0])
    len_W = len(W)
    hessian = mat(zeros((len_W, len_W)))

    for i in range(len_W):
        for j in range(len_W):
            sum = 0
            for l in range(len(XX)):
                e = exp(dot(mat(XX[l]), W))
                sum += -XX[l][i] * XX[l][j] * e / (1 + e) ** 2
            if i == j:
                sum -= punishment
            hessian[i, j] = sum / len(XX)
    return hessian


def gradientOfLogisticRegres(W, XX, y, punishment):
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
        sum -= punishment * W[i]
        gradient[0, i] = sum / len(XX)
    return gradient.T


def accuracy(W, XX, Y):
    tmpXX=appendOne(XX)
    sum = 0
    for i in range(len(Y)):
        if (dot(tmpXX[i], W) > 0 and Y[i] == 1) or (dot(tmpXX[i], W) < 0 and Y[i] == 0):
            sum += 1
    return sum / len(Y)
def appendOne(XX):
    tmpXX = []
    for X in XX:
        tmpX = []
        tmpX.append(1)
        tmpX += X
        tmpXX.append(tmpX)
    XX = tmpXX
    return XX
def test():
    # W = [1, 5, 4]
    # XX = [[1, 2, 5], [1, 9, 3]]
    # hessian = hessianOfLogisticRegress(W, X)
    # W2,loss=newtonLogisticRegress(XX,[0,1],0.00001)

    XX, Y = generateData(50, 5, u2k=True, s2k=True)

    XXTrain = XX[0:len(XX) // 4]
    YTrain = Y[0:len(Y) // 4]
    XXTest = XX[len(XX) // 4:len(XX) // 2]
    YTest = Y[len(Y) // 4:len(Y) // 2]
    XXF = XX[len(XX) // 2:len(XX)]
    YF = Y[len(Y) // 2:len(Y)]

    grid = [0, 1, 5, 10, 20, 50, 100, 200, 500]
    ac = 0
    af = 0
    for punishment in grid:

        W, loss = newtonLogisticRegress(XXTrain, YTrain, punishment=punishment)

        tmp = accuracy(W, XXTest, YTest)

        ac = tmp
        acW = W
        acP = punishment


        af = tmp
        afW = W
        afP = punishment


        print(acP, ac, accuracy(acW, XXF, YF), end="##")
        print(afP, af, accuracy(afW, XXF, YF))


    """
    for i in range(len(Y)):

        if Y[i] == 1:

            plt.scatter(XX[i][1], XX[i][2], s=75, c='G', alpha=.5)
        else:
            plt.scatter(XX[i][1], XX[i][2], s=75, c='R', alpha=.5)
    X = numpy.arange(-20, 20, 0.01)
    plt.scatter(X,[-W[0]/W[2]-W[1]*x/W[2] for x in X] ,0.1)
    plt.show()
    """

if __name__ == '__main__':
    test()
