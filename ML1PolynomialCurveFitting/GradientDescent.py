import sys

import matplotlib.pyplot as plt
import numpy
from numpy import polyval, logspace, mat, math

from DataGenerator import generateData
from Visualization import visualPoly


def gradientDescent(n, X, T, lr, batch=1, targetAverageRSS=0.001, maxItrTimes=10):
    wSize = n + 1
    count = 0
    W = [5 for i in range(wSize)]
    mins = 1000
    batchX = []
    batchT = []
    lastrss=sys.maxsize
    for i in range(maxItrTimes):
        for x, t in zip(X, T):
            batchX.append(x)
            batchT.append(t)
            count += 1
            if count % batch == 0:
                gradient = getGradient(batchT, W, batchX)
                W = [w + lr * g / batch for w, g in zip(W, gradient)]
                rss = lsm(T, X, W, range=10, isaverage=True)
                if rss>lastrss:
                    lr*=0.8
                elif rss==lastrss:
                    lr=lr+0.1
                lastrss=rss
                if rss <targetAverageRSS:
                    return rss,W,rss<0.001
                print("%d %.5f %f"%(count,lr,rss))
                batchX = []
                batchT = []
        # visualPoly(*[W, type], isShow=True)
    return rss, W, rss < 0.01


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
def lsm(T, X, W, range=-1, isaverage=False):
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
    dataNum=100
    n = 5
    lr=10
    maxItrTimes=100
    batch=1

    X, T = generateData(dataNum)

    e, resultW, b = gradientDescent(n, X, T, lr, batch=batch, maxItrTimes=maxItrTimes)
    visualPoly(*[resultW, "test"], title="%s poly%d datanum%d lr%.3f maxIrt%d batch%d"%("gradientDescent",n,dataNum,lr,maxItrTimes,batch),savePath="DataGif/GradientDescent/",isShow=True)
    # plt.plot(X, T, 'r*', linewidth=2)
    # plt.show()
