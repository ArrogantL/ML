import math

import matplotlib.pyplot as plt
import numpy
from numpy import mat, polyval, zeros

from AnalyticalSolution import analyticalSolve
from DataGenerator import generateData
from GradientDescent import lsm


def conjugateGradient(n,X, T, lnLambada=None,limit=0.00000001, MaxIterationNum=100):
    """
    :param n 多项式次数，注意多项式次数是len(W)-1
    :param X: 训练数据
    :param T: 训练数据
    :param limit:  当r=B-AW的内积rT*r小于limit时停止迭代
    :param lnLambada 惩罚参数的以e为底的对数，控制岭回归惩罚力度
    :param MaxIterationNum: 最大迭代次数

    :return: 参数，迭代次数
    """
    dim = n+1
    if lnLambada == None:

        lambada=0
    else:
        lambada = math.e ** lnLambada

    XX = mat([[x ** i for i in range(dim)] for x in X])
    vectorT = mat(T).T
    # A = XX.T * XX
    A = XX.T * XX + lambada * numpy.eye(dim)  # 带惩罚项
    B = XX.T * vectorT
    W = mat(zeros((dim, 1)))
    r = B - A * W
    p = r.copy()
    num = 0
    while num < MaxIterationNum:
        num += 1
        alpha = (r.T * r / (p.T * A * p))[0, 0]
        W += alpha * p
        lastr = r.copy()
        r -= alpha * A * p
        if (r.T * r)[0, 0] < limit:
            break
        beta = (r.T * r / (lastr.T * lastr))[0, 0]
        p = r + beta * p
    return W.T.tolist()[0], num


if __name__ == '__main__':
    X, T = generateData(4)
    W, num = conjugateGradient(9,X, T,lnLambada=-10)
    print(num)
    x = numpy.arange(0, 1, 0.001)
    s=list(W)
    s.reverse()
    y = polyval(s, x)
    plt.plot(x, y, 'r-', linewidth=2)
    plt.show()
    # plt.plot(X, T, 'r*', linewidth=2)
# plt.show()
