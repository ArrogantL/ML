import math

import numpy
from numpy import mat

from DataGenerator import generateData
from Visualization import visualPoly, visualResultAndSampleAndTarget


def analyticalSolve(n, X, T, lnLambada=None):
    '''
    :param n:多项式次数
    :param X:样本自变量x
    :param T:样本因变量t
    :param lnLambada:  lambada的以自然数为底的对数，如果设为1000则表示不设只lambada
    :return: 解析解拟合的次数由低到高的权重向量
    '''
    dim = n + 1
    if lnLambada == None:

        lambada = 0
    else:
        lambada = math.e ** lnLambada

    XX = mat([[x ** i for i in range(dim)] for x in X])
    vectorT = mat(T).T
    # print(XX,XX.max())

    XXT = XX.T
    return ((lambada * numpy.eye(dim) + XXT * XX).I * XXT * vectorT).T.tolist()[0]


if __name__ == '__main__':
    X, T = generateData(20)
    W1 = analyticalSolve(9, X, T)
    W2 = analyticalSolve(9, X, T, lnLambada=None)
    W3 = analyticalSolve(9, X, T, lnLambada=0)
    visualPoly(W1, W2, W3, "no lambda", "ln1", "ln0", X=X,T=T,title="lndiffer", isShow=True,savePath="None")
    # plt.plot(X, T, 'r*', linewidth=2)
# plt.show()
