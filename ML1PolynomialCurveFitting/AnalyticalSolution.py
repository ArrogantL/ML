import math

import matplotlib.pyplot as plt
import numpy
from numpy import mat, polyval

from DataGenerator import generateData
from Visualization import visualPoly


def analyticalSolve(n, X, T, lnLambada=None):
    '''
    :param n:
    :param X:
    :param T:
    :param lnLambada:  lambada的以自然数为底的对数，如果设为1000则表示不设只lambada
    :return:
    '''
    dim = n + 1
    if lnLambada == None:

        lambada=0
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
    W2 = analyticalSolve(9, X, T, lnLambada=-18)
    W3 = analyticalSolve(9, X, T, lnLambada=0)
    visualPoly(W1,W2,W3,"ln1000","ln1","ln0",title="lndiffer",savePath="")
    # plt.plot(X, T, 'r*', linewidth=2)
# plt.show()
