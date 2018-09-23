from math import sin, pi

import matplotlib.pyplot as plt
import numpy as np


def generateData(num):
    X=np.random.rand(num)

    S=np.random.randn(num)
    tmp=[sin( 2*pi * m) for m in X]
    T=[]
    for i in range(num):
        T.append(tmp[i]+S[i]/100)

    assert len(X)==len(T)
    return X,T
if __name__ == '__main__':
    X, T = generateData(500)

    plt.plot(X, T, 'r*', linewidth=2)
    plt.show()