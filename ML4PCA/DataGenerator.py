from math import sin, pi

import matplotlib.pyplot as plt
import numpy as np

from numpy.random.mtrand import multivariate_normal


def generateData(num,mean=100,var=0):
    X=np.random.rand(num)

    S=np.random.randn(num)
    tmp=[sin( 2*pi * m) for m in X]
    T=[]
    for i in range(num):
        T.append(tmp[i]+S[i]/100)

    assert len(X)==len(T)
    R=[]
    for i in range(len(X)):
        R.append(multivariate_normal([mean], [[var]]))
        # R.append(0)
    D=[]
    for x,t,r in zip(X,T,R):
        D.append((x,t,r))
    return D,X,T,R
if __name__ == '__main__':
    D, X, T, R = generateData(500)

    plt.plot(X, T, 'r*', linewidth=2)
    plt.show()