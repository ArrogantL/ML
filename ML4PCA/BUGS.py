from numpy import *


def bug1():
    # w, v = la.eig(np.diag((1, 2, 3)))
    # WandLam = ed(np.diag((1, 2, 3)))
    D = np.diag((1, 2, 3))

    sum = zeros(3)
    for X in D:
        sum += X
    mean = sum / 3

    # 这里我们可以发现i——mean计算正常，但是赋值的时候自动取整了。
    # 这是numpy矩阵的数值保留，如果把初始参数变成浮点数就能解决这个问题
    for i in range(3):
        i_mean = D[i] - mean

        D[i] = i_mean
