from numpy.random import *


def generateData(num, feature_num):
    """

    :param num:数据量
    :param feature_num:特征数
    :param u2u: 均值是否相互独立
    :param s2k: 方差是否与yk有关
    :param type: 数据的分布，"normal"、"binomial"、"beta"
    :return:
    """
    XX = []
    Y = []
    for i in range(num):
        if randint(0, 2) == 0:
            XX.append(multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]]))
        else:
            XX.append(multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]]))
    return XX, Y


if __name__ == '__main__':
    XX, Y = generateData(10, 100, True, False, "binomial")
    print(XX)
