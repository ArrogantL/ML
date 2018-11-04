from numpy.random import *


def generateTwoDimensionalData\
                (num,featurenum=4, means=[[0,0], [0,2],[2,0],[2,2]],
                               vars=[[0.1, 0], [0, 0.1]]):
    """

    :param vars:
    :param means:
    :param num:数据量
    :return:
    """
    XX = []
    Y = []
    for i in range(num):
        if randint(0, featurenum) == 0:
            XX.append(multivariate_normal(means[0], vars))
        elif randint(0, featurenum) == 1:
            XX.append(multivariate_normal(means[1], vars))
        elif randint(0, featurenum) == 2:
            XX.append(multivariate_normal(means[2], vars))
        elif randint(0, featurenum) == 3:
            XX.append(multivariate_normal(means[3], vars))


    return XX, Y
