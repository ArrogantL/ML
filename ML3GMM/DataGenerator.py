from numpy.random import *


def generateData(num, feature_num, u2u=False, s2k=False, type="normal"):
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
    feature = []

    for j in range(feature_num):
        feature.append((randint(10), randint(10), randint(10), randint(10)))

    for i in range(num):
        rint = randint(0, 2)
        X = []
        for j in range(feature_num):
            if j == 0:
                tmp = 0
            else:
                tmp = u2u
            if type == "normal":
                if rint == 1:
                    X.append(normal(feature[j][0] + tmp * feature[j - 1][0], feature[j][2] + tmp * feature[j - 1][2]))
                else:
                    X.append(normal(feature[j][1] + tmp * feature[j - 1][1],
                                    feature[j][2 + s2k] + tmp * feature[j - 1][2 + s2k]))
            elif type == "binomial":
                if rint == 1:
                    X.append(binomial(feature[j][0],
                                      feature[j][2]/10  ))
                else:
                    X.append(binomial(feature[j][1] ,
                                      feature[j][2 ] /10))
            elif type == "beta":
                if rint == 1:
                    X.append(beta(abs(feature[j][0])+1,
                                  abs(feature[j][2] ) +1))
                else:
                    X.append(beta(abs(feature[j][1]+1 ),
                                      abs(feature[j][2])+1 ))
            else:
                return None

            XX.append(X)
            Y.append(rint)

    return XX, Y

if __name__ == '__main__':
    XX, Y = generateData(10, 100, True, False, "binomial")
    print(XX)
