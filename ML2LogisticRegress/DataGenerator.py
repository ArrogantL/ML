from numpy.random import *


def generateData(num, feature_num):
    XX = []
    Y = []
    feature = []
    for j in range(feature_num):
        feature.append((randint(0, 10),randint(0, 10),randint(0, 10)))
    for i in range(num):
        rint = randint(0, 2)
        X = []
        for j in range(feature_num):
            #Xi与Y有关,其中方差仅仅与i有关，均值与y有关
            if rint==1:
                X.append(normal(feature[j][0],feature[j][2]))
            else:
                X.append(normal(feature[j][1], feature[j][2]))

        XX.append(X)
        Y.append(rint)

    return XX, Y


if __name__ == '__main__':
    XX, Y = generateData(2, 10)
    print(XX)
    print(Y)
