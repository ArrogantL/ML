from numpy.random import *


def generateData(num, feature_num,u2k=True,s2k=False):

    XX = []
    Y = []
    feature = []
    feature.append((randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10)))

    for j in range(feature_num-1):
        #feature.append((randint(0, 10), randint(0, 10), randint(0, 10), randint(0, 10)))
        feature.append((feature[j][0]+randint(0, 10),feature[j][1]+randint(0, 10),feature[j][2]+randint(0, 10),feature[j][3]+randint(0, 10)))
    for i in range(num):
        rint = randint(0, 2)
        X = []

        for j in range(feature_num):

            if rint==1:
                X.append(normal(feature[j][0],feature[j][2]))
            else:
                X.append(normal(feature[j][0+u2k], feature[j][2+s2k]))

        XX.append(X)
        Y.append(rint)

    return XX, Y


if __name__ == '__main__':
    X=[0,12]
    tmp=[3]
    tmp+=X
    print(tmp)
