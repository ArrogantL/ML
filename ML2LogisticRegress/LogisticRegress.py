from matplotlib import *
from numpy import *
from sklearn.model_selection import train_test_split

from DataGenerator import *


def quasiNewtonLogisticRegress(XX, y):
    XX = appendOne(XX)

    assert len(XX) == len(y)
    # 初始化W
    W = ones((1, len(XX[0]))).T
    tmp = sys.maxsize
    flag = 1
    while True:
        gradient = gradientOfLogisticRegres(W, XX, y)
        # hessianI = hessianOfLogisticRegress(W, XX).I
        if flag == 1:
            hessianI = eye(len(W))
            flag = 0
        else:
            hessianI = quasiHessianI(dg, delta, hessianI)
        delta = -hessianI * gradient
        loss = -gradient.T * hessianI * gradient

        W += delta
        dg = gradientOfLogisticRegres(W, XX, y) - gradient

        print(len(W), W.T)
        if abs(loss[0, 0] / 2) < 0.1:
            break

    return W, loss


def quasiHessianI(dg, dW, B):
    return B + (dg - B * dW) * dW.T / (dW.T * dW)

def gradientDescent(XX, Y,lr,batch=1,maxRound=100,lambada=0,punishment=0):
    """
    梯度下降法拟合
    """

    XX = appendOne(XX)
    assert len(XX) == len(Y)
    # 初始化W
    W = zeros((1, len(XX[0]))).T
    count = 0

    batchXX = []
    batchY = []
    bestW=W
    bestAccuracy=0
    for i in range(100):
        for X, y in zip(XX, Y):
            batchXX.append(X)
            batchY.append(y)
            count += 1
            if count % batch == 0:
                gradient = gradientOfLogisticRegres(W, batchXX, batchY, punishment=punishment)
                W+=gradient
                f = accuracy(W, XX, Y, isAppendOne=False)
                if f>bestAccuracy:
                    bestAccuracy=f
                    bestW=W


                print("%d %f %e" % (count, f, abs(gradient.T * gradient)))

                if abs(gradient.T*gradient) <= 0.00000001:
                    break
                batchXX = []
                batchY = []

    return bestAccuracy,bestW

def newtonLogisticRegress(XX, y, punishment=0):
    XX = appendOne(XX)
    assert len(XX) == len(y)
    # 初始化W
    W = zeros((1, len(XX[0]))).T
    tmp = sys.maxsize
    i = 0
    while True:
        gradient = gradientOfLogisticRegres(W, XX, y, punishment=punishment)
        hessianI = hessianOfLogisticRegress(W, XX, punishment=punishment).I
        delta = -hessianI * gradient
        loss = -gradient.T * hessianI * gradient
        # print("Round Loss %d : %f"%(i,loss))
        if abs(loss[0, 0] / 2) < 10 ** -10:
            break
        W += delta
        i += 1
    return W, loss


def hessianOfLogisticRegress(W, XX, punishment):
    assert len(W) == len(XX[0])
    len_W = len(W)
    hessian = mat(zeros((len_W, len_W)))

    for i in range(len_W):
        for j in range(len_W):
            sum = 0
            for l in range(len(XX)):
                e = exp(dot(mat(XX[l]), W))
                sum += -XX[l][i] * XX[l][j] * e / (1 + e) ** 2
            if i == j:
                sum -= punishment
            hessian[i, j] = sum / len(XX)
    return hessian


def gradientOfLogisticRegres(W, XX, y, punishment):
    assert len(W) == len(XX[0])
    assert len(y) == len(XX)
    len_W = len(W)
    gradient = mat(zeros((1, len_W)))
    mat_W = mat(W)
    for i in range(len_W):
        sum = 0
        for l in range(len(XX)):
            e = exp(dot(mat(XX[l]), mat_W))
            sum += y[l] * XX[l][i] - XX[l][i] * e / (1 + e)
        sum -= punishment * W[i]
        gradient[0, i] = sum / len(XX)
    return gradient.T


def accuracy(W, XX, Y,type=0,isAppendOne=True):
    tmpXX=XX
    if isAppendOne:
        tmpXX = appendOne(XX)

    TP = 0
    TN=0
    FP=0
    FN=0
    for i in range(len(Y)):


        if dot(tmpXX[i], W) > 0 :
            if  Y[i] == 1 :
                TP += 1
            else:
                FP+=1
        if dot(tmpXX[i], W) < 0 :
            if Y[i] == 0:
                TN+=1
            else:
                FN+=1
    assert TP+FP+TN+FN==len(Y)
    if type==0:
        return (TP+TN)/len(Y)
    elif type==1:
        return TP/(TP+FP)
    elif type==2:
        return TP/(TP+FN)
    elif type==3:
        return ((TP+TN)/len(Y),TP/(TP+FP),TP/(TP+FN))
    else:
        return None





def appendOne(XX):
    tmpXX = []
    for X in XX:
        tmpX = []
        tmpX.append(1)
        tmpX += X
        tmpXX.append(tmpX)
    XX = tmpXX
    return XX



def test():
    #正常情况
    #XX, Y = generateData(50, 5, u2u=False, s2k=True,type="normal")
    #S与y有关
    #XX, Y = generateData(50, 5, u2u=False, s2k=True)
    #X特征之间不再相互独立
    #XX, Y = generateData(50, 5, u2u=True, s2k=False)
    #不是正太分布
    #XX, Y = generateData(50, 5, u2u=True, s2k=False,type="beta")
    # XX, Y = generateData(50, 5, u2u=True, s2k=False,type="binomial")
    #XXTrain, XXTest, YTrain, YTest = train_test_split(XX, Y, test_size=0.25, random_state=0)  # 随机选择25%作为测试集，剩余作为训练集

    grid = [0, 1, 5, 10, 20, 50, 100, 200, 500,5000,50000]
    print("Round\taverage_accuracy_in_Train\taverage_accuracy_in_Test")
    acTrain=0
    acTest=0
    for i in range(1000):
        i+=1
        XX, Y = generateData(50, 3, u2u=False, s2k=False, type="beta")
        XXTrain, XXTest, YTrain, YTest = train_test_split(XX, Y, test_size=0.25, random_state=0)  # 随机选择25%作为测试集，剩余作为训练集
        try:
            W, loss = newtonLogisticRegress(XXTrain, YTrain, punishment=0)
        except:
            continue
        f = accuracy(W, XXTrain, YTrain)
        values = accuracy(W, XXTest, YTest)
        acTrain += f
        acTest+=values
        print("%d\t%f\t%f\t"%(i, acTrain/i, acTest/i))


if __name__ == '__main__':
    test()


