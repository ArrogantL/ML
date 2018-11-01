from numpy import *
from numpy.linalg import det

from DataGenerator import generateData
import matplotlib.pyplot as plt


def gaussianMixtureModel(XX, k):
    # start: random model:P(K) means vars
    D=len(XX[0])
    pk = [1/k] * k
    means = [mat([i]).T for i in range(k)]
    vars = [mat(eye(D))] * k
    tmp=[]
    for X in XX:
        tmp.append(mat(X).T)
    XX=tmp
    pkx=[[0]*k]*len(XX)
    count=1
    while True:
        count+=1
        # E: recal P(K|X)
        tmp=pkx
        rss =0

        pkx = []
        for i in range(len(XX)):
            pkx.append([])
            for j in range(k):
                v=calPKX(pk, means, vars, XX[i], j)
                pkx[i].append(v)
                rss+=abs(v-tmp[i][j])
        if count>30:
            break




        # M: max likehood with respect model:P(K) means vars
        tmpmeans=[]
        tmpvars=[]
        tmppk=[]
        for j in range(k):
            Nk = 0
            mean = mat(zeros(D)).T
            var = mat(zeros((D,D)))
            for i in range(len(XX)):
                v=pkx[i][j][0,0]
                mean+=v*XX[i]

                Nk += v
            for i in range(len(XX)):
                delt = XX[i] - mean/Nk
                var += pkx[i][j][0,0] * delt * delt.T

            tmpmeans.append(mean/Nk)
            tmpvars.append(var/Nk)
            tmppk.append(Nk/len(XX))
        means=tmpmeans
        vars=tmpvars
        pk=tmppk


    return pk,means,vars


def calPKX(pk, means, vars, X, k):
    assert k < len(pk)
    px = 0

    for i in range(len(pk)):

        mean = means[i]
        var = vars[i]
        vv=det(var)
        c=exp(-1 / 2 * (X - mean).T * var.I * (X - mean))
        pi = pk[i] * power((2 * math.pi), -len(X) / 2) * power(vv, -1 / 2) * c
        px += pi
        if i == k:
            pxk = pi
    v= pxk / px
    return v



if __name__ == '__main__':
    XX, Y = generateData(100, 2)
    pk, means, vars = gaussianMixtureModel(XX, 2)

    # 计算颜色值
    # color = np.arctan2(y, x)
    # 绘制散点图
    colors = ['red', 'green', 'black']
    labels={}


    for X in XX:
        p,k=max((calPKX(pk, means, vars, X, k)[0,0], k) for k in range(2))
        labels.setdefault(k,[])
        labels[k].append(X)
    for k in labels:
        for X in labels[k]:
            plt.scatter(X[0], X[1], s=75, c=colors[k], alpha=0.08)
    # 设置坐标轴范围

    # 不显示坐标轴的值

    plt.show()