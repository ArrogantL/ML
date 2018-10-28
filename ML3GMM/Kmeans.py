from numpy import *

from DataGenerator import generateData
import matplotlib.pyplot as plt

def kmeans(XX,k):
    #随机生成k个聚点,聚点的维数要和X相同,clusterPoint
    KK=[]
    for i in range(k):
        KK.append([i]*len(XX[0]))
    Labels = mark(XX, KK)
    recalClusterPoint(XX, Labels, k)
    while True:
        #对XX打标签，标签0，1,2,，，k-1
        tmp = Labels
        Labels=mark(XX,KK)
        for k in Labels:
            if Labels[k]==tmp[k]:
                return Labels
        #重新计算聚点
        recalClusterPoint(XX,Labels,k)


def mark(XX,KK):
    Labels = {}
    for x in range(len(XX)):
        X=XX[x]
        d,l=min((distanceOfKmeans(X,KK[i]),i) for i in range(len(KK)))
        Labels.setdefault(l,[])
        Labels[l].append(x)
    return Labels

def distanceOfKmeans(X,K):
    #内积
    sum=0
    for i in range(len(X)):
        sum+=abs(X[i]-K[i])**2
    return sum
def recalClusterPoint(XX,Labels,k):
    KK = []
    for i in range(k):
        KK.append([0] * len(XX[0]))
    for k in Labels:
        for x in Labels[k]:
            X=XX[x]
            KK[k]=[KK[k][m]+X[m] for m in range(len(KK[0]))]
        KK[k]=[KK[k][m]/len(Labels[k]) for m in range(len(KK[0]))]
    return KK

if __name__ == '__main__':
    XX, Y = generateData(100, 2)
    Labels=kmeans(XX,2)

    # 计算颜色值
    #color = np.arctan2(y, x)
    # 绘制散点图
    colors=['red','green','black']
    for k in Labels:
        for i in Labels[k]:
            plt.scatter(XX[i][0],XX[i][1], s=75, c=colors[k], alpha=0.08)
    # 设置坐标轴范围


    # 不显示坐标轴的值


    plt.show()
