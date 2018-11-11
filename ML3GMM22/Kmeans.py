from numpy import *
import numpy as np
from DataGenerator import generateTwoDimensionalData
import matplotlib.pyplot as plt

def kmeans(XX,clusternum):
    """
    使用kmeans方法讲一组向量XX聚为k类。
    :param XX:样本
    :param k:聚类数量
    :return:labels={类序号:[属于该类的样本],0:[X1,X2,,,Xn]}
    """
    #随机生成k个聚点,聚点的维数要和X相同,clusterPoint。注意初始化k个点不能重叠
    KK=[]
    for i in range(clusternum):
        KK.append([i]*len(XX[0]))
    labels = mark(XX, KK)
    KK=recalClusterPoint(XX, labels, clusternum)
    count=0
    while True:
        #对XX打标签，标签0，1,2,，，k-1
        tmp = labels
        labels=mark(XX,KK)
        flag=True
        for l in labels:
            if labels[l]!=tmp[l]:
                flag=False
        count+=1
        if flag and count>10:
            return labels,np.array(KK)
        #重新计算聚点
        KK =recalClusterPoint(XX,labels,clusternum)


def mark(XX,KK):
    Labels = {}
    for k in range(len(KK)):
        Labels.setdefault(k,[])
    for x in range(len(XX)):
        X=XX[x]
        d,l=min((distanceOfKmeans(X,KK[i]),i) for i in range(len(KK)))
        Labels[l].append(x)
    return Labels

def distanceOfKmeans(X,K):
    sum=0
    for i in range(len(X)):
        sum+=power(X[i]-K[i],2)
    return power(sum,0.5)
def recalClusterPoint(XX,Labels,clusternum):
    KK = []
    for i in range(clusternum):
        KK.append([0] * len(XX[0]))
    for k in Labels:
        for x in Labels[k]:
            X=XX[x]
            KK[k]=[KK[k][m]+X[m] for m in range(len(KK[0]))]
        if len(Labels[k])!=0:
            KK[k]=[KK[k][m]/len(Labels[k]) for m in range(len(KK[0]))]
    return KK

if __name__ == '__main__':
    clusternum=4
    XX, Y = generateTwoDimensionalData(200)
    labels,means=kmeans(XX, clusternum)
    # 计算颜色值
    #color = np.arctan2(y, x)
    # 绘制散点图
    colors=['red','yellow','green','black']
    for k in labels:
        for i in labels[k]:
            plt.scatter(XX[i][0],XX[i][1], s=75, c=colors[k], alpha=0.5)
    plt.show()
