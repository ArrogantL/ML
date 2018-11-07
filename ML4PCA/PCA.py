from numpy import *
from numpy import linalg
import matplotlib.pyplot as plt

from DataGenerator import generateData


def pca(datamatrix, pcnum):
    """
    pca分析，输出低维坐标、由低维坐标还原的高维坐标、主向量
    :param datamatrix:
    :param pcnum: number of principal component
    :return: lowD, newD,topdvects 低维坐标、由低维坐标还原的高维坐标、主向量
    """
    datamatrix = mat(datamatrix)
    # 中心化
    meanmatrix = datamatrix.mean(axis=0)
    remeaneddata = datamatrix - meanmatrix
    # 计算cov矩阵
    datacov = remeaneddata.T * remeaneddata
    # 特征分解,得到特征向量矩阵,eig_vect是按列存储的
    vals, vects = linalg.eig(datacov)
    indexs = argsort(vals)
    indexs = indexs[len(indexs) - pcnum:len(indexs)]
    topdvects = vects[:, indexs]
    # 数据转到新空间
    lowD = remeaneddata * topdvects
    newD = lowD * topdvects.T + meanmatrix
    return lowD, newD,topdvects


if __name__ == '__main__':
    D, X, T, R = generateData(100, mean=10, var=0.02)
    lowD, newD,topdvects = pca(mat(D), 2)
    A = []
    B = []
    for X in newD:
        A.append(X[0, 0])
        B.append(X[0, 1])
    plt.plot(A, B, 'r*', linewidth=2)
    plt.show()
