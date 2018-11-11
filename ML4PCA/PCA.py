from numpy import *
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DataGenerator import generateData,generateThiDimensionalData


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
    # 列向量
    topdvects = vects[:, indexs]
    # 数据转到新空间
    lowD = remeaneddata * topdvects
    newD = lowD * topdvects.T + meanmatrix
    return lowD.A, newD.A,topdvects.A


if __name__ == '__main__':
    # D  = generateData(num=200)
    D = generateThiDimensionalData(num=200)
    lowD, newD,topdvects = pca(mat(D), 2)
    A = newD[:,0]
    B = newD[:,1]
    C = newD[:, 2]
    # plt.plot(A, B, 'r*', linewidth=2)
    # plt.show()
    colors = ['blue', 'red', 'black', 'green']
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(newD[:, 0], newD[:, 1], newD[:, 2], s=75, c="blue",
               alpha=0.5)  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
