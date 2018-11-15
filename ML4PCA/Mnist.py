import struct

import matplotlib.pyplot as plt
import numpy as np
from numpy import *

from PCA import pca


def readData():
    """
    参考mnist官网以及github实现数据读取，读取结果是一个28*28的list，取值为0-255整数
    :return:
    """
    with open("data/train-images-idx3-ubyte", 'rb') as f1:
        buf = f1.read()
    temp = struct.unpack_from('>784B', buf, struct.calcsize('>IIII'))
    im = np.reshape(temp, (28, 28))
    return im


def mnistProcess():
    """
    分析minist-5,目标图片是数字5的手写
    将不同pcnum的结果保存到data/mnistprocess

    这里将每一行当作一个样本，即28个28维样本。
    可选的其他样本划分方式：
    1. 加入行序号：由于行号规律强，PCA通常不会将其作为主成分，所以加不加影响不大。
    2. 28*28个三维样本，（行号，列号，灰度）：行列号对PCA影响不大，仅剩下一个维度无法PCA

    """
    im = readData()
    for i in range(50):
        # i=17与i=18，人眼就非常难以分辨了。
        lowD, newD, topdvects= pca(im, i)
        PSNR=analyzePSNR(im,newD)
        print(i,PSNR)
        plt.imshow(np.matrix.tolist(newD), cmap='gray')
        plt.savefig("data/mnistprocess/" + str(i) + ".png")
def analyzePSNR(im,im2):
    SUM=np.sum(np.power(im-im2,2))
    MSE = SUM / (im.shape[0]*im.shape[1])
    PSNR = 10 * math.log((255.0 * 255.0 / (MSE)), 10)
    return PSNR


if __name__ == '__main__':
    mnistProcess()
