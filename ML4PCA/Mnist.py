import struct

import matplotlib.pyplot as plt
import numpy as np
from numpy import *

from PCA import pca


def readData():
    with open("data/train-images-idx3-ubyte", 'rb') as f1:
        buf = f1.read()
    temp = struct.unpack_from('>784B', buf, struct.calcsize('>IIII'))
    im = np.reshape(temp, (28, 28))
    return im


def mnistProcess():
    """
    分析minist-5,目标图片是数字5的手写
    将不同pcnum的结果保存到data/mnistprocess
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
