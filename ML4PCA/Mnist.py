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
    for i in range(30):
        print(i)
        # i=17与i=18，人眼就非常难以分辨了。
        lowD, newD, topdvects= pca(im, i)
        plt.imshow(np.matrix.tolist(newD), cmap='gray')
        plt.savefig("data/mnistprocess/" + str(i) + ".png")


if __name__ == '__main__':
    mnistProcess()
