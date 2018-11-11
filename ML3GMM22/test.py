import numpy as np
from numpy import tile
from numpy.linalg import det

from DataGenerator import generateTwoDimensionalData
import matplotlib.pyplot as plt



if __name__ == '__main__':
    logpxi_array=np.arange(12.0).reshape(3,4)
    print(logpxi_array)
    k=2
    print(logpxi_array - np.tile(logpxi_array[:, k], (4, 1)).T)
    print(np.exp(logpxi_array - np.tile(logpxi_array[:, k], (4, 1)).T))
    logpxi_array[:, k] = np.sum(np.exp(logpxi_array - np.tile(logpxi_array[:, k], (4, 1)).T), axis=1)
    print(logpxi_array)
    print(np.argmax(logpxi_array,axis=1))