from numpy import *
from numpy.linalg import det

from DataGenerator import generateTwoDimensionalData
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X=mat(zeros((3,3)))
    print(X.T*X)