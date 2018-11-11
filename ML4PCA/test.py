import numpy as np
from numpy import *
from numpy import linalg as la
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a=np.arange(30).reshape(5,6)
    print(a,end="\n\n\n")
    print(a[0:3][:,2:4])