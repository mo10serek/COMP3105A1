import numpy.linalg.solve as np
import cvxopt.solvers as solvers
import math
from matplotlib import pyplot as plt

w = np.array([[3], [4]])
X = np.array([[4, 1], [3, 3], [7, 8]])
y = np.array([[6], [13], [23]])

def minimizeL2(X, y):
    XInverse = np.linalg.inv(X);

    w = XInverse * y

    return w

def minimizeL1(X, y):
    n = len(X)

    y_ = np.matmul(X, w)
    fn = y_ - y

    sum = 0
    for counter in fn:
        for counter2 in counter:
            sum = sum + math.fabs(counter2)

    sum = sum / n

#def minimizeLinf(X, y):

