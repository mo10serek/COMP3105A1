import cvxopt
import numpy as np
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
    d = len(X[0])
    identityMatrixN = np.identity(n)
    negativeIdentityMatrixN = np.negative(identityMatrixN)
    negativeIdentityMatrixN = negativeIdentityMatrixN.astype(int)
    emptyMatrixNxD = np.zeros((n, d)).astype(int)
    negativeX = np.negative(X)

    GFirstRow = np.concatenate((emptyMatrixNxD, negativeIdentityMatrixN), axis=1)
    GSecondRow = np.concatenate((X, negativeIdentityMatrixN), axis=1)
    GThirdRow = np.concatenate((negativeX, negativeIdentityMatrixN), axis=1)

    G = np.concatenate((GFirstRow, GSecondRow), axis=0)
    G = np.concatenate((G, GThirdRow), axis=0)
    G = G.astype(np.double)
    print("before transpose", G)
    #G = G.transpose()
    G = matrix(G)
    print("after transpose", G)

    emptyVectorN = np.zeros((n, 1)).astype(int)
    negativeY = np.negative(y)


    h = np.concatenate((emptyVectorN, y), axis=0)
    h = np.concatenate((h, negativeY), axis=0)
    h = h.astype(np.double)
    h = h.flatten()
    h = matrix(h)
    print(h)

    firstHalfOfC = np.zeros(d)
    secondHathOfC = np.ones(n)

    c = np.concatenate((firstHalfOfC, secondHathOfC))
    c = matrix(c)
    print(c)

    w = solvers.lp(c, G, h)
    return w

def minimizeLinf(X, y):
    n = len(X)
    d = len(X[0])

from cvxopt import matrix, solvers

A = matrix([[0., 0., 1., 4., -1., -4.],[0., 0., 2., 5., -2., -5.], [0., 0., 3., 6., -3., -6.], [-1., 0., -1., 0., -1., 0.],[0., -1., 0., -1., 0., -1.]])
b = matrix([0.0, 0.0, 3.0, 5.0, -3.0, -5.0])
c = matrix([0., 0., 0., 1., 1.])

print(A)
print(b)
print(c)

sol=solvers.lp(c,A,b)
print(sol['x'])

X = np.array([[1,2,3],[4,5,6]])

y = np.array([[3],[5]])

w = minimizeL1(X, y)
print(w['x'])



#def minimizeLinf(X, y):

