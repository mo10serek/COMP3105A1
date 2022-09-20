import cvxopt
import numpy as np
import cvxopt.solvers as solvers
from cvxopt import matrix;
import math
from matplotlib import pyplot as plt

w = np.array([[3], [4]])
X = np.array([[4, 1], [3, 3], [7, 8]])
y = np.array([[6], [13], [23]])

def minimizeL2(X, y):
    print("Starting L2")
    XT = X.T

    w = np.matmul((np.matmul(XT,X)^-1),np.matmul(XT,y))

    return w

def minimizeL1(X, y):
    print("Starting L1")
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
    G = matrix(G)

    emptyVectorN = np.zeros((n, 1)).astype(int)
    negativeY = np.negative(y)

    h = np.concatenate((emptyVectorN, y), axis=0)
    h = np.concatenate((h, negativeY), axis=0)
    h = h.astype(np.double)
    h = h.flatten()
    h = matrix(h)

    firstHalfOfC = np.zeros(d)
    secondHathOfC = np.ones(n)

    c = np.concatenate((firstHalfOfC, secondHathOfC))
    c = matrix(c)

    w = solvers.lp(c, G, h)
    return w

def minimizeLinf(X, y):
    print("Starting Linf")
    n = len(X)
    d = len(X[0])
    negativeIdentityVectorNRow = np.ones(n)
    negativeIdentityVectorNCol = negativeIdentityVectorNRow.reshape((n,1))
    negativeIdentityVectorN = negativeIdentityVectorNCol.astype(int)
    emptyMatrixNxD = np.zeros((n, d)).astype(int)
    negativeX = np.negative(X)

    GFirstRow = np.concatenate((emptyMatrixNxD, negativeIdentityVectorN), axis=1)
    GSecondRow = np.concatenate((X, negativeIdentityVectorN), axis=1)
    GThirdRow = np.concatenate((negativeX, negativeIdentityVectorN), axis=1)

    G = np.concatenate((GFirstRow, GSecondRow), axis=0)
    G = np.concatenate((G, GThirdRow), axis=0)
    G = G.astype(np.double)
    G = matrix(G)

    print(G)

    emptyVectorN = np.zeros((n, 1)).astype(int)
    negativeY = np.negative(y)

    h = np.concatenate((emptyVectorN, y), axis=0)
    h = np.concatenate((h, negativeY), axis=0)
    h = h.astype(np.double)
    h = h.flatten()
    h = matrix(h)

    print(h)

    firstHalfOfC = np.zeros(d)
    secondHathOfC = np.ones(1)

    c = np.concatenate((firstHalfOfC, secondHathOfC))
    c = matrix(c)

    print(c)

    w = solvers.lp(c, G, h)
    return w

#def linearRegL20bj(x, X, y):



X = np.array([[5, 3, 8],[34, 32, 1], [87, 5, 9]])

y = np.array([[3],[5], [8]])

w = minimizeLinf(X, y)
print(w)



def loadData(dataset_folder, dataset_name):
    #auto-mpg remove origin and car name columns, and any rows with missing data, mpg is y, rest is X
    #parkinsons use status as y, rest as X
    #Sonar 

    X = np.zeros(3,5)
    return X,y