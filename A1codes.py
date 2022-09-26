import cvxopt
import numpy as np
import cvxopt.solvers as solvers
from cvxopt import matrix
import math
from autograd import grad
from matplotlib import pyplot as plt
#from jax import grad

w = np.array([[3], [4]])
X = np.array([[4, 1], [3, 3], [7, 8]])
y = np.array([[6], [13], [23]])

def L2Norm(vector):
    sum = 0
    print(vector)
    for counter in range(len(vector)):
        sum = sum + math.pow(vector[counter], 2)
    return sum

def squaredLoss(y_hats, y_actual, n):
    a = y_hats - y_actual
    sum = 0
    for counter in range(len(a)):
        sum = sum + math.pow(a[counter], 2)
    sum = sum/(2 * n)
    return sum

def minimizeL2(X, y):
    #print("Starting L2")
    XT = X.T

    w = np.matmul(np.linalg.matrix_power(np.matmul(XT,X),-1),np.matmul(XT,y))

    return w

def absoluteLoss(y_hats, y_actual, n):
    a = y_hats - y_actual
    sum = 0
    for counter in range(len(a)):
        sum = sum + math.abs(a[counter])
    sum = sum/n
    return sum

def minimizeL1(X, y):
    #print("Starting L1")
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
    return w["x"][0:d]

def infinityLoss(y_hats, y_actual, n):
    a = y_hats - y_actual
    sum = 0
    for counter in range(len(a)):
        sum = sum + math.abs(a[counter])
    return sum

def minimizeLinf(X, y):
    #print("Starting Linf")
    n = len(X)
    d = len(X[0])
    negativeIdentityVectorNRow = np.ones(n)
    negativeIdentityVectorNCol = negativeIdentityVectorNRow.reshape((n,1))
    negativeIdentityVectorN = -negativeIdentityVectorNCol.astype(int)
    emptyMatrixNxD = np.zeros((n, d)).astype(int)
    negativeX = np.negative(X)

    GFirstRow = np.concatenate((emptyMatrixNxD, negativeIdentityVectorN), axis=1)
    GSecondRow = np.concatenate((X, negativeIdentityVectorN), axis=1)
    GThirdRow = np.concatenate((negativeX, negativeIdentityVectorN), axis=1)

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
    secondHathOfC = np.ones(1)

    c = np.concatenate((firstHalfOfC, secondHathOfC))
    c = matrix(c)

    w = solvers.lp(c, G, h)
    return w["x"][0:d]

def synRegExperiments():

    d = 5  # dimension
    noise = 0.2
    for counter in range(100):
        n = 30  # number of data points
        X = np.random.randn(n, d)  # input matrix
        X = np.concatenate((np.ones((n, 1)), X), axis=1)  # augment input
        w_true = np.random.randn(d + 1, 1)  # true model parameters
        y = X @ w_true + np.random.randn(n, 1) * noise  # ground truth label

        l2w = minimizeL2(X, y)
        l1w = minimizeL1(X, y)
        linfw = minimizeLinf(X, y)

        l2modelOnL2Loss = squaredLoss(np.matmul(X, l2w), y, n)
        l2modelOnL1Loss = squaredLoss(np.matmul(X, l1w), y, n)
        l2modelOnLInfLoss = squaredLoss(np.matmul(X, linfw), y, n)
        l1modelOnL2Loss = squaredLoss(np.matmul(X, l2w), y, n)
        l1modelOnL1Loss = squaredLoss(np.matmul(X, l1w), y, n)
        l1modelOnLInfLoss = squaredLoss(np.matmul(X, linfw), y, n)
        lInfModelOnL2Loss = squaredLoss(np.matmul(X, l2w), y, n)
        lInfModelOnL1Loss = squaredLoss(np.matmul(X, l1w), y, n)
        lInfModelOnLInfLoss = squaredLoss(np.matmul(X, linfw), y, n)

        trainingLoss = np.array([[l2modelOnL2Loss, l2modelOnL1Loss, l2modelOnLInfLoss],
                                 [l1modelOnL2Loss, l1modelOnL1Loss, l1modelOnLInfLoss],
                                 [lInfModelOnL2Loss, lInfModelOnL1Loss, lInfModelOnLInfLoss]])

        n = 1000
        X = np.random.randn(n, d)  # input matrix
        X = np.concatenate((np.ones((n, 1)), X), axis=1)  # augment input
        w_true = np.random.randn(d + 1, 1)  # true model parameters
        y = X @ w_true + np.random.randn(n, 1) * noise  # ground truth label

        l2w = minimizeL2(X, y)
        l1w = minimizeL1(X, y)
        linfw = minimizeLinf(X, y)

        l2modelOnL2Loss = squaredLoss(np.matmul(X, l2w), y, n)
        l2modelOnL1Loss = squaredLoss(np.matmul(X, l1w), y, n)
        l2modelOnLInfLoss = squaredLoss(np.matmul(X, linfw), y, n)
        l1modelOnL2Loss = squaredLoss(np.matmul(X, l2w), y, n)
        l1modelOnL1Loss = squaredLoss(np.matmul(X, l1w), y, n)
        l1modelOnLInfLoss = squaredLoss(np.matmul(X, linfw), y, n)
        lInfModelOnL2Loss = squaredLoss(np.matmul(X, l2w), y, n)
        lInfModelOnL1Loss = squaredLoss(np.matmul(X, l1w), y, n)
        lInfModelOnLInfLoss = squaredLoss(np.matmul(X, linfw), y, n)

        testLoss = np.array([[l2modelOnL2Loss, l2modelOnL1Loss, l2modelOnLInfLoss],
                            [l1modelOnL2Loss, l1modelOnL1Loss, l1modelOnLInfLoss],
                            [lInfModelOnL2Loss, lInfModelOnL1Loss, lInfModelOnLInfLoss]])

    return trainingLoss, testLoss


def linearRegL20bj(w, X, y):
    n = len(X)

    a = np.matmul(X, w) + y
    total = 0

    for counter in range(len(a)):
        total = total + math.pow(a[counter], 2)

    obj_func = total/(2*n)

    grad = (X * a)/ n

    return obj_func, grad

def geb(obj_func, w_init, X, y, eta, max_iter, tol):
    w = w_init

    for counter in range(max_iter):
        w = obj_func(w, X, y)
        if (L2Norm(w) < tol):
            break
        w = w - eta * obj_func(w, X, y)
    return w

def logisticRegObj(w, X, y):
    n = len(X[0])

    Xw = np.matmul(X, w)

    print(Xw)
    a_ = np.logaddexp(0, -Xw)
    print(a_)
    print(y)
    print()

    leftPart = y * np.logaddexp(0, -Xw)

    rightPart = (1 - y) * (0.4342944819 * Xw + np.logaddexp(0, -Xw))

    obj_value = (leftPart + rightPart)/n

    grad = (np.logaddexp(0, Xw) - y) * X

    return obj_value, grad


def synClsExperiments():

    for counter in range(4):
        for counter2 in range(3):
            # data generation
            if counter2 == 0:
                if counter == 0:
                    m = 10
                else:
                    m = 50 * math.pow(2, (counter - 1))
            else:
                m = 100  # number of data points *per class*
            if counter2 == 1:
                d = math.pow(2, counter)
            else:
                d = 2  # dimension
            c0 = np.ones([1, d])  # class 0 center
            c1 = -np.ones([1, d])  # class 1 center
            X0 = np.random.randn(m, d) + c0  # class 0 input
            X1 = np.random.randn(m, d) + c1  # class 1 input
            X = np.concatenate((X0, X1), axis=0)
            X = np.concatenate((np.ones((2 * m, 1)), X), axis=1)  # augmentation
            y = np.concatenate([np.zeros([m, 1]), np.ones([m, 1])], axis=0)
            # learning
            if counter2 == 2:
                eta = 0.1 * (10^counter)
            else:
                eta = 0.1  # learning rate
            max_iter = 1000  # maximum number of iterations
            tol = 1e-10  # tolerance
            w_init = np.random.randn(d + 1, 1)
            w_logit = gb(logisticRegObj, w_init, X, y, eta, max_iter, tol)

#synRegExperiments()

#obj_func, grad = linearRegL20bj(w, X, y)

#print(obj_func)
#print(grad)

#obj_func, grad = logisticRegObj(w, X, y)

#print(obj_func)
#print(grad)
#print(round(np.exp(np.log(5))))

def loadData(dataset_folder, dataset_name):
    #auto-mpg remove origin and car name columns, and any rows with missing data, mpg is y, rest is X
    #parkinsons use status as y, rest as X
    #Sonar 

    X = np.zeros(3,5)
    return X,y