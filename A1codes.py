import cvxopt
import pandas as pd
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
    print(len(vector))
    for counter in range(len(vector)):
        try:
            sum = sum + math.pow(vector[counter], 2)
        except OverflowError:
            sum = sum + float('inf')
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

    f = open("trainingData.txt", "w")

    d = 5  # dimension
    noise = 0.2

    trainingLoss = np.empty((3, 3))
    testLoss = np.empty((3, 3))
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

        createTable(trainingLoss, "training loss", f)

        n = 1000
        X = np.random.randn(n, d)  # input matrix
        X = np.concatenate((np.ones((n, 1)), X), axis=1)  # augment input
        #w_true = np.random.randn(d + 1, 1)  # true model parameters
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

        createTable(testLoss, "test loss", f)

    f.close()
    return trainingLoss, testLoss

def linearRegL20bj(w, X, y):
    n = len(X)

    a = np.matmul(X, w) + y

    total = 0

    for counter in range(len(a)):
        total = total + math.pow(a[counter], 2)

    obj_val = total/(2*n)

    XT = X.T

    inside = np.matmul(X, w) - y

    grad = np.matmul(XT, inside)/ n

    return obj_val, grad

def geb(obj_func, w_init, X, y, eta, max_iter, tol):
    w = w_init

    for counter in range(max_iter):
        obj_val, grad = obj_func(w, X, y)
        if (L2Norm(grad) < tol):
            break
        w = eta * grad
    return w

def logisticRegObj(w, X, y):
    n = len(X[0])

    Xw = np.matmul(X, w)

    leftPart = y * np.logaddexp(0, -Xw)

    rightPart = (1 - y) * (0.4342944819 * Xw + np.logaddexp(0, -Xw))

    obj_value = (leftPart + rightPart)/n

    y_hat_munis_y = (np.logaddexp(0, Xw) - y)

    grad = np.matmul(y_hat_munis_y.T, X)

    grad = grad[0]
    return obj_value, grad

def synClsExperiments():

    f = open("trainingData.txt", "w")

    for step in range(100):
        train_acc = np.empty((4, 3))
        test_acc = np.empty((4, 3))

        for counter in range(4):
            for counter2 in range(3):
                # data generation
                if counter2 == 0:
                    if counter == 0:
                        m = 10
                    else:
                        m = 50 * int(math.pow(2, (counter - 1)))
                else:
                    m = 100  # number of data points *per class*
                if counter2 == 1:
                    d = int(math.pow(2, counter))
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
                w_logit = geb(logisticRegObj, w_init, X, y, eta, max_iter, tol)

                w_average = 0
                for counter3 in range(len(w_logit)):
                    w_average = w_logit[counter3]
                w_average = w_average/len(w_logit)

                train_acc[counter][counter2] = w_average

        for counter in range(4):
            for counter2 in range(3):
                # data generation
                if counter2 == 0:
                    if counter == 0:
                        m = 10
                    else:
                        m = 50 * int(math.pow(2, (counter - 1)))
                else:
                    m = 100  # number of data points *per class*
                if counter2 == 1:
                    d = int(math.pow(2, counter))
                else:
                    d = 2  # dimension
                m = m + 1000
                c0 = np.ones([1, d])  # class 0 center
                c1 = -np.ones([1, d])  # class 1 center
                X0 = np.random.randn(m, d) + c0  # class 0 input
                X1 = np.random.randn(m, d) + c1  # class 1 input
                X = np.concatenate((X0, X1), axis=0)
                X = np.concatenate((np.ones((2 * m, 1)), X), axis=1)  # augmentation
                y = np.concatenate([np.zeros([m, 1]), np.ones([m, 1])], axis=0)
                # learning
                if counter2 == 2:
                    eta = 0.1 * (10 ^ counter)
                else:
                    eta = 0.1  # learning rate
                max_iter = 1000  # maximum number of iterations
                tol = 1e-10  # tolerance
                w_init = np.random.randn(d + 1, 1)
                w_logit = geb(logisticRegObj, w_init, X, y, eta, max_iter, tol)

                w_average = 0
                for counter3 in range(len(w_logit)):
                    w_average = w_logit[counter3]
                w_average = w_average / len(w_logit)

                test_acc[counter][counter2] = w_average
    f.close()

    print(train_acc)
    return train_acc, test_acc

def createTable(table, name, f):

    if name == "training loss":
        f.write("\n")
        f.write("Table: Different training losses for different models\n")
        f.write("   Model     | L_2 loss  | L_2 loss  | L_inf loss\n")
        f.write("  L_2 Model  |" + str(table[0][0]) + " | " + str(table[0][1]) + " | " + str(table[0][2]) + "\n")
        f.write("  L_1 Model  |" + str(table[1][0]) + " | " + str(table[1][1]) + " | " + str(table[1][2]) + "\n")
        f.write(" L_inf Model |" + str(table[2][0]) + " | " + str(table[2][1]) + " | " + str(table[2][2]) + "\n")
        f.write("\n")

    if name == "test loss":
        f.write("\n")
        f.write("Table: Different test losses for different models\n")
        f.write("   Model     | L_2 loss  | L_2 loss  | L_inf loss\n")
        f.write("  L_2 Model  |" + str(table[0][0]) + " | " + str(table[0][1]) + " | " + str(table[0][2]) + "\n")
        f.write("  L_1 Model  |" + str(table[1][0]) + " | " + str(table[1][1]) + " | " + str(table[1][2]) + "\n")
        f.write(" L_inf Model |" + str(table[2][0]) + " | " + str(table[2][1]) + " | " + str(table[2][2]) + "\n")
        f.write("\n")

    if name == "training accuracy":
        f.write("\n")
        f.write("Table: Training accuracies with different hyper-parameters\n")
        f.write(" n Train Accuracy  | d Train Accuracy  | eta  Train Accuracy\n")
        f.write(" 10  " + str(table[0][0]) + " | 1 " + str(table[0][1]) + " | 0.001 " + str(table[0][2]) + "\n")
        f.write(" 50  " + str(table[1][0]) + " | 2 " + str(table[1][1]) + " | 0.01  " + str(table[1][2]) + "\n")
        f.write(" 100 " + str(table[2][0]) + " | 3 " + str(table[2][1]) + " | 0.1   " + str(table[2][2]) + "\n")
        f.write(" 200 " + str(table[3][0]) + " | 4 " + str(table[3][1]) + " | 1.0   " + str(table[3][2]) + "\n")
        f.write("\n")

    if name == "test accuracy":
        f.write("\n")
        f.write("Table: Training accuracies with different hyper-parameters\n")
        f.write(" n Train Accuracy  | d Train Accuracy  | eta  Train Accuracy\n")
        f.write(" 10   " + str(table[0][0]) + " | 1 " + str(table[0][1]) + " | 0.001 " + str(table[0][2]) + "\n")
        f.write(" 50   " + str(table[1][0]) + " | 2 " + str(table[1][1]) + " | 0.01  " + str(table[1][2]) + "\n")
        f.write(" 100  " + str(table[2][0]) + " | 3 " + str(table[2][1]) + " | 0.1   " + str(table[2][2]) + "\n")
        f.write(" 200  " + str(table[3][0]) + " | 4 " + str(table[3][1]) + " | 1.0   " + str(table[3][2]) + "\n")
        f.write(" 1000 " + str(table[3][0]) + " | 4 " + str(table[3][1]) + " | 1.0   " + str(table[3][2]) + "\n")
        f.write("\n")

#synRegExperiments()
synClsExperiments()

def loadData(dataset_folder, dataset_name):
    #auto-mpg remove origin and car name columns, and any rows with missing data, mpg is y, rest is X
    #parkinsons use status as y, rest as X
    #Sonar last column will be split off to labels, with Rs changing to 0 and Ms to 1s, the rest will be input features
    
    #read file
    
    #input file data into matrixies
    if(dataset_name == "auto-mpg.data"):
        data = pd.read_csv(dataset_folder,delim_whitespace=True)
        print("\n\n")
        data.columns=["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"]
        print(data)
        data = data.drop(columns=["origin","car name"])
        y = data["mpg"]
        X = data.drop(columns=["mpg"])
        X = X[X.horsepower != "?"]
        X = X.to_numpy()
        y = y.to_numpy()
        print(y)
        print(X)
        return X,y
    elif(dataset_name == "parkinsons.data"):
        data = pd.read_csv(dataset_folder,sep=",")
        y = data["status"]
        X = data.drop(columns="status")
        print("\n\n")
        X = X.to_numpy()
        y = y.to_numpy()
        print(y)
        print(X)
        return X,y
        
    elif(dataset_name == "sonar.all-data"):
        print("\n\n")
        data = pd.read_csv(dataset_folder,sep=",")
        y = np.where(data["R"]=='R',0,1)
        #have to convert y into 1s and 0s rather than Rs and Ms
        
        X = data.drop(columns="R")
        X = X.to_numpy()
        print(y)
        print(X)
        return X,y
    return




def realExperiments(dataset_folder,dataset_name):

    return


loadData("C:/Users/Kubaz/Downloads/parkinsons.data", "parkinsons.data")