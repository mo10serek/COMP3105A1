# COMP 3105 Fall 2022 Assignment 1
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
import numpy as np
from matplotlib import pyplot as plt
import A1codes


def augmentX(X):
	# append a column of ones to X
    n = X.shape[0]
    return np.concatenate((np.ones((n, 1)), X),  axis=1)


def _plotReg():

	# simple 2D example
	n = 30  # number of data points
	d = 1  # dimension
	noise = 0.2  # noise level
	X = np.random.randn(n, d)  # input matrix
	y = X + np.random.randn(n, 1) * noise + 2  # ground truth label

	plt.scatter(X, y, marker='x', color='k')  # plot data points

	# learning
	X = augmentX(X)
	w_L2 = A1codes.minimizeL2(X, y)
	y_hat_L2 = X @ w_L2
	w_L1 = A1codes.minimizeL1(X, y)
	y_hat_L1 = X @ w_L1
	w_Linf = A1codes.minimizeLinf(X, y)
	y_hat_Linf = X @ w_Linf

	# plot models
	plt.plot(X[:, 1], y_hat_L2, 'b', marker=None)
	plt.plot(X[:, 1], y_hat_L1, 'r', marker=None)
	plt.plot(X[:, 1], y_hat_Linf, 'g', marker=None)
	plt.show()


def _plotCls():

	# 2D classification example
	m = 100
	d = 2
	c0 = np.array([[1, 1]])  # cls 0 center
	c1 = np.array([[-1, -1]])  # cls 1 center
	X0 = np.random.randn(m, 2) + c0
	X1 = np.random.randn(m, 2) + c1

	# plot data points
	plt.scatter(X0[:, 0], X0[:, 1], marker='x')
	plt.scatter(X1[:, 0], X1[:, 1], marker='o')

	X = np.concatenate((X0, X1), axis=0)
	X = augmentX(X)
	y = np.concatenate([np.zeros([m, 1]), np.ones([m, 1])], axis=0)  # class labels

	# gradient descent
	w_gd = A1codes.geb(A1codes.linearRegL20bj, np.random.randn(d + 1, 1), X, y, 0.1, 1000, 1e-10)

	# plot models
	x_grid = np.arange(-4, 4, 0.01)
	plt.plot(x_grid, (-w_gd[0]-w_gd[1]*x_grid) / w_gd[2], '--k')
	plt.show()


if __name__ == "__main__":

	_plotReg()
	_plotCls()
