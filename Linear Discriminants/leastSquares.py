import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    ones_column = np.ones((data.shape[0], 1))
    X = np.hstack((ones_column, data))
    W = np.matmul(np.linalg.pinv(X), label)
    
    bias = W[0]
    weight = W[1:]

    #erste Spalte von w ist w0
    #zweite und dritte ist w1 und w2

    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    return weight, bias
