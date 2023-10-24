import numpy as np
from EM.getLogLikelihood import getLogLikelihood, normal

def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    N, D = X.shape
    K = len(weights)
    gamma = []
    for n in range(N): # for each data point
        gamma_j = []
        for j in range(K): # for each Gaussian
            gamma_j_zaehler = weights[j]*normal(means[j], covariances[:, :, j], X[n], N, K, D) # "(x_n|mu_j, SIGMA_j)"
            gamma_j_nenner = sum([weights[k]*normal(means[k], covariances[:, :, k], X[n], N, K, D) for k in range(K)])
            gamma_j.append(gamma_j_zaehler/gamma_j_nenner)
        gamma.append(gamma_j)
    return [logLikelihood, gamma]
