import numpy as np
from getLogLikelihood import getLogLikelihood
from EStep import EStep


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####

    N, K = gamma.shape
    D = X.shape[1]
    N_ = 0
    N_j = []
    pi_j = []
    mu_j = []
    for j in range (K):
        N_ = 0 
        gamma_j_x = 0
        for n in range (N):
            N_ += gamma[n][j]
            gamma_j_x += gamma[n][j] * X[n]
        muj = gamma_j_x / N_

        N_j.append(N_)
        pi_j.append(N_/N)
        mu_j.append(muj)

    weights = pi_j
    means = mu_j

    covariances_1 = []
    for j in range (K):
        covariance_j = 0
        for n in range (N):
            covariance_j += gamma[n][j] * np.outer(np.transpose(X[n] - mu_j[j]), (X[n] - mu_j[j]))
        covariances_1.append(covariance_j)
        #covariances = covariances/N_j[j]
    covariances_1 = np.transpose(covariances_1, (2, 1, 0))
    for j in range (K):
        covariances_1[:,:,j] = covariances_1[:,:,j]/N_j[j]
    
    covariances = covariances_1
    logLikelihood = getLogLikelihood(means, weights, covariances, X)


    return weights, means, covariances, logLikelihood
