import numpy as np

def normal(mean, covariance, x, N, K, D):
    factor = 1/(((2*np.pi)**(D/2)) * ((np.linalg.det(covariance))**(1/2)))
    exponent = -0.5* np.dot(np.dot((x-mean), np.linalg.inv(covariance)), np.transpose(x-mean))
    return factor*np.exp(exponent)

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians +
    # covariances    : Covariance matrices for each gaussian DxDxK 
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    
    logLikelihood = 0
    bracket = []
    N, D = X.shape
    K = len(weights)
    for n in range(N):
        bracket = []
        for k in range(K):
            bracket.append(weights[k]*normal(means[k], covariances[:, :, k], X[n], N, K, D)) # "(x_n|mu_k, SIGMA_k)"
        logLikelihood += np.log(sum(bracket))
    return logLikelihood

