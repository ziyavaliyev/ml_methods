import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
import cvxopt


def svmlin(X, t, C):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)


    #####Insert your code here for subtask 2a#####
    N = X.shape[0]
    zero_vector = np.zeros(N)
    ones_vector = np.ones(N)

    # Compute H-Matrix
    H = np.empty((N,N))

    for n in range(N):
        for m in range(N):
            H[n][m] = t[n]*t[m]*np.dot(X[n], X[m])

    q = (-1)* np.ones(N)

    n = H.shape[1]
    G = np.vstack([-np.eye(n), np.eye(n)])
    b = np.double(0)
    A = t.reshape((1, N))

    LB = np.zeros(N)
    UB = np.ones(N) * C
    h = np.hstack([-LB, UB])

    f = (-1) * np.ones(N)

    res = cvxopt.solvers.qp(P=cvxopt.matrix(H), q=cvxopt.matrix(f), G=cvxopt.matrix(G), h=cvxopt.matrix(h), A=cvxopt.matrix(A), b=cvxopt.matrix(b))

    alpha = np.array(res['x']).reshape((-1,))

    sv = np.where(alpha > 1e-6, True, False)

    if not sv.any():
        raise ValueError('No support vectors found!')
    else:
        slack = np.where(alpha > C - 1e-6, True, False)

        w = (alpha[sv] * t[sv]).dot(X[sv])
        b = np.mean(t[sv] - w.dot(X[sv].T))
        result = X.dot(w) + b

    return alpha, sv, w, b, result, slack