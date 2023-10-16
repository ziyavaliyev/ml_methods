import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    pos = np.arange(-5, 5.0, 0.01)
    N = len(pos)
    res = []
    for i in range (N):
        kn = []
        for j in range(N):
            kn.append(abs(pos[i] - samples[j]))
        V = 2*(np.sort(kn))[k-1]
        p = k/(N*V)
        res.append(p)
    
    estDensity = np.transpose(np.array([pos, [probability for probability in res]]))


    return estDensity
