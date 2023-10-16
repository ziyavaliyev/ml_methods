import numpy as np
from gauss1D import gauss1D
from parameters import parameters


def k(u, h):
    exponent = -0.5 * (u**2) / (h**2)
    prefactor = 1.0 / (np.sqrt(2 * np.pi) * h)
    return np.exp(exponent) * prefactor


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created

    pos = np.arange(-5, 5, 0.1)

    KK = 0
    den = []
    for j in range (len(pos)):
        KK = 0
        for i in range (len(pos)):
            KK += k(samples[i]-pos[j], h)
        den.append(float(KK/len(pos)))
    #den = np.array(den)
    estDensity = np.array([[position for position in pos], [density for density in den]])
    #stack(pos, den)
    estDensity = np.transpose(estDensity)

    return estDensity
