import numpy as np
from estGaussMixEM import estGaussMixEM
from EM.getLogLikelihood import getLogLikelihood


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    skin_gaussian_weights, skin_gaussian_means, skin_gaussian_covariances = estGaussMixEM(sdata, K, n_iter, epsilon)
    non_skin_gaussian_weights, non_skin_gaussian_means, non_skin_gaussian_covariances = estGaussMixEM(ndata, K, n_iter, epsilon)

    height, width, _ = img.shape

    non_skin_log = np.ndarray((height, width))
    skin_log = np.ndarray((height, width))
    result = np.ndarray((height, width))

    for i in range (height):
        for j in range(width):
            skin_log[i, j] = np.exp(getLogLikelihood(skin_gaussian_means, skin_gaussian_weights, skin_gaussian_covariances, np.array([img[i, j, 0], img[i, j, 1], img[i, j, 2]])))
            non_skin_log[i, j] = np.exp(getLogLikelihood(non_skin_gaussian_means, non_skin_gaussian_weights, non_skin_gaussian_covariances, np.array([img[i, j, 0], img[i, j, 1], img[i, j, 2]])))

            result[i, j] = skin_log[i, j] / non_skin_log[i, j]
            result[i, j] = 1 if result[i, j] > theta else 0
    



    return result
