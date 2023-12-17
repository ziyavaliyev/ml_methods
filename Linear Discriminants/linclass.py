import numpy as np

def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    # Perform linear classification i.e. class prediction
    class_pred = []

    for i in range (data.shape[0]):
        class_pred.append(1 if np.matmul(np.transpose(weight), data[i]) + bias > 0 else -1)
    return class_pred


