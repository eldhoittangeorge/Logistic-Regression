import numpy as np





def costFunc(trainedOutputVector, originalOutputVector):
    cost = 0
    size = trainedOutputVector.shape

    first = np.multiply(-originalOutputVector, np.log(trainedOutputVector))
    second = np.multiply((1 - originalOutputVector), np.log(1 - trainedOutputVector))
    cost = np.sum(first - second)
    cost = (1 / size[0]) * cost
    return cost
