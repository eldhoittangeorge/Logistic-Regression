import numpy as np
from lrCost import costFunc

cost = 0
hOne = 0
hTwo = 0
hThree = 0

totalData = np.loadtxt("ex2data1.txt", delimiter=',')

trainingDataFeatures = np.hstack((np.ones((100, 1)), totalData[0:101, 0:2]))

trainingDataOutputs = totalData[0:101, 2:3]

dataSize = totalData.shape

iniWt = np.zeros((3, 1))


def sigmoidFunc(array, theta):
    hyp = np.dot(array, theta)
    return (1 / (1 + np.exp(-hyp)))


sigmoid = sigmoidFunc(trainingDataFeatures, iniWt)

print("the cost before training {}".format(costFunc(sigmoid, trainingDataOutputs)))

for i in range(100000):

    for j in range(0, dataSize[0]):
        hOne = hOne + ((sigmoid[j] - trainingDataOutputs[j]) * trainingDataFeatures[j, 0])
        hTwo = hTwo + ((sigmoid[j] - trainingDataOutputs[j]) * trainingDataFeatures[j, 1])
        hThree = hThree + ((sigmoid[j] - trainingDataOutputs[j]) * trainingDataFeatures[j, 2])

    iniWt[0] = iniWt[0] - (0.001 / dataSize[0]) * (hOne)
    iniWt[1] = iniWt[0] - (0.001 / dataSize[0]) * (hTwo)
    iniWt[2] = iniWt[0] - (0.001 / dataSize[0]) * (hThree)

    sigmoid = sigmoidFunc(trainingDataFeatures, iniWt)

result = float(1 / (1 + np.exp(-(iniWt[0] + iniWt[1] * 45 + iniWt[2] * 85))))  # sigmoid
print("the cost after training {}".format(costFunc(sigmoid, trainingDataOutputs)))
print("the outputs is {} ".format(result))
