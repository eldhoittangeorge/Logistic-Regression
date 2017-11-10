import numpy as np


'''
totalData = np.loadtxt("ex2data1.txt",delimeter=',')
dataSize = totalData.shape
trainingDataFeatures = np.hstack((hp.ones((dataSize[0])),totalData[0:dataSize[0]+1,0:2]))
trainingOutputs = totalData[0:dataSize[0]+1,2:3]

iniWt = np.zeros((3,1))
'''


def costFunc(trainedOutputVector, originalOutputVector):
    cost = 0
    size = trainedOutputVector.shape
    for i in range(0, size[0]):
        cost = cost + (originalOutputVector[i] * np.log(trainedOutputVector[i]) + (1 - originalOutputVector[i]) * np.log(1 - trainedOutputVector[i]))
    cost = -(1 / size[0]) * cost
    return cost
