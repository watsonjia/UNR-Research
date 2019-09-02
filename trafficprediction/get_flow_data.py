import numpy as np, matplotlib.pyplot as plt, readData
days = readData.days
flow = np.array(readData.flow)
flowList = np.array(readData.flowList)
time = np.array(readData.time)
postMile = np.array(readData.postMile)
lanes = np.array(readData.lanes)
flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))
flowArray = []
for i, val in enumerate(flow):
    flowArray.append(np.array(flow[i].reshape(288, 136)))

flowArray = np.asarray(flowArray)
timeSlot = 288
points = 136
m = input('Enter a time interval size: ')
q = 3
pTmp = 5
p = (pTmp - 1) / 2

def DataSet(fPM):
    inputArray = []
    outputArray = []
    outputArrayList = []
    col = np.where(postMile == fPM)[0][0]
    k = col
    tmpMatrix = np.array([])
    tmpMatrix2 = np.array([])
    for i in range(14, days):
        for j in range(m, timeSlot - 24):
            tmpMatrix = np.append(tmpMatrix, flowArray[i, j - m:j, k])
            inputArray.append(tmpMatrix)
            outputArray.append([flowArray[(i, j, k)]])
            tmpMatrix2 = np.append(tmpMatrix2, flowArray[i, j:j + 24, k])
            outputArrayList.append(np.array([tmpMatrix2]))
            tmpMatrix = np.array([])
            tmpMatrix2 = np.array([])

    inputArray = np.array(inputArray).T
    outputArray = np.array(outputArray).T
    outputArrayList = np.array(outputArrayList).T
    return (
     inputArray, outputArray, outputArrayList)
