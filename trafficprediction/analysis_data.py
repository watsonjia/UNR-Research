import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from itertools import groupby
from operator import itemgetter
import readData

np.set_printoptions(threshold=np.nan)

days = readData.days
flow = np.array(readData.flow)
flowList = np.array(readData.flowList)
time = np.array(readData.time)
postMile = np.array(readData.postMile)
lanes = np.array(readData.lanes)

#flow = (flow - np.min(flow))/(np.max(flow) - np.min(flow))

flowArray = []

for i, val in enumerate(flow):
    flowArray.append(np.array(flow[i].reshape(24*12, 136)))
    

flowArray = np.asarray(flowArray)

sFlow = []
point=65
filterSize = 3





for i in range(0, 136):
    flowAtPoint = flowArray[:, :, i].flatten()
    for j in range(len(flowAtPoint) - filterSize):
        flowAtPoint[j] = np.mean(flowAtPoint[j:j+filterSize])
    sFlow.append(np.array(flowAtPoint.reshape(31, 288)))
#print(np.shape(sFlow))

sFlow = np.asarray(sFlow)

'''
plt.subplot(211)
plt.plot(flowArray[0, :, 0])
plt.xlabel("Time slot", fontsize=24)
plt.ylabel("Traffic", fontsize=24)
plt.axis([0, 288, 0, 340])
plt.text(120, 290, "Actual traffic", fontsize=18)

plt.subplot(212)
plt.plot(sFlow[0, 0, :].T)
plt.xlabel("Time slot", fontsize=24)
plt.ylabel("Traffic", fontsize=24)
plt.axis([0, 288, 0, 340])
plt.text(120, 290, "Actual traffic", fontsize=18)

plt.subplots_adjust(left=None, wspace=0.3, hspace=0.4, top=None)

plt.show()
'''


'''
for i in range(0, 136):
    flowAtPoint = flowArray[:, 100, i]
    print(np.shape(flowAtPoint))
    a = range(0, 31)
    flowAtPoint.flatten()
    print(np.shape(flowAtPoint))
    plt.scatter(a, flowAtPoint.flatten())
    plt.axis([-1, 32, 0, 1])
    plt.show()


point = 65
day = 5
flowAtPoint = flowArray[:, :, point]

array = []

for i in range (0, len (flowAtPoint[5, 0:288, :]) - 5):
    array.append(np.mean(flowAtPoint[5, i:i+5, :]))


plt.plot(flowAtPoint[5, 0:288, :])
plt.show()
'''
#print(np.shape(flowArray))
#a=[1]*31
#for i in range(0, 10):
    #plt.scatter(a, flowArray[:, 100, 60+i])
    #plt.show()
    
    
    
    
