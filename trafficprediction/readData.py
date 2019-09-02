import numpy as np
import csv


days = 31
flowList =  [[], [], [], [], [],[], [], [], [], [],[], [], [], [], [],[], [], [], [], [],[], [], [], [], [],[], [], [], [], [],[]]
flow = [[], [], [], [], [],[], [], [], [], [],[], [], [], [], [],[], [], [], [], [],[], [], [], [], [],[], [], [], [], [],[]]
time = []
postMile = []
lanes = []


#print(flow)

for i in range (0, days):
    flowFile = "Data/Oct2017_Flow/pems_output_"  + str(i) + ".csv"
    csvFile = open (flowFile, "r")
    #print (flow)
    csvReader = csv.reader(csvFile)
    next(csvReader)
    for row in csvReader:
        flowList[i].append(row) 



for i in range(0, len(flowList[15])):
	#print("##################")
	#flowList[16][i][1]
	#flowList[15][i][1]
	if (flowList[16][i][1] != flowList[15][i][1]):
		flowList[16].insert(i, [flowList[15][i][0], flowList[15][i][1], flowList[15][i][2], flowList[15][i][3], flowList[15][i][4], flowList[15][i][5], flowList[15][i][6]])
	if (flowList[17][i][1] != flowList[15][i][1]):
		flowList[17].insert(i, [flowList[18][i][0], flowList[18][i][1], flowList[18][i][2], flowList[18][i][3], flowList[18][i][4], flowList[18][i][5], flowList[18][i][6]])

'''	
for i in range(0, days):
	print("################")
	print(i)
	print(len(flowList[i]))	
'''
for i in range(0, days):
	for row in flowList[i]:
		flow[i].append(float(row[4]))

for row in flowList[0]:
    time.append(row[0])
    postMile.append(float(row[1]))
    lanes.append(int(row[5]))

'''
for i in range(0, days):
    print("############")
    print(len(flowList[i]))
    for j in range(0, len(flowList[0])):
        if (len(flowList[i][j]) != 7):
            print("####")
            print(flowList[i][j])
            print(i)
            print(j)

for i in range(0, days):
    for j in range(i, days):
        if (len(flowList[i]) != len(flowList[j])):
            print("####")
            print(i)
            print(j)
        for k in range(0, len(flowList[0])):
            if (flowList[i][k][0] != flowList[j][k][0]):
                print("########################")
                print(flowList[i][k][0])
                print(flowList[j][k][0])
                print(i)
                print(j)
                print(k)

'''


        
np.asarray(flow)
np.asarray(lanes)
np.asarray(postMile)



