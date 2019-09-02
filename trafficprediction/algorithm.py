import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import csv
import predictionData
import functions

#layers_dims = [5, 20, 7, 5, 1] 
#layers_dims = [20, 25, 25, 1] 

def L_layer_model(X, Y, layers_dims, learning_rate = 0.5, num_iterations = 10000, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                  
    
    parameters = functions.initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):

        AL, caches = functions.L_model_forward(X, parameters)
        
        cost = functions.compute_cost(AL, Y)
    
        grads = functions.L_model_backward(AL, Y, caches)
 
        parameters = functions.update_parameters(parameters, grads, learning_rate)
                
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    #plt.ylabel('cost')
    #plt.xlabel('iterations (per tens)')
    #plt.title("Learning rate =" + str(learning_rate))
    #plt.show()
    
    return parameters

layers_dims = [20, 25,  25, 1] 



outList = []
def CostList(parameters):
    costList = []
    averageErrorList = []
    for i in range(0 + predictionData.p, 136 - predictionData.p):
        X, Y, Y_List = predictionData.DataSet(predictionData.postMile[i])
        pred_test, cost = functions.predict(X[:, 4000:4600], Y[:, 4000:4600], parameters)
        averageError = functions.averageError(pred_test, Y[:, 4000:4600])
        costList.append(cost)
        averageErrorList.append(averageError)

    outList.append(costList)
    outList.append(averageErrorList)

def Algorithm(inputData, outputData, outputList, parameters, param):

    train_x = inputData[:, 0:4000]
    train_y = outputData[:, 0:4000]
    test_x =  inputData[:, 4000:4600]
    test_y =  outputData[:, 4000:4600]
    test_y_List = outputList 

    if (param == True):
        startTime = time.time()
        parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 10000, print_cost = True)
        endTime = time.time()
    t1 = time.time()
    pred_test, cost = functions.predict(test_x, test_y, parameters)
    t2 = time.time()
    avgError = functions.averageError(pred_test, test_y)
    outList.append(pred_test[0])
    outList.append(test_y[0])
    outList.append([cost])
    outList.append([avgError])
    if (param == True):
        outList.append([endTime - startTime])
    outList.append([t2 -t1])

    costList = []
    averageErrorList =[]

    for j in range(0, np.shape(test_y_List)[0]):
        pred_test, cost = functions.predict(test_x, test_y_List[j][:, 4000:4600], parameters)
        test_x = np.concatenate((test_x[1:6, :], pred_test, test_x[6: , :]), 0)
        costList.append(cost)
        averageError = functions.averageError(pred_test, test_y_List[j][:, 4000:4600])
        averageErrorList.append(averageError)
    
    outList.append(costList)
    outList.append(averageErrorList)

listPostMile = [51.72, 42.18, 31.83, 6.62]
listPostMile_1 = [53.57, 43.46, 34.36, 4.48]


for i in range(0, 4):
    inputData, outputData, outputList = predictionData.DataSet(listPostMile [i])
    Algorithm(inputData, outputData, outputList, None, True)


inputData, outputData, outputList = predictionData.DataSet(40.68)
train_x = inputData[:, 0:4000]
train_y = outputData[:, 0:4000]
test_x =  inputData[:, 4000:4600]
test_y =  outputData[:, 4000:4600]
test_y_List = outputList
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 10000, print_cost = True)


for i in range(0, 4):
    inputData, outputData, outputList = predictionData.DataSet(listPostMile_1 [i])
    Algorithm(inputData, outputData, outputList, parameters, False)


CostList (parameters)
print (outList)

'''
postMile_1 = [1.78, 2.43, 6.29, 7.57, 9.95, 12.41, 13.95, 14.3, 29.94, 30.34, 30.54, 31.79, 31.83, 33.9, 34.36, 35.58, 35.78, 37.07, 38.88, 38.96, 39.35, 39.81, 40.18, 42.18, 42.59, 42.77, 44.21, 44.81, 45.76, 46.38, 47.31, 47.87, 48.5, 49.01, 49.33, 50.6, 51.72, 51.97, 52.18, 52.78, 53.31, 53.57, 53.9, 54.37, 55.9, 56.59, 56.59, 57.45, 57.75, 59.49]
Y = []
for i in range(0, 50):
    inputData, outputData, outputList = predictionData.DataSet(postMile_1[i])
    X =  inputData[:, 4000:4288:3]
    test_Y =  outputData[:, 4000:4288:3]
    pred_test, cost = functions.predict(X, test_Y, parameters)
    Y.append(pred_test.tolist())
    
print (Y)
with open("traffic.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(Y)
'''


with open("prediction_results_1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(outList)

'''
learning_rate = 0
for i in range (0, 10):
    learning_rate = learning_rate + 0.2
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations = 10000, print_cost = True)
    pred_test, cost = functions.predict(test_x, test_y, parameters)
    avgError = functions.averageError(pred_test, test_y)
    #print(pred_test)
    print(cost)
    print(np.shape(test_x))
    print(np.shape(pred_test))
    print(np.shape(test_y))
    print(avgError)
'''

'''
costList = []
averageErrorList =[]
for i in range(0, np.shape(test_y_List)[0]):
    print(np.shape(test_y_List[i]))
    pred_test, cost = functions.predict(test_x, test_y_List[i][:, 4000:4600], parameters)
    test_x[0] = pred_test
    print (cost)
    costList.append(cost)
    averageError = functions.averageError(pred_test, test_y_List[i][:, 4000:4600])
    averageErrorList.append(averageError)
    print(averageError)
    
print (costList)
print(averageErrorList)
'''

'''
for i in range(0 + predictionData.p, 136 - predictionData.p):
    X, Y, Y_List = predictionData.DataSet(predictionData.postMile[i])
    print(predictionData.postMile[i])
    print(np.shape(X))
    print(np.shape(Y[:, 4000:4600]))
    pred_test, cost = functions.predict(X[:, 4000:4600], Y[:, 4000:4600], parameters)
    averageError = functions.averageError(pred_test, Y[:, 4000:4600])
    costList.append(cost)
    averageErrorList.append(averageError)
print (costList)
print(averageErrorList)
'''
'''
plt.figure(1)
x = np.arange(0, 500)
plt.plot(x, pred_test[0], 'bs', label='Predicted') 
plt.plot( x, test_y[0], 'g^', label='Actual')
plt.gca().legend(('Predicted','Actual'))
plt.xlabel("Time slot")
plt.ylabel("Normalized traffic")
plt.figure(2)
x = np.arange(0, 500)
plt.plot(pred_test[0]-test_y[0] )
#plt.axis([0, 1000, 0, 0.050])
plt.xlabel("Time slot")
plt.ylabel("Error")
plt.show()
#pred_test = predict(test_x, test_y, parameters)
'''
