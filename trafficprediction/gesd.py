import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import t as student_t
import statsmodels.api as sm
from itertools import groupby
from operator import itemgetter
import csv
import analysis_data


sFlow = analysis_data.sFlow

def Deviation(array, mu):
    dList = [round(abs(mu-i), 2) for i in array]
    deviation = np.median(dList)
    return deviation
    



def MeanStatistics(X):
    mu = np.mean(X)
    deviation =np.sqrt(np.sum([(i-mu)*(i-mu) for i in X])/(len(X) - 1))
    return (mu, deviation)
    


    
def FindScore(X, parameters):
    score =  (max([abs (i - parameters[0]) for i in X]))/parameters[1]
    return score
    
def RemoveValue(X, parameters):
    d = [abs(i - parameters[0]) for i in X]
    elements = sorted(list(zip(X, d)), key= lambda l: l[1])
    mdi = X.index(elements[-1][0])
    X = X[:mdi] + X[mdi + 1:]
    return X
    
def FindLambda(alpha, n, i):
    i = i+1
    p = 1 - alpha/(2*(n-i+1))
    t = student_t.ppf(p, (n - i - 1))
    lam = t * (n - i) / float(np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
   
    return lam


    

def GESD (X, alpha, k):
    mn_o = int(round(k*len(X)))
    R = []
    lamda = []
    n = len(X)
    for i in range (0, mn_o):
        parameters = MeanStatistics(X)
        score = FindScore (X, parameters)
        R.append(score)
        X = RemoveValue(X, parameters)
        lam = FindLambda (alpha, n, i)
        lamda.append(lam)
    diff = [j - k for j, k in zip (R, lamda)]
    P = [diff.index(k) for k in diff if k > 0]
    if (P == []):
        n_o = 0
    else:
        n_o = max(P) + 1
    
    return n_o



