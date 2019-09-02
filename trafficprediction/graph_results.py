import numpy as np
import matplotlib.pyplot as plt
import test_outlier_detection

xvals = []
precision = []
recall = []
fMeasure = []

points = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25]

for i in points:
    P, R, F = test_outlier_detection.anomaly_detection(2, 2, i)
    xvals.append(i)
    precision.append(P)
    recall.append(R)
    fMeasure.append(F)

xvals = np.array(xvals)
precision = np.array(precision)
recall = np.array(recall)
fMeasure = np.array(fMeasure)

plt.plot(xvals, precision, 'g', label='Precision')
plt.plot(xvals, recall, 'r', label='Recall')
plt.plot(xvals, fMeasure, 'b', label='F-measure')
plt.title('Statistical Metrics\nVarying Gamma with Alpha and Beta Fixed at 2.0')
plt.xlabel('Gamma value')
plt.ylabel('Metric value')
plt.legend()
plt.show()
