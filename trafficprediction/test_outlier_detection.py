from keras.models import load_model
import numpy as np
import statsmodels.api as sm
import analysis_data


def convert_LSTM_X(A):
    A = np.array(A)
    A = A.reshape(A.shape[0], A.shape[1], 1)
    return A


def convert_LSTM_Y(B):
    B = np.array(B)
    B = B.reshape(B.shape[0], 1)
    return B


def populate_time_segments(INPUT_SIZE):
    T = []
    for i in range(0, MAX_TIMES - int(INPUT_SIZE)):
        T.append(times[i:i + int(INPUT_SIZE)])

    T = convert_LSTM_X(T)
    return T


def outlier_data(INPUT_SIZE):
    T_outlier = []
    for i in range(0, MAX_TIMES - int(INPUT_SIZE)):
        segment = np.copy(times[i:i + int(INPUT_SIZE)])
        deviation = np.std(segment)

        segment.sort()

        segment[0] = segment[0] - LOWER_BOUND * deviation
        segment[segment.size - 1] = segment[(segment.size - 1)] + LOWER_BOUND * deviation

        T_outlier.append(segment)

    T_outlier = convert_LSTM_X(T_outlier)
    return T_outlier


def outlier_test(param, xi, m, dm):
    return m - param * dm > xi or m + param * dm < xi


# Get flow data
flow = analysis_data.sFlow

# Flow parameters
location = 45
VALUES_PER_DAY = 288
DAYS = 7
MAX_TIMES = VALUES_PER_DAY * DAYS
times = flow[location, 0:MAX_TIMES, :].flatten()

# Outlier generation bound
LOWER_BOUND = 3.0

# Load trained LSTM RNNs
meanModel = load_model('meanModel.h5')
deviationModel = load_model('deviationModel.h5')
medianDeviationModel = load_model('medianDeviationModel.h5')
MestimateDeviationModel = load_model('MestimateDeviationModel.h5')


def anomaly_detection(ALPHA, BETA, GAMMA):
    # Get time series data with outliers and normalize
    X_outlier = outlier_data(12)
    X_outlier = (X_outlier - np.min(X_outlier)) / (np.max(X_outlier) - np.min(X_outlier))

    # Counters for outlier statistics
    truePositives = 0
    falseNegatives = 0
    falsePositives = 0

    # Go through each time segment in the time series containing outliers in each segment
    for i in range(X_outlier.shape[0]):
        segment = X_outlier[i, :]

        # Check if each element of the segment is an outlier (by default, outliers at edges of array)
        for j in range(X_outlier.shape[1]):
            xi = segment[j]
            fifty_percent = segment.size // 2
            bound = fifty_percent // 2

            inputSegment = [segment[fifty_percent - bound:fifty_percent + bound]]
            inputSegment = np.array(inputSegment)

            mean = meanModel.predict_on_batch(inputSegment)
            deviation = deviationModel.predict_on_batch(inputSegment)
            isMeanOutlier = outlier_test(ALPHA, xi, mean, deviation)

            median = np.median(segment)
            medianDeviation = medianDeviationModel.predict_on_batch(inputSegment)
            isMedianOutlier = outlier_test(BETA, xi, median, medianDeviation)

            MAD = sm.robust.scale.mad(segment)
            if MAD == 0:
                Mestimate = median
            else:
                Mestimate = sm.robust.norms.estimate_location(segment, MAD, norm=None, axis=0, initial=None, maxiter=30,
                  tol=1e-06)
            MestimateDeviation = MestimateDeviationModel.predict_on_batch(inputSegment)
            isMestimateOutlier = outlier_test(GAMMA, xi, Mestimate, MestimateDeviation)

            isOutlier = isMeanOutlier and (isMedianOutlier or isMestimateOutlier) or isMedianOutlier and isMestimateOutlier
            if j == 0 or j == X_outlier.shape[1] - 1:
                if isOutlier:
                    truePositives += 1
                else:
                    falseNegatives += 1
            elif isOutlier:
                falsePositives += 1

    # Calculate outlier statistics
    precision = truePositives / (truePositives + falsePositives)
    recall = truePositives / (truePositives + falseNegatives)
    fMeasure = 2 * precision * recall / (precision + recall)

    return precision, recall, fMeasure
