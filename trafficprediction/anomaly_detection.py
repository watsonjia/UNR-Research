from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
import statsmodels.api as sm
import analysis_data


# Convert to LSTM for training on input X
def convert_LSTM_X(A):
    A = np.array(A)
    A = A.reshape(A.shape[0], A.shape[1], 1)
    return A


# Convert to LSTM for training on labels Y
def convert_LSTM_Y(B):
    B = np.array(B)
    B = B.reshape(B.shape[0], 1)
    return B


# Find maximum deviation of array
def max_deviation(array, mu):
    dList = [np.round(abs(mu - x), 2) for x in array]
    maxDeviation = np.max(dList)
    return maxDeviation


# Get M-estimator of array Z
def Mestimator(Z):
    MAD = sm.robust.scale.mad(Z)
    if MAD == 0:
        mu = np.median(Z)
    else:
        mu = sm.robust.norms.estimate_location(Z, MAD, norm=None, axis=0, initial=None, maxiter=30, tol=1e-06)
    deviation = max_deviation(Z, mu)
    parameters = (mu, 1.4826 * deviation)
    return parameters


# Populate time segments of size INPUT_SIZE from time series
def populate_time_segments(INPUT_SIZE):
    T = []
    T_halved = []
    fifty_percent = INPUT_SIZE // 2
    bound = fifty_percent // 2
    for i in range(0, MAX_TIMES - int(INPUT_SIZE)):
        T.append(times[i:i + int(INPUT_SIZE)])
        T_halved.append(times[i + bound:i + int(INPUT_SIZE) - bound])

    T = convert_LSTM_X(T)
    T_halved = convert_LSTM_X(T_halved)
    return (
     T, T_halved)


# Get flow data and normalize
flow = analysis_data.sFlow
flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow))

# Time series parameters
location = 45
VALUES_PER_DAY = 288
DAYS = 7
MAX_TIMES = VALUES_PER_DAY * DAYS

# Populate time segments
times = flow[location, 0:MAX_TIMES, :].flatten()
X, X_halved = populate_time_segments(12)

# Generate labelled mean and deviation from mean values
Y_mean = []
Y_deviation = []
for i in range(X.shape[0]):
    mean = np.mean(X_halved[i, :])
    Y_mean.append(mean)
    Y_deviation.append(max_deviation(X_halved[i, :], mean))

Y_mean = convert_LSTM_Y(Y_mean)
Y_deviation = convert_LSTM_Y(Y_deviation)

# Generate labelled median and deviation from median values
Y_median = []
Y_medianDeviation = []
for i in range(X.shape[0]):
    median = np.median(X[i, :])
    Y_median.append(median)
    Y_medianDeviation.append(max_deviation(X_halved[i, :], median))

Y_median = convert_LSTM_Y(Y_median)
Y_medianDeviation = convert_LSTM_Y(Y_medianDeviation)

# Generate labelled M-estimate and deviation from M-estimate values
Y_Mestimate = []
Y_MestimateDeviation = []
for i in range(X.shape[0]):
    MAD = sm.robust.scale.mad(X[i, :])
    if MAD == 0:
        M_estimate = Y_median[i, :]
    else:
        M_estimate = sm.robust.norms.estimate_location((X[i, :]), MAD, norm=None, axis=0, initial=None, maxiter=30, tol=1e-06)
    Y_Mestimate.append(M_estimate)
    Y_MestimateDeviation.append(max_deviation(X_halved[i, :], M_estimate))

Y_Mestimate = convert_LSTM_Y(Y_Mestimate)
Y_MestimateDeviation = convert_LSTM_Y(Y_MestimateDeviation)

# Mean prediction LSTM
meanModel = Sequential()
meanModel.add(LSTM(1, batch_input_shape=(None, None, 1), return_sequences=True))
meanModel.add(LSTM(1, return_sequences=True))
meanModel.add(LSTM(1, return_sequences=False))
meanModel.add(Dense(1))
meanModel.add(Dense(1))
meanModel.add(Dense(1))
meanModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
meanModel.summary()
meanHistory = meanModel.fit(X_halved, Y_mean, epochs=10)
meanModel.save('meanModel.h5')

# Deviation from mean LSTM
deviationModel = Sequential()
deviationModel.add(LSTM(1, batch_input_shape=(None, None, 1), return_sequences=True))
deviationModel.add(LSTM(1, return_sequences=False))
deviationModel.add(Dense(1))
deviationModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
deviationModel.summary()
deviationHistory = deviationModel.fit(X_halved, Y_deviation, epochs=10)
deviationModel.save('deviationModel.h5')

# Deviation from median LSTM
medianDeviationModel = Sequential()
medianDeviationModel.add(LSTM(1, batch_input_shape=(None, None, 1), return_sequences=True))
medianDeviationModel.add(LSTM(1, return_sequences=False))
medianDeviationModel.add(Dense(1))
medianDeviationModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
medianDeviationModel.summary()
medianDeviationHistory = medianDeviationModel.fit(X_halved, Y_medianDeviation, epochs=10)
medianDeviationModel.save('medianDeviationModel.h5')

# Deviation from M-estimator LSTM
MestimateDeviationModel = Sequential()
MestimateDeviationModel.add(LSTM(1, batch_input_shape=(None, None, 1), return_sequences=False))
MestimateDeviationModel.add(Dense(1))
MestimateDeviationModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
MestimateDeviationModel.summary()
MestimateDeviationHistory = MestimateDeviationModel.fit(X_halved, Y_MestimateDeviation, epochs=10)
MestimateDeviationModel.save('MestimateDeviationModel.h5')
