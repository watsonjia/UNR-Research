import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    """

    :param Z:
    :return:
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

"""
def identity(Z):

    A = Z
    cache = Z
    return A, cache
"""

def relu(Z):
    """

    :param Z:
    :return:
    """

    A = np.maximum(0, Z)
    cache = Z

    return A, cache


def sigmoid_backward(dA, Z):
    """

    :param dA:
    :param Z:
    :return:
    """

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def relu_backward(dA, Z):
    """

    :param dA:
    :param Z:
    :return:
    """

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

"""
def identity_backward(dA, Z):

    dZ = np.array(dA, copy=True)

    return dZ
"""

def initialize_parameters_deep(layer_dims):
    """

    :param layer_dims: python array containing the dimensions of each layer in the NN
    :return: parameters
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """

    :param A: activations from previous layer
    :param W: weights matrix of numpy shape (size of current layer, size of previous layer)
    :param b: bias vector of numpy shape (size of current layer, 1)
    :return: Z (the input of the activation function)
             cache (a python tuple containing A, W, b for computing the backwards pass efficiently)
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    assert (Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """

    :param A_prev: activations from previous layer
    :param W: weights matrix of numpy shape (size of current layer, size of previous layer)
    :param b: bias vector of numpy shape (size of current layer, 1)
    :param activation: activation to be used in this layer stored as a string "relu" or "sigmoid"
    :return: A (the output of the activation function, or post-activation value)
             cache (a python tuple containing "linear_cache" and "activation_cache" to compute backward pass efficiently
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """

    :param X: data, numpy array of shape (input size, number of examples)
    :param parameters: output of initialize_parameters_deep
    :return: AL (last post-activation value)
             caches (list of every cache containing every cache of linear_activation_forward
    """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(l + 1)], parameters["b" + str(l + 1)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """

    :param AL: probability vector corresponding to label predictions, with shape (1, number of examples)
    :param Y: the "true" label vector with shape (1, number of examples)
    :return:
    """

    m = Y.shape[1]

    #cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))) / m
    cost = np.sqrt(((np.sum((AL - Y) * (AL - Y))) / m))

    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """

    :param dZ:
    :param cache:
    :return:
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """

    :param dA:
    :param cache:
    :param activation:
    :return:
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """

    :param AL: probability vector containing the output of the forward propagation
    :param Y: the label vector
    :param caches: list of caches containing every cache of linear_activation_forward containing "relu" and
                   the cache of linear_activation_forward containing "identity"
    :return: grads (a dictionary containing grads["dA" + str(l)], grads["dW" + str(l)], grads["db" + str(l)])
    """

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #dAL = - (np.divide(np.sqrt(np.divide((AL - Y) * (AL - Y), m)), AL - Y))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_prev_temp, db_prev_temp = \
            linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_prev_temp
        grads["db" + str(l + 1)] = db_prev_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """

    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    """

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * grads["dW" + str(l + 1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate * grads["db" + str(l + 1)])

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost):
    """

    :param X:
    :param Y:
    :param layers_dims:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :return:
    """

    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range (0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def predict(X, Y, parameters):

    AL, cache = L_model_forward(X, parameters)

    return AL


def averageError(pred, Y):
    m = Y.shape[1]
    error = np.sum(np.absolute(pred - Y))

    return error / m
