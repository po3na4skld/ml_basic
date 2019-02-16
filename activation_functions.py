import numpy as np


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a, z


def tanh(z):
    a = np.tanh(z)
    return a, z


def relu(z):
    a = np.maximum(0, z)
    return a, z


def sigmoid_gradient(dA, z):
    A, _ = sigmoid(z)
    dZ = dA * A * (1 - A)
    return dZ


def tanh_gradient(dA, z):
    A, _ = tanh(z)
    dZ = dA * (1 - np.square(A))
    return dZ


def relu_gradient(dA, z):
    A, _ = relu(z)
    dZ = np.multiply(dA, np.int(A > 0))
    return dZ
