import numpy as np


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()