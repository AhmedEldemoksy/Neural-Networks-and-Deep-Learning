import numpy as np
def binary_step(x):
    return np.where(x >= 0, 1, 0)

def linear(x):
    return x
print('binary_step',binary_step(0.5))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
print('Sigmoid',sigmoid(0))


def relu(x):
    return np.maximum(0, x)
print('relu',relu([-1, 0, 2]))


def tanh(x):
    return np.tanh(x)
print('tanh',tanh(0.2))


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability improvement
    return exp_x / exp_x.sum(axis=0)
print('softmax',softmax([2,1,0]))