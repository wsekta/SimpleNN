import numpy as np

class Layer:
    weights = None
    bias = None
    act_fun = None

    def __init__(self, in_dim, out_dim, activ_func):
        self.weights = 2 * np.random.random((in_dim, out_dim)) - 1
        self.bias = 2 * np.random.random((out_dim,)) - 1
        self.act_fun = activ_func

    def forward(self, X):
        return self.act_fun(X @ self.weights + self.bias)

    def prop_error(self, error):
        return error @ self.weights.T

    def update(self, error, input, output, learning_rate):
        delta = learning_rate * error * self.act_fun(output, True)
        self.weights += input.T @ delta
        self.bias += np.sum(delta,axis=0)
