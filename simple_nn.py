import numpy as np
import Layers


def lu(x, deriv=False):
    if (deriv == True):
        return 1
    return x


def relu(x, deriv=False):
    if (deriv == True):
        return 1 * (x > 0)
    return x * (x > 0)


def sigm(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class Neural_net:
    layers = []

    def __init__(self, seed=True):
        if (seed == True):
            np.random.seed(1)

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        l = [X]
        for i in range(len(self.layers)):
            l.append(self.layers[i].forward(l[i]))
        return l[len(l) - 1]

    def split_into_batch(self, X, y, size):
        for i in range((X.shape[0] + size - 1) // size):
            yield (X[i * size:(i + 1) * size], y[i * size:(i + 1) * size])

    def train(self, X, y, epoch=1, batch_size=100, learning_rate=0.01):
        for e in range(epoch):
            mixer = np.arange(X.shape[0])
            np.random.shuffle(mixer)
            X_tmp = X[mixer]
            y_tmp = y[mixer]
            for (X_train, y_train) in self.split_into_batch(X_tmp, y_tmp, batch_size):
                l = [X_train]
                for i in range(len(self.layers)):
                    l.append(self.layers[i].forward(l[i]))
                l_error = [y_train - l[len(l) - 1]]
                for i in range(len(self.layers)):
                    l_error.append(self.layers[len(self.layers) - 1 - i].prop_error(l_error[i]))
                    self.layers[len(self.layers) - 1 - i].update(l_error[i], l[len(l) - 2 - i], l[len(l) - 1 - i],
                                                                 learning_rate)

    def test(self, X, y, do_print=True):
        error = np.mean(np.sum((y - self.predict(X)) * (y - self.predict(X)), axis=1))
        if (do_print):
            print(error)
        return error
