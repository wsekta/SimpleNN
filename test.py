import simple_nn as snn
import numpy as np
from sklearn import datasets

net = snn.Neural_net()
net.add_layer(snn.Layers.Layer(4, 7, snn.sigm))
net.add_layer(snn.Layers.Layer(7, 3, snn.sigm))

iris = datasets.load_iris()
X = iris.data
y = iris.target
b = np.zeros((y.size, 3))
b[np.arange(y.size), y] = 1
y = b

a = np.arange(y.shape[0])
np.random.shuffle(a)

X_train = (X[a])[:120]
y_train = (y[a])[:120]
X_test = (X[a])[120:]
y_test = (y[a])[120:]

net.test(X_test, y_test)
for i in range(100):
    net.train(X_train, y_train, epoch=10000, batch_size=121,learning_rate=0.01)
    net.test(X_test, y_test)