import simple_nn as snn
import numpy as np


net = snn.Neural_net()
net.add_layer(snn.Layers.Layer(2, 5, snn.relu))
net.add_layer(snn.Layers.Layer(5, 3, snn.relu))
net.add_layer(snn.Layers.Layer(3, 1, snn.lu))

X = np.random.random((10000,2))
y = np.array([np.sinc(X.T[0])+np.cos(X.T[1])]).T

a = np.arange(y.shape[0])
np.random.shuffle(a)

X_train = (X[a])[:9000]
y_train = (y[a])[:9000]
X_test = (X[a])[9000:]
y_test = (y[a])[9000:]

net.test(X_test, y_test)
for i in range(100):
    net.train(X_train, y_train, batch_size=1000,learning_rate=0.0001)
    net.test(X_test, y_test)