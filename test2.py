import simple_nn as snn
import numpy as np
from sklearn import datasets

net = snn.Neural_net()
net.add_layer(snn.Layers.Layer(3, 7, snn.sigm))
net.add_layer(snn.Layers.Layer(7, 1, snn.sigm))

X = np.array([[0,0,0],
	[0,0,1],
	[0,1,0],
	[0,1,1],
	[1,0,0],
	[1,0,1],
	[1,1,0],
	[1,1,1]])
y = np.array([[0],
	[1],
	[1],
	[0],
	[1],
	[0],
	[0],
	[1]])


for i in range(100):
    net.train(X, y, epoch=10000, batch_size=121,learning_rate=0.01)
    net.test(X, y)