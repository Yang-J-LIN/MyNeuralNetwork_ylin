import numpy as np
import layers
import layer
import activation_func

def forward_propagation(X, Layers):
    num = Layers.num_Layers
    temp = X

    for i in range(num):

        temp = activation_func.calc_activation( \
        np.dot(Layers.weights['W' + str(i + 1)], temp) + Layers.bias['b' + str(i + 1)], \
        Layers.Layer_list[i].activation_func)
        Layers.values['Z' + str(i + 1)] = temp

    return Layers.values['Z' + str(num)]

""" X = np.array([1,2,3])
a = layers.Layers(3)
a.add(layer.Layer(4))
a.add(layer.Layer(5))
a.initialize()

print(forward_propagation(X, a))

input() """
