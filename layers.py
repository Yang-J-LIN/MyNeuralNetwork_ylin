# class Layers

import numpy as np
import layer

class Layers:
    def __init__(self, num_input):
        self.Layer_list = []
        self.num_Layers = 0
        self.num_input = num_input
        self.weights = {}
        self.bias = {}
        self.values = {}

    def add(self, layer):
        self.Layer_list.append(layer)
        self.num_Layers = self.num_Layers + 1

    def initialize(self):
        for i in range(self.num_Layers):
            if i == 0:
                self.weights['W' + str(i + 1)] = np.random.ranf([self.Layer_list[i].num_neuron, self.num_input])
            else:
                self.weights['W' + str(i + 1)] = np.random.ranf([self.Layer_list[i].num_neuron, self.Layer_list[i - 1].num_neuron])
            self.bias['b' + str(i + 1)] = 0
            self.values['Z' + str(i + 1)] = 0



