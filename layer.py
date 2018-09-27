import numpy as np

class Layer:

    def __init__(self):
        self.num_neuron = 1
        self.activation_func = "relu"
        self.dropout_prob = 1

    def __init__(self, num_neuron, activation_func="relu", dropout_prob=1):
        self.num_neuron = num_neuron
        self.activation_func = activation_func
        self.dropout_prob = dropout_prob


