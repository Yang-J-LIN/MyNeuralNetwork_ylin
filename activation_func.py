# Activation functions

import numpy as np

activation_func_list = ['sigmoid', 'relu']


def calc_activation(Z, func):
    if func == 'sigmoid':
        return sigmoid(Z)
    else:
        return relu(Z)


def calc_gradient(Z, func):
    if func == 'sigmoid':
        return d_sigmoid(Z)
    else:
        return d_relu(Z)

# ------------------ sigmoid function ------------------ #
def sigmoid(Z):
    return 1 / (1 + np.e ** (-Z))


def d_sigmoid(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

# -------------------- relu function -------------------- #

def relu(Z):
    return ((Z > 0) * Z)


def d_relu(Z):
    return (Z > 0)
