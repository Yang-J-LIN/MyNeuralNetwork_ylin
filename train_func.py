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


def backward_propagation(Layers, X, Y, Y_hat, loss_function='cross_entropy'):
    grad_weights = {}
    grad_bias = {}

    #dWL = calc_cost(Y, Y_hat, loss_function)
    dWL = calc_cost_grad(Y, Y_hat)

    print(Layers.values['Z3'], dWL)

    for i in reversed(range(Layers.num_Layers)):
        
        m = Layers.values['Z' + str(i)].shape[1]
        temp = np.dot(Layers.weights['W' + str(i + 1)], Layers.values['Z' + str(i)]) + Layers.bias['b' + str(i + 1)]
        dWL = np.multiply(dWL, activation_func.calc_gradient(temp, Layers.Layer_list[i].activation_func))
        grad_weights['dW' + str(i + 1)] = np.dot(dWL, Layers.values['Z' + str(i)].T) / m
        grad_bias['db' + str(i + 1)] = np.sum(dWL, axis=1, keepdims=True) / m
        dWL = np.dot(Layers.weights['W' + str(i + 1)].T, dWL)
    
    return grad_weights, grad_bias


def update_parameters(Layers, grad_weights, grad_bias, learning_rate):
    for i in range(Layers.num_Layers):
        Layers.weights['W' + str(i + 1)] = Layers.weights['W' + str(i + 1)] - learning_rate * grad_weights['dW' + str(i + 1)]
        Layers.bias['b' + str(i + 1)] = Layers.bias['b' + str(i + 1)] - learning_rate * grad_bias['db' + str(i + 1)]
    return


def calc_cost(Y, Y_hat, loss_function='cross_entropy', regularization=None, lambd = 0):
    cost = 0
    if loss_function == 'cross_entropy':
#        cost = - (np.dot(Y, np.log(Y_hat.T)) + np.dot((1 - Y), np.log((1 - Y_hat).T))) / Y.shape[1]
        cost =  - np.sum(Y - Y_hat) / 2
        # print(cost)
    else:
        cost =  - abs(np.sum(Y - Y_hat) / 2)
        #print(cost)
    cost = np.squeeze(cost)
    return cost

def calc_cost_grad(Y, Y_hat, loss_function='cross_entropy', regularization=None, lambd = 0):
    cost_grad = 0
    if loss_function == 'cross_entropy':
        cost_grad = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    else:
        pass
    return cost_grad

def train(X, Y, Layers, loss_function='cross_entropy', learning_rate = 0.005, num_iteration=10000 ,regularization=None, lambd=0):
    Layers.values['Z0'] = X
    for i in range(num_iteration):
        Y_hat = forward_propagation(X, Layers)
        grad_weights, grad_bias = backward_propagation(Layers, X, Y, Y_hat)
        update_parameters(Layers, grad_weights, grad_bias, learning_rate)
        #print(Layers.values['Z3'])
    return



""" X = np.array([[1,2,3], [4,5,6]]).T / 10

Y = np.array([[0.4], [0.6]]).T
a = layers.Layers(3)
a.add(layer.Layer(10))
a.add(layer.Layer(5))
a.add(layer.Layer(1, 'sigmoid'))
a.initialize()

train(X, Y, a) """