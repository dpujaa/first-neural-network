import numpy as np

#1) Building a Neuron

def sigmoid(x):
    # activation function is f(x)=1/(1+e^-x)
    return 1/(1+ np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights=weights
        self.bias=bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    

#assigning the weights and bias for the neuron (example)
weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4

# creating an instance of the Neuron class
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3 --- example inputs
# print(n.feedforward(x))     # should print the output of the neuron after applying the sigmoid activation function = 0.9990889488055994




#2) Combining Neurons into a Neural Network

class OurNeuralNetwork:
#     A neural network with:
#     - 2 inputs
#     - a hidden layer with 2 neurons (h1, h2)
#     - an output layer with 1 neuron (o1)
#   Each neuron has the same weights and bias:
#     - w = [0, 1]
#     - b = 0
    
    def __init__(self):
        weights = np.array ([0,1])
        bias = 0

        self.h1= Neuron(weights, bias)
        self.h2= Neuron(weights, bias)
        self.o1= Neuron(weights, bias)

    def feedforward(self, x):
        out_h1= self.h1.feedforward(x)
        out_h2= self.h2.feedforward(x)

        out_o1= self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1
    

# creating an instance of the OurNeuralNetwork class

network= OurNeuralNetwork()
x= np.array([2,3])
print(network.feedforward(x))  

# should print the output of the neural network after applying the sigmoid activation function = 0.7216325609518421

