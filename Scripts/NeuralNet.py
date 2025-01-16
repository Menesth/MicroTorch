from Scripts.MicroTensor import Mtensor
import random

class Neuron:

    def __init__(self, in_features, bias = True):
        self.weights = [Mtensor(random.gauss(0.0, 1.0)) for _ in range(in_features)]
        if bias:
            self.bias = Mtensor(random.uniform(-0.1, 0.1))
        else:
            self.bias = 0.0

    def __call__(self, x):
        if isinstance(x, float):
            lin_comb = self.weights[0] * x + self.bias
        else:
            lin_comb = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return lin_comb

    def parameters(self):
        return self.weights + [self.bias] if self.bias else self.weights
    
class ReLUNeuron:

    def __init__(self, in_features, bias = True):
        self.weights = [Mtensor(random.gauss(0.0, 1.0)) for _ in range(in_features)]
        if bias:
            self.bias = Mtensor(random.uniform(-0.1, 0.1))
        else:
            self.bias = 0.0

    def __call__(self, x):
        if isinstance(x, float):
            lin_comb = self.weights[0] * x + self.bias
        else:
            lin_comb = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return lin_comb.ReLU()

    def parameters(self):
        return self.weights + [self.bias] if self.bias else self.weights

class Linear:

    def __init__(self, in_features, out_features, bias = True):
        self.out_features = out_features
        self.layer = [Neuron(in_features, bias = bias) for _ in range(out_features)]

    def __call__(self, x):
        if self.out_features > 1:
            return [[neuron(xi) for neuron in self.layer] for xi in x]
        return [neuron(xi) for neuron in self.layer for xi in x]
    
    def parameters(self):
        return [param for neuron in self.layer for param in neuron.parameters()]    
    
class LinearReLU:

    def __init__(self, in_features, out_features, bias = True):
        self.out_features = out_features
        self.layer = [ReLUNeuron(in_features, bias = bias) for _ in range(out_features)]

    def __call__(self, x):
        if self.out_features > 1:
            return [[neuron(xi) for neuron in self.layer] for xi in x]
        return [neuron(xi) for neuron in self.layer for xi in x]
    
    def parameters(self):
        return [param for neuron in self.layer for param in neuron.parameters()]

class Sequential:

    def __init__(self, layers):
        self.layers = [layer for layer in layers]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
