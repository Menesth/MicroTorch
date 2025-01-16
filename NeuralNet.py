from MicroTensor import Mtensor
import random

class Linear:
    def __init__(self, in_features, out_features, bias = True):
        self.weights = [Mtensor(random.gauss(0.0, 1.0)) for _ in range(in_features)]
        self.boolean = bias
        if self.boolean:
            self.bias = Mtensor(random.uniform(-0.1, 0.1))
        else:
            self.bias = Mtensor(0.0)

    def __call__(self, x):
        out = list()
        for row in x:
            currout = list()
            for xi_row, wi in zip(row, self.weights):
                currout.append(xi_row * wi + self.bias)
            out.append(currout)
        return out
    
    def parameters(self):
        return self.weights + [self.bias] if self.boolean else self.weights
    
class LinearReLU:
    def __init__(self, in_features, out_features, bias = True):
        self.weights = [Mtensor(random.gauss(0.0, 1.0)) for _ in range(in_features)]
        self.boolean = bias
        if self.boolean:
            self.bias = Mtensor(random.uniform(-0.1, 0.1))
        else:
            self.bias = Mtensor(0.0)
            
    def __call__(self, x):
        out = list()
        for row in x:
            currout = list()
            for xi_row, wi in zip(row, self.weights):
                linearcomb = xi_row * wi + self.bias
                currout.append(linearcomb.ReLU())
            out.append(currout)
        return out

    def parameters(self):
        return self.weights + [self.bias] if self.boolean else self.weights