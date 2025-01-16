from Scripts.MicroTensor import Mtensor

class GradientDescent():

    def __init__(self, params, lr = 1e-3):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0

    def step(self):
        for param in self.params:
            param.val -= self.lr * param.grad