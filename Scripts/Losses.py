from Scripts.MicroTensor import Mtensor 

class MSELoss():
    def __init__(self):
        pass

    def __call__(self, yhat, y):
        self.out = sum((yi - yhati)**2 for yi, yhati in zip(y, yhat))
        return self.out
    
    def backward(self):
        self.out.backward()