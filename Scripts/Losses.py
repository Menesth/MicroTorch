from MicroTensor import Mtensor 

class MSELoss():
    def __init__(self):
        pass

    def __call__(self, x, y):
        self.out = Mtensor(0.0)
        for row_x in x:
            for xi, yi in zip(row_x, y):
                self.out += (yi - xi) * (yi - xi)
        return self.out
    
    def backward(self):
        self.out.backward()