from math import exp, log

class Mtensor():
    def __init__(self, val, _children=()):
        self.val = val
        self.grad = 0
        self._backward = lambda: None
        self._children = set(_children)

    def __repr__(self):
        return f"Mtensor({self.val}, grad = {self.grad})" if self.grad else f"Mtensor({self.val})"

    def __add__(self, y):
        out = Mtensor(val = self.val + y.val, _children = (self, y))

        def _backward():
            self.grad += 1.0 * out.grad
            y.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, y):
        out = Mtensor(val = self.val - y.val, _children = (self, y))

        def _backward():
            self.grad += 1.0 * out.grad
            y.grad += -1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, y):
        out = Mtensor(val = self.val * y.val, _children = (self, y))

        def _backward():
            self.grad += y.val * out.grad
            y.grad += self.val * out.grad
        out._backward = _backward
        return out

    def ReLU(self):
        relu = self.val if self.val > 0 else 0
        out = Mtensor(val = relu, _children = (self, ))

        def _backward():
            if relu > 0:
                self.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def comp_graph(self):
        graph = []
        visited = set()
        def build_graph(x):
            if x not in visited:
                visited.add(x)
                for child in x._children:
                    build_graph(child)
                graph.append(x)
        build_graph(self)
        return graph[::-1]

    def backward(self):
        self.grad = 1.0
        comp_graph = self.comp_graph()
        for elmt in comp_graph:
            elmt._backward()