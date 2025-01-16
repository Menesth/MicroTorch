class Mtensor():
    def __init__(self, val, _children=()):
        self.val = val
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(_children)

    def __repr__(self):
        return f"Mtensor({self.val}, grad = {self.grad})" if self.grad else f"Mtensor({self.val})"

    def __add__(self, y):
        y = y if isinstance(y, Mtensor) else Mtensor(y)
        out = Mtensor(val = self.val + y.val, _children = (self, y))
        def _backward():
            self.grad += 1.0 * out.grad
            y.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __sub__(self, y):
        y = y if isinstance(y, Mtensor) else Mtensor(y)
        out = Mtensor(val = self.val - y.val, _children = (self, y))
        def _backward():
            self.grad += 1.0 * out.grad
            y.grad += -1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, y):
        y = y if isinstance(y, Mtensor) else Mtensor(y)
        out = Mtensor(val = self.val * y.val, _children = (self, y))
        def _backward():
            self.grad += y.val * out.grad
            y.grad += self.val * out.grad
        out._backward = _backward

        return out

    def __pow__(self, y):
        out = Mtensor(self.val**y, (self,))
        def _backward():
            self.grad += (y * self.val**(y-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * (-1)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def ReLU(self):
        out = Mtensor(0 if self.val  < 0 else self.val, (self,),)
        def _backward():
            self.grad += (out.val > 0) * out.grad
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
