import math

#Data structure to store the scalar value and its gradient
class Value:
    #the value of the number, from whom it is created and what operation created it
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, o): # self + other
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            o.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
        
    def __mul__(self, o): # self * other
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o), '*')

        def _backward():
            self.grad += o.data * out.grad
            o.grad += self.data * out.grad
        out._backward = _backward
        
        return out


    def __pow__(self, o): # self ** other
        assert isinstance(o, (int,float)), "only supporting int/float powers for now"
        out = Value(self.data**o, (self, ), f'**{o}')

        def _backward():
            self.grad += o * (self.data ** (o - 1)) * out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, o): # other ** self
        return self * o

    def __truediv__(self,o): # self / other
        return self * o**-1

    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self,o): # self - other
        return self + (-o)


    def __radd__(self,o): # other + self
        return self + o
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1)/(math.exp(2 * x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v._prev:
              build_topo(child)
            topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
          node._backward()
          