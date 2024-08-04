import random

from engine import Value
class Neuron:
    def __init__(self, nin, actFun='tanh'): #nin= number of inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.actFun = actFun

    def __call__(self,x):
        # w = x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

        out = 0
        if self.actFun == 'tanh':
            out = act.tanh()
        elif self.actFun == 'sigmoid':
            out = act.sigmoid()
        elif self.actFun == 'ReLU':
            out = act.ReLU()
        else:
            raise ValueError("Invalid activation function")

        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self,nin,nout,act='tanh'):
        self.act = act
        self.neurons = [Neuron(nin,act) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
class MLP: #multi layer perceptron
    def __init__(self,nin,nouts,act='tanh'):
        self.act = act
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1],act) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
